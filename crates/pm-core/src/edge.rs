//! Non-maximum suppression, hysteresis, subpixel edge localization, and edge
//! chaining (README §6, §7.1).

use std::collections::HashSet;

use crate::feature::{EdgeChain, EdgePoint, EdgeSet};
use crate::gradient::GradientField;

/// Edge extraction parameters.  Defaults follow README Appendix A.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EdgeParams {
    /// Percentile of the gradient magnitude used for the adaptive high threshold.
    pub high_percentile: f32,
    /// Absolute floor for the high threshold.
    pub high_floor: f32,
    /// Low threshold as a fraction of the high threshold.
    pub low_ratio: f32,
    /// Chains shorter than this many points are discarded.
    pub min_chain_length: usize,
    /// Whether to apply parabola subpixel localization.
    pub subpixel: bool,
}

impl Default for EdgeParams {
    fn default() -> Self {
        Self {
            high_percentile: 0.90,
            high_floor: 0.08,
            low_ratio: 0.4,
            min_chain_length: 4,
            subpixel: true,
        }
    }
}

/// High and low hysteresis thresholds derived from the gradient magnitude.
#[must_use]
pub fn adaptive_thresholds(gradient: &GradientField, params: &EdgeParams) -> (f32, f32) {
    let high = percentile(&gradient.magnitude, params.high_percentile).max(params.high_floor);
    (high * params.low_ratio, high)
}

/// Value at a percentile of `values` in `[0, 1]`, using nearest-rank selection.
#[must_use]
pub fn percentile(values: &[f32], percentile: f32) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    let index = ((values.len() - 1) as f32 * percentile.clamp(0.0, 1.0)).round() as usize;
    let index = index.min(sorted.len() - 1);
    let (_, value, _) = sorted.select_nth_unstable_by(index, f32::total_cmp);
    *value
}

/// Extract subpixel edge points and chains from a gradient field.
#[must_use]
pub fn extract_edges(gradient: &GradientField, params: &EdgeParams) -> EdgeSet {
    let (low, high) = adaptive_thresholds(gradient, params);
    let (width, height) = (gradient.width, gradient.height);
    if width == 0 || height == 0 {
        return EdgeSet::default();
    }

    // 1. Non-maximum suppression along the gradient normal.
    let mut nms = vec![false; width * height];
    for y in 0..height {
        for x in 0..width {
            let m0 = gradient.magnitude_at(x, y);
            if m0 <= 0.0 {
                continue;
            }
            let index = y * width + x;
            let (gx, gy) = (gradient.gx[index], gradient.gy[index]);
            let norm = gx.hypot(gy);
            if norm <= 0.0 {
                continue;
            }
            let (nx, ny) = (gx / norm, gy / norm);
            let xf = x as f32;
            let yf = y as f32;
            let m_minus = gradient.sample_magnitude(xf - nx, yf - ny);
            let m_plus = gradient.sample_magnitude(xf + nx, yf + ny);
            if m0 >= m_minus && m0 >= m_plus && (m0 > m_minus || m0 > m_plus) {
                nms[index] = true;
            }
        }
    }

    // 2. Hysteresis: strong seeds grow through weak 8-connected candidates.
    let mut candidate = vec![false; width * height];
    let mut accepted = vec![false; width * height];
    let mut stack = Vec::new();
    for y in 0..height {
        for x in 0..width {
            let index = y * width + x;
            if !nms[index] {
                continue;
            }
            let magnitude = gradient.magnitude[index];
            if magnitude >= low {
                candidate[index] = true;
            }
            if magnitude >= high {
                accepted[index] = true;
                stack.push((x, y));
            }
        }
    }
    while let Some((x, y)) = stack.pop() {
        for (dx, dy) in NEIGHBORS {
            let nx = x as isize + dx;
            let ny = y as isize + dy;
            if nx < 0 || ny < 0 || nx >= width as isize || ny >= height as isize {
                continue;
            }
            let index = ny as usize * width + nx as usize;
            if candidate[index] && !accepted[index] {
                accepted[index] = true;
                stack.push((nx as usize, ny as usize));
            }
        }
    }

    // 3. Subpixel localization and point records, in deterministic scan order.
    let mut pixels_by_point: Vec<(usize, usize)> = Vec::new();
    let mut points = Vec::new();
    for y in 0..height {
        for x in 0..width {
            let index = y * width + x;
            if !accepted[index] {
                continue;
            }
            let m0 = gradient.magnitude[index];
            let (gx, gy) = (gradient.gx[index], gradient.gy[index]);
            let norm = gx.hypot(gy);
            if norm <= 0.0 {
                continue;
            }
            let (rx, ry) = (gx / norm, gy / norm);

            let mut px = x as f32;
            let mut py = y as f32;
            if params.subpixel {
                let m_minus = gradient.sample_magnitude(px - rx, py - ry);
                let m_plus = gradient.sample_magnitude(px + rx, py + ry);
                let denominator = 2.0 * (m_minus - 2.0 * m0 + m_plus);
                if denominator.abs() > 1e-8 && m0 >= m_minus && m0 >= m_plus {
                    let delta = ((m_minus - m_plus) / denominator).clamp(-0.5, 0.5);
                    px += delta * rx;
                    py += delta * ry;
                }
            }

            let (nx, ny, sign) = gradient
                .normal_at(x, y)
                .expect("accepted edge has a non-zero gradient");
            pixels_by_point.push((x, y));
            points.push(EdgePoint {
                x: px,
                y: py,
                nx,
                ny,
                magnitude: m0,
                sign,
                chain_id: 0,
                weight: m0,
            });
        }
    }

    // 4. 8-connected chains over accepted pixels.
    let pixel_to_point: std::collections::HashMap<(usize, usize), usize> = pixels_by_point
        .iter()
        .enumerate()
        .map(|(point, &pixel)| (pixel, point))
        .collect();
    let pixel_set: HashSet<(usize, usize)> = pixels_by_point.iter().copied().collect();
    let mut visited: HashSet<(usize, usize)> = HashSet::new();
    let mut chains = Vec::new();
    let mut kept_indices: HashSet<usize> = HashSet::new();
    let mut chain_id = 0u32;

    for &start in &pixels_by_point {
        if visited.contains(&start) {
            continue;
        }
        let ordered = order_component(start, &pixel_set, &mut visited);
        if ordered.len() < params.min_chain_length {
            continue;
        }
        let mut chain_points = Vec::with_capacity(ordered.len());
        for pixel in ordered {
            if let Some(&point_index) = pixel_to_point.get(&pixel) {
                points[point_index].chain_id = chain_id;
                chain_points.push(point_index);
                kept_indices.insert(point_index);
            }
        }
        chains.push(EdgeChain {
            id: chain_id,
            points: chain_points,
        });
        chain_id += 1;
    }

    // Discard points belonging to rejected short components and remap indices.
    let mut kept = Vec::with_capacity(kept_indices.len());
    let mut old_to_new = vec![usize::MAX; points.len()];
    for (index, point) in points.iter().enumerate() {
        if kept_indices.contains(&index) {
            old_to_new[index] = kept.len();
            kept.push(*point);
        }
    }
    for chain in &mut chains {
        chain.points = chain
            .points
            .iter()
            .map(|&index| old_to_new[index])
            .collect();
    }

    EdgeSet {
        points: kept,
        chains,
    }
}

const NEIGHBORS: [(isize, isize); 8] = [
    (-1, -1),
    (0, -1),
    (1, -1),
    (-1, 0),
    (1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
];

/// Order one 8-connected component by a greedy nearest-neighbor walk.  The walk
/// starts at an endpoint when one exists and prefers straight continuations.
fn order_component(
    start: (usize, usize),
    pixel_set: &HashSet<(usize, usize)>,
    visited: &mut HashSet<(usize, usize)>,
) -> Vec<(usize, usize)> {
    let mut component = Vec::new();
    let mut frontier = vec![start];
    visited.insert(start);
    while let Some(current) = frontier.pop() {
        component.push(current);
        for (dx, dy) in NEIGHBORS {
            let nx = current.0 as isize + dx;
            let ny = current.1 as isize + dy;
            if nx < 0 || ny < 0 {
                continue;
            }
            let neighbor = (nx as usize, ny as usize);
            if pixel_set.contains(&neighbor) && visited.insert(neighbor) {
                frontier.push(neighbor);
            }
        }
    }

    // Start from an endpoint when possible.
    component.sort_unstable();
    let start_index = component
        .iter()
        .position(|&p| degree(p, pixel_set) <= 1)
        .unwrap_or(0);
    let start = component[start_index];

    let mut in_chain: HashSet<(usize, usize)> = HashSet::new();
    let mut order = Vec::with_capacity(component.len());
    let mut current = start;
    in_chain.insert(current);
    order.push(current);
    let mut direction = (0.0f32, 0.0f32);

    loop {
        let candidates: Vec<(usize, usize)> = NEIGHBORS
            .iter()
            .filter_map(|&(dx, dy)| {
                let nx = current.0 as isize + dx;
                let ny = current.1 as isize + dy;
                if nx < 0 || ny < 0 {
                    return None;
                }
                let candidate = (nx as usize, ny as usize);
                (pixel_set.contains(&candidate) && !in_chain.contains(&candidate))
                    .then_some(candidate)
            })
            .collect();
        let Some(next) = choose_next(current, &candidates, direction) else {
            break;
        };
        direction = (
            next.0 as f32 - current.0 as f32,
            next.1 as f32 - current.1 as f32,
        );
        in_chain.insert(next);
        order.push(next);
        current = next;
    }

    // Branches or a broken greedy walk: append remaining points nearest-first so
    // no accepted edge point is silently dropped.
    while order.len() < component.len() {
        let Some((&nearest, _)) = component
            .iter()
            .filter(|p| !in_chain.contains(p))
            .map(|p| {
                let dx = p.0 as f32 - current.0 as f32;
                let dy = p.1 as f32 - current.1 as f32;
                (p, dx * dx + dy * dy)
            })
            .min_by(|a, b| a.1.total_cmp(&b.1))
        else {
            break;
        };
        in_chain.insert(nearest);
        order.push(nearest);
        current = nearest;
    }

    order
}

fn choose_next(
    current: (usize, usize),
    candidates: &[(usize, usize)],
    direction: (f32, f32),
) -> Option<(usize, usize)> {
    if candidates.is_empty() {
        return None;
    }
    let dir_norm = direction.0.hypot(direction.1);
    if dir_norm <= 0.0 {
        return candidates.iter().copied().min();
    }
    candidates.iter().copied().max_by(|a, b| {
        let score = |p: &(usize, usize)| {
            let dx = p.0 as f32 - current.0 as f32;
            let dy = p.1 as f32 - current.1 as f32;
            let norm = dx.hypot(dy);
            if norm <= 0.0 {
                0.0
            } else {
                (dx * direction.0 + dy * direction.1) / norm
            }
        };
        score(a).total_cmp(&score(b))
    })
}

fn degree(pixel: (usize, usize), pixel_set: &HashSet<(usize, usize)>) -> usize {
    NEIGHBORS
        .iter()
        .filter(|&&(dx, dy)| {
            let nx = pixel.0 as isize + dx;
            let ny = pixel.1 as isize + dy;
            nx >= 0 && ny >= 0 && pixel_set.contains(&(nx as usize, ny as usize))
        })
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gray::GrayF32;

    fn smooth_step(width: usize, height: usize, x0: f32, edge_width: f32) -> GrayF32 {
        let mut data = Vec::with_capacity(width * height);
        for _y in 0..height {
            for x in 0..width {
                let value = 0.5 * (1.0 + ((x as f32 - x0) / edge_width).tanh());
                data.push(value);
            }
        }
        GrayF32::new(width, height, data).unwrap()
    }

    #[test]
    fn subpixel_edge_position_is_accurate() {
        for &x0 in &[9.15f32, 9.5, 10.37, 11.82] {
            let image = smooth_step(24, 8, x0, 1.4);
            let gradient = GradientField::compute(&image);
            let edges = extract_edges(&gradient, &EdgeParams::default());
            let row: Vec<&EdgePoint> = edges
                .points
                .iter()
                .filter(|p| (p.y - 4.0).abs() <= 1.0)
                .collect();
            assert!(!row.is_empty(), "no edge at x0={x0}");
            let mean_x = row.iter().map(|p| p.x).sum::<f32>() / row.len() as f32;
            assert!((mean_x - x0).abs() < 0.12, "x0={x0}, mean_x={mean_x}");
            let point = row[0];
            assert!(point.nx.abs() > 0.95, "normal {point:?}");
            assert_eq!(point.sign, 1);
        }
    }

    #[test]
    fn inverted_step_has_negative_polarity() {
        let image = smooth_step(24, 8, 12.3, 1.4);
        let mut inverted = image.clone();
        for value in &mut inverted.data {
            *value = 1.0 - *value;
        }
        let gradient = GradientField::compute(&inverted);
        let edges = extract_edges(&gradient, &EdgeParams::default());
        let point = edges
            .points
            .iter()
            .find(|p| (p.y - 4.0).abs() <= 1.0)
            .expect("edge");
        assert_eq!(point.sign, -1);
        assert!((point.x - 12.3).abs() < 0.12, "x {}", point.x);
    }

    #[test]
    fn short_chains_are_discarded() {
        let mut image = GrayF32::filled(40, 20, 0.0);
        // A long horizontal line (two edge chains) and a two-pixel dot.
        for x in 10..30 {
            image.data[8 * 40 + x] = 1.0;
        }
        image.data[15 * 40 + 5] = 1.0;
        image.data[15 * 40 + 6] = 1.0;
        let gradient = GradientField::compute(&image);
        let params = EdgeParams {
            min_chain_length: 10,
            ..EdgeParams::default()
        };
        let edges = extract_edges(&gradient, &params);
        assert!(!edges.points.is_empty());
        assert!(
            edges.points.iter().all(|p| !(p.x < 9.0 && p.y > 13.0)),
            "isolated short chain survived"
        );
        assert!(edges.points.iter().any(|p| p.x > 15.0 && p.y < 11.0));
    }

    #[test]
    fn hysteresis_keeps_connected_weak_edges_only() {
        // Synthetic gradient field: a strong vertical edge in the top half, a
        // weak continuation in the bottom half, and a disconnected weak edge.
        let width = 32;
        let height = 40;
        let mut gx = vec![0.0f32; width * height];
        let mut magnitude = vec![0.0f32; width * height];
        for y in 0..20 {
            gx[y * width + 16] = 0.5;
            magnitude[y * width + 16] = 0.5;
        }
        for y in 20..40 {
            gx[y * width + 16] = 0.2;
            magnitude[y * width + 16] = 0.2;
        }
        for y in 25..35 {
            gx[y * width + 4] = 0.2;
            magnitude[y * width + 4] = 0.2;
        }
        let gradient = GradientField {
            width,
            height,
            gx,
            gy: vec![0.0; width * height],
            magnitude,
        };
        let params = EdgeParams {
            high_percentile: 0.9,
            high_floor: 0.3,
            low_ratio: 0.5,
            min_chain_length: 4,
            subpixel: true,
        };
        let edges = extract_edges(&gradient, &params);
        let connected = edges
            .points
            .iter()
            .filter(|p| (p.x - 16.0).abs() < 1.0 && p.y > 22.0 && p.y < 38.0)
            .count();
        assert!(connected >= 10, "connected weak edge count {connected}");
        let disconnected = edges
            .points
            .iter()
            .filter(|p| (p.x - 4.0).abs() < 1.0 && p.y > 26.0 && p.y < 34.0)
            .count();
        assert_eq!(disconnected, 0, "disconnected weak edge survived");
    }

    #[test]
    fn adaptive_threshold_uses_floor() {
        let image = GrayF32::filled(8, 8, 0.4);
        let gradient = GradientField::compute(&image);
        let params = EdgeParams::default();
        let (low, high) = adaptive_thresholds(&gradient, &params);
        assert!((high - params.high_floor).abs() < 1e-6);
        assert!((low - params.high_floor * params.low_ratio).abs() < 1e-6);
    }
}
