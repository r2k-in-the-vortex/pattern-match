//! Scaffold check: the committed Lean-generated golden vectors are readable.
//!
//! Semantic conformance checks arrive with the geometry module (M1).

use std::fs;
use std::path::PathBuf;

#[derive(Debug, serde::Deserialize)]
struct GoldenFile {
    format: String,
    version: u32,
    vectors: Vec<serde_json::Value>,
}

#[test]
fn committed_golden_vectors_are_readable() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("testdata/golden/vectors.json");
    let text =
        fs::read_to_string(&path).unwrap_or_else(|err| panic!("read {}: {err}", path.display()));
    let parsed: GoldenFile =
        serde_json::from_str(&text).unwrap_or_else(|err| panic!("parse {}: {err}", path.display()));

    assert_eq!(parsed.format, "pattern-match-golden");
    assert_eq!(parsed.version, 1);
    assert!(
        parsed.vectors.is_empty(),
        "M0 has no vectors yet; add vectors with M1"
    );
}
