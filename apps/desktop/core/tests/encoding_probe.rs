//! The real file that started this, decoded the way the engine decodes it.
//!
//! `#[ignore]`d: it reads a path on the developer's machine, so it is a check to run by
//! hand against a file you have rather than part of the suite.

use std::path::PathBuf;

#[test]
#[ignore]
fn a_korean_csv_off_a_real_disk_reads_as_a_table() {
    let path = PathBuf::from(std::env::var("ENCODING_PROBE").expect("ENCODING_PROBE=<file>"));
    let bytes = std::fs::read(&path).expect("the file");
    assert!(
        String::from_utf8(bytes.clone()).is_err(),
        "this probe is for a file that is not UTF-8"
    );
    let (text, encoding) =
        ailoy_desktop_core::decode_for_tests(bytes, false).expect("it has to read as text");
    println!("encoding: {encoding}");
    println!("first line: {}", text.lines().next().unwrap_or(""));
    assert_eq!(encoding, "CP949");
    assert!(text.contains(','), "a csv has separators in it");
}
