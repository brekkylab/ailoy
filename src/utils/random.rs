#[allow(dead_code)]
pub fn get_random_f64() -> f64 {
    let mut buf = [0u8; 8];
    getrandom::fill(&mut buf).expect("Failed to get random bytes");
    let random_u64 = u64::from_le_bytes(buf);
    let random_f64 = (random_u64 as f64) / (u64::MAX as f64);
    random_f64
}

pub fn generate_random_hex_string(length: usize) -> anyhow::Result<String> {
    let mut bytes = vec![0u8; length];
    getrandom::fill(&mut bytes).map_err(|e| anyhow::anyhow!(e.to_string()))?;
    Ok(hex::encode(bytes))
}
