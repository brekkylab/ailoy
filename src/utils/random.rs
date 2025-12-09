pub fn generate_random_hex_string(length: usize) -> anyhow::Result<String> {
    let mut bytes = vec![0u8; length];
    getrandom::fill(&mut bytes).map_err(|e| anyhow::anyhow!(e.to_string()))?;
    Ok(hex::encode(bytes))
}
