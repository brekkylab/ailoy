use std::{fs, path::Path};

use anyhow::Context;

fn main() -> anyhow::Result<()> {
    #[cfg(feature = "python")]
    {
        // `stub_info` is a function defined by `define_stub_info_gatherer!` macro.
        let stub = ailoy::ffi::py::stub_info()?;
        stub.generate()?;

        let stub_filepath = stub.python_root.join("ailoy/_core.pyi");
        // Inject "CacheResultT" TypeVar
        inject_typevar(stub_filepath, "CacheResultT")?;
    }
    Ok(())
}

/// Inject TypeVar declaration after the last import line in a .pyi file
fn inject_typevar<P: AsRef<Path>>(file_path: P, typevar_name: &str) -> anyhow::Result<()> {
    let file_path = file_path.as_ref();

    // Read the file
    let content = fs::read_to_string(file_path)
        .with_context(|| format!("Failed to read file: {}", file_path.display()))?;

    let lines: Vec<&str> = content.lines().collect();

    // TypeVar declaration to inject
    let typevar_declaration = format!("{} = typing.TypeVar(\"{}\")", typevar_name, typevar_name);

    // Check if TypeVar already exists
    if lines
        .iter()
        .any(|line| line.trim() == typevar_declaration.trim())
    {
        println!(
            "TypeVar {} already exists in {}",
            typevar_name,
            file_path.display()
        );
        return Ok(());
    }

    // Find the last import line
    let last_import_index = lines
        .iter()
        .enumerate()
        .rev() // Search from the end
        .find(|(_, line)| {
            let trimmed = line.trim();
            trimmed.starts_with("import ") || trimmed.starts_with("from ")
        })
        .map(|(i, _)| i);

    let Some(last_import_index) = last_import_index else {
        anyhow::bail!("No import statements found in {}", file_path.display());
    };

    // Build new content
    let mut new_lines = Vec::with_capacity(lines.len() + 2);

    // Add lines up to and including the last import
    new_lines.extend_from_slice(&lines[..=last_import_index]);

    // Add blank line and TypeVar declaration
    new_lines.push("");
    new_lines.push(&typevar_declaration);

    // Add remaining lines
    new_lines.extend_from_slice(&lines[last_import_index + 1..]);

    // Write back to file
    let new_content = new_lines.join("\n");
    fs::write(file_path, new_content)
        .with_context(|| format!("Failed to write file: {}", file_path.display()))?;

    println!(
        "Injected TypeVar '{}' into {}",
        typevar_name,
        file_path.display()
    );
    Ok(())
}
