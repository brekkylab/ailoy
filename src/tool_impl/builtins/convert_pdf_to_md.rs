#[cfg(feature = "sandbox")]
use std::sync::Arc;
use std::{
    io::Read as _,
    path::{Path, PathBuf},
};

use anyhow::Context as _;

#[cfg(feature = "sandbox")]
use crate::tool::ToolFunc;
use crate::{
    datatype::Value,
    message::{ToolDesc, ToolDescBuilder},
    tool::Tool,
};

const TOOL_NAME: &str = "convert_pdf_to_md";
const DOCLING_PACKAGE: &str = "docling>=2,<3";
const CONVERSION_TIMEOUT_SECS: u64 = 600;

const SANDBOX_INPUT_PDF: &str = "/workspace/__ailoy_input.pdf";
const SANDBOX_OUTPUT_MD: &str = "/workspace/__ailoy_output.md";
const SANDBOX_SCRIPT_PATH: &str = "/workspace/__ailoy_docling.py";

const DOCLING_SOURCE: &str = r#"
import logging
import os
from pathlib import Path

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
logging.getLogger("docling").setLevel(logging.CRITICAL)


def build_converter():
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import (
        PdfPipelineOptions,
        TableStructureOptions,
    )
    from docling.document_converter import DocumentConverter, PdfFormatOption

    pipeline_options = PdfPipelineOptions(
        do_ocr=False,
        do_table_structure=True,
        table_structure_options=TableStructureOptions(
            do_cell_matching=True,
            mode="accurate",
        ),
        accelerator_options={"num_threads": 4, "device": "auto"},
        do_picture_classification=False,
        do_picture_description=False,
        do_chart_extraction=False,
        do_code_enrichment=False,
        do_formula_enrichment=False,
        generate_page_images=False,
        generate_picture_images=False,
    )
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
        }
    )


if __name__ == "__main__":
    pdf_path = Path(os.environ["AILOY_PDF_PATH"])
    output_path = Path(os.environ["AILOY_OUTPUT_PATH"])
    markdown = build_converter().convert(pdf_path).document.export_to_markdown()
    output_path.write_text(markdown, encoding="utf-8")
"#;

#[cfg(feature = "sandbox")]
pub async fn build_convert_pdf_to_md_tool(
    sandbox: Arc<crate::sandbox::Sandbox>,
) -> anyhow::Result<Tool> {
    // Install docling once at tool-build time so per-call latency is purely conversion.
    // docling pulls PyTorch and other heavy deps, so we give it the full conversion budget.
    //
    // After docling installs opencv-python (which requires X11/libxcb), replace it with
    // opencv-python-headless so the sandbox doesn't need display system libraries.
    let install = sandbox
        .shell_with_timeout(
            &format!(
                "pip install '{DOCLING_PACKAGE}' \
                 && pip install --force-reinstall --no-deps opencv-python-headless"
            ),
            CONVERSION_TIMEOUT_SECS,
        )
        .await
        .context("failed to run pip install for docling")?;
    if install.timed_out {
        anyhow::bail!("pip install docling timed out after {CONVERSION_TIMEOUT_SECS}s");
    }
    if install.exit_code != 0 {
        anyhow::bail!(
            "failed to pre-install docling (exit {}): {}",
            install.exit_code,
            install.stderr
        );
    }

    // Write the conversion script once; it persists for the session lifetime.
    sandbox
        .write_file(SANDBOX_SCRIPT_PATH, DOCLING_SOURCE.as_bytes())
        .await
        .context("failed to write docling script to sandbox")?;

    let desc = convert_pdf_to_md_tool_desc();

    let f: ToolFunc = ToolFunc::new(move |args: Value| {
        let sandbox = sandbox.clone();
        async move {
            let pdf_path = match validate_pdf_path(&args) {
                Ok(p) => p,
                Err(e) => return e,
            };

            if let Err(e) = sandbox.copy_from_host(&pdf_path, SANDBOX_INPUT_PDF).await {
                return error_value(
                    &pdf_path.to_string_lossy(),
                    "execution",
                    &format!("failed to copy PDF into sandbox: {e}"),
                );
            }

            let cmd = format!(
                "AILOY_PDF_PATH={SANDBOX_INPUT_PDF} AILOY_OUTPUT_PATH={SANDBOX_OUTPUT_MD} \
                 python3 {SANDBOX_SCRIPT_PATH}"
            );
            let result = match sandbox
                .shell_with_timeout(&cmd, CONVERSION_TIMEOUT_SECS)
                .await
            {
                Ok(r) => r,
                Err(e) => {
                    return error_value(
                        &pdf_path.to_string_lossy(),
                        "execution",
                        &format!("sandbox error: {e}"),
                    );
                }
            };

            if result.timed_out {
                return error_value(
                    &pdf_path.to_string_lossy(),
                    "execution",
                    &format!("docling timed out after {CONVERSION_TIMEOUT_SECS}s"),
                );
            }
            if result.exit_code != 0 {
                return error_value(
                    &pdf_path.to_string_lossy(),
                    "execution",
                    &format!(
                        "docling failed (exit {}): {}",
                        result.exit_code,
                        result.stderr.trim()
                    ),
                );
            }

            let markdown = match sandbox.read_file(SANDBOX_OUTPUT_MD).await {
                Ok(md) => md,
                Err(e) => {
                    return error_value(
                        &pdf_path.to_string_lossy(),
                        "execution",
                        &format!("failed to read markdown output from sandbox: {e}"),
                    );
                }
            };

            let size_chars = markdown.chars().count();
            match write_host_temp_file(&markdown) {
                Ok(md_path) => crate::to_value!({
                    "md_path": md_path.to_string_lossy().to_string(),
                    "size_chars": size_chars as i64
                }),
                Err(e) => error_value(
                    &pdf_path.to_string_lossy(),
                    "execution",
                    &format!("failed to write output file: {e}"),
                ),
            }
        }
    });

    Ok(Tool::new(desc, Arc::new(f)))
}

#[cfg(not(feature = "sandbox"))]
pub async fn build_convert_pdf_to_md_tool() -> anyhow::Result<Tool> {
    anyhow::bail!("sandbox feature required for convert_pdf_to_md")
}

fn write_host_temp_file(content: &str) -> anyhow::Result<PathBuf> {
    use std::io::Write as _;
    let mut file = tempfile::Builder::new()
        .prefix("convert_pdf_to_md-")
        .suffix(".md")
        .tempfile()
        .context("failed to create temp file")?;
    file.write_all(content.as_bytes())
        .context("failed to write markdown to temp file")?;
    let (_, path) = file.keep().context("failed to persist temp file")?;
    Ok(path)
}

fn convert_pdf_to_md_tool_desc() -> ToolDesc {
    let mut desc = ToolDescBuilder::new(TOOL_NAME)
        .parameters(crate::to_value!({
            "type": "object",
            "properties": {
                "pdf_path": {
                    "type": "string",
                    "description": "Path to the PDF file to convert"
                }
            },
            "required": ["pdf_path"]
        }))
        .build();
    desc.description = Some("Convert a local PDF file to Markdown using Docling.".to_string());
    desc.returns = Some(crate::to_value!({
        "oneOf": [
            {
                "type": "object",
                "properties": {
                    "md_path": { "type": "string" },
                    "size_chars": { "type": "integer", "minimum": 0 }
                },
                "required": ["md_path", "size_chars"]
            },
            {
                "type": "object",
                "properties": {
                    "pdf_path": { "type": "string" },
                    "error": { "type": "string" },
                    "phase": { "type": "string" }
                },
                "required": ["error", "phase"]
            }
        ]
    }));
    desc
}

fn validate_pdf_path(args: &Value) -> Result<PathBuf, Value> {
    let raw_path = args
        .pointer("/pdf_path")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .to_string();

    if raw_path.is_empty() {
        return Err(error_value(
            "",
            "validation",
            "missing required parameter: pdf_path",
        ));
    }

    let resolved = match resolve_input_path(&raw_path) {
        Ok(path) => path,
        Err(err) => {
            return Err(error_value(&raw_path, "validation", &err.to_string()));
        }
    };
    let resolved_string = resolved.to_string_lossy().into_owned();

    if !resolved.exists() {
        return Err(error_value(
            &resolved_string,
            "validation",
            "input path does not exist",
        ));
    }
    if !resolved.is_file() {
        return Err(error_value(
            &resolved_string,
            "validation",
            "input path must be a file",
        ));
    }

    let canonical = match resolved.canonicalize() {
        Ok(path) => path,
        Err(err) => {
            return Err(error_value(
                &resolved_string,
                "validation",
                &format!("failed to canonicalize input path: {err}"),
            ));
        }
    };
    let canonical_string = canonical.to_string_lossy().into_owned();

    match is_pdf_file(&canonical) {
        Ok(true) => Ok(canonical),
        Ok(false) => Err(error_value(
            &canonical_string,
            "validation",
            "input path must be a PDF file",
        )),
        Err(err) => Err(error_value(
            &canonical_string,
            "validation",
            &format!("failed to read input file: {err}"),
        )),
    }
}

fn resolve_input_path(raw_path: &str) -> anyhow::Result<PathBuf> {
    let expanded = shellexpand::tilde(raw_path).into_owned();
    let path = PathBuf::from(expanded);
    if path.is_absolute() {
        Ok(path)
    } else {
        Ok(std::env::current_dir()
            .context("failed to get current working directory")?
            .join(path))
    }
}

fn is_pdf_file(path: &Path) -> std::io::Result<bool> {
    let mut file = std::fs::File::open(path)?;
    let mut header = [0_u8; 5];
    match file.read_exact(&mut header) {
        Ok(()) => Ok(&header == b"%PDF-"),
        Err(err) if err.kind() == std::io::ErrorKind::UnexpectedEof => Ok(false),
        Err(err) => Err(err),
    }
}

fn error_value(pdf_path: &str, phase: &str, error: &str) -> Value {
    crate::to_value!({
        "pdf_path": pdf_path,
        "error": error,
        "phase": phase
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_minimal_pdf(path: &Path, text: &str) {
        let text = escape_pdf_string(text);
        let stream = format!("BT\n/F1 24 Tf\n72 100 Td\n({text}) Tj\nET");
        let objects = vec![
            "<< /Type /Catalog /Pages 2 0 R >>".to_string(),
            "<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_string(),
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 200] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>".to_string(),
            format!("<< /Length {} >>\nstream\n{stream}\nendstream", stream.len()),
            "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>".to_string(),
        ];

        let mut pdf = Vec::from("%PDF-1.4\n".as_bytes());
        let mut offsets = Vec::with_capacity(objects.len());

        for (index, object) in objects.iter().enumerate() {
            offsets.push(pdf.len());
            pdf.extend_from_slice(format!("{} 0 obj\n{}\nendobj\n", index + 1, object).as_bytes());
        }

        let xref_offset = pdf.len();
        pdf.extend_from_slice(
            format!("xref\n0 {}\n0000000000 65535 f \n", objects.len() + 1).as_bytes(),
        );
        for offset in offsets {
            pdf.extend_from_slice(format!("{offset:010} 00000 n \n").as_bytes());
        }
        pdf.extend_from_slice(
            format!(
                "trailer\n<< /Root 1 0 R /Size {} >>\nstartxref\n{}\n%%EOF\n",
                objects.len() + 1,
                xref_offset
            )
            .as_bytes(),
        );

        std::fs::write(path, pdf).expect("failed to write test pdf");
    }

    fn escape_pdf_string(text: &str) -> String {
        text.replace('\\', "\\\\")
            .replace('(', "\\(")
            .replace(')', "\\)")
    }

    #[test]
    fn test_convert_pdf_to_md_tool_desc_sets_name_schema_and_returns() {
        let desc = convert_pdf_to_md_tool_desc();

        assert_eq!(desc.name, TOOL_NAME);
        assert_eq!(
            desc.parameters
                .pointer("/required/0")
                .and_then(|v| v.as_str()),
            Some("pdf_path")
        );
        assert_eq!(
            desc.description.as_deref(),
            Some("Convert a local PDF file to Markdown using Docling.")
        );
        assert_eq!(
            desc.returns
                .as_ref()
                .and_then(|v| v.pointer("/oneOf/0/required/0"))
                .and_then(|v| v.as_str()),
            Some("md_path")
        );
    }

    #[test]
    fn test_missing_pdf_path_returns_validation_error() {
        let err = validate_pdf_path(&crate::to_value!({})).unwrap_err();
        assert_eq!(
            err.pointer("/phase").and_then(|v| v.as_str()),
            Some("validation")
        );
        assert_eq!(
            err.pointer("/error").and_then(|v| v.as_str()),
            Some("missing required parameter: pdf_path")
        );
    }

    #[test]
    fn test_nonexistent_pdf_path_returns_validation_error() {
        let err = validate_pdf_path(&crate::to_value!({
            "pdf_path": "definitely-missing-file.pdf"
        }))
        .unwrap_err();
        assert_eq!(
            err.pointer("/phase").and_then(|v| v.as_str()),
            Some("validation")
        );
        assert_eq!(
            err.pointer("/error").and_then(|v| v.as_str()),
            Some("input path does not exist")
        );
    }

    #[test]
    fn test_directory_path_returns_validation_error() {
        let dir = tempfile::tempdir().expect("failed to create temp dir");
        let err = validate_pdf_path(&crate::to_value!({
            "pdf_path": dir.path().to_string_lossy().to_string()
        }))
        .unwrap_err();
        assert_eq!(
            err.pointer("/phase").and_then(|v| v.as_str()),
            Some("validation")
        );
        assert_eq!(
            err.pointer("/error").and_then(|v| v.as_str()),
            Some("input path must be a file")
        );
    }

    #[test]
    fn test_non_pdf_file_returns_validation_error() {
        let dir = tempfile::tempdir().expect("failed to create temp dir");
        let path = dir.path().join("note.txt");
        std::fs::write(&path, "not a pdf").expect("failed to write test file");

        let err = validate_pdf_path(&crate::to_value!({
            "pdf_path": path.to_string_lossy().to_string()
        }))
        .unwrap_err();

        assert_eq!(
            err.pointer("/phase").and_then(|v| v.as_str()),
            Some("validation")
        );
        assert_eq!(
            err.pointer("/error").and_then(|v| v.as_str()),
            Some("input path must be a PDF file")
        );
    }

    #[test]
    fn test_docling_source_reads_from_env_vars() {
        assert!(DOCLING_SOURCE.contains("AILOY_PDF_PATH"));
        assert!(DOCLING_SOURCE.contains("AILOY_OUTPUT_PATH"));
        assert!(DOCLING_SOURCE.contains("DocumentConverter"));
        assert!(DOCLING_SOURCE.contains("export_to_markdown"));
    }

    #[cfg(feature = "sandbox")]
    #[tokio::test]
    #[ignore = "requires docling installation and model artifacts"]
    async fn test_convert_pdf_to_md_smoke() {
        use crate::sandbox::{Sandbox, SandboxConfig};

        // docling loads PyTorch models at runtime; 512 MiB (default) is not enough.
        let sandbox = Arc::new(
            Sandbox::new(SandboxConfig {
                memory_mib: 4096,
                ..SandboxConfig::default()
            })
            .await
            .expect("failed to create sandbox"),
        );
        let tool = build_convert_pdf_to_md_tool(sandbox)
            .await
            .expect("failed to build convert_pdf_to_md tool");

        let dir = tempfile::tempdir().expect("failed to create temp dir");
        let pdf_path = dir.path().join("hello.pdf");
        write_minimal_pdf(&pdf_path, "Hello Docling");

        let args = crate::message::Part::function(
            "call-1",
            TOOL_NAME,
            crate::to_value!({
                "pdf_path": pdf_path.to_string_lossy().to_string()
            }),
        );
        let msg = tool.call(&args).await.expect("tool call failed");
        let value = msg.contents[0].as_value().expect("expected value response");

        assert!(
            value.is_object(),
            "expected object response, got: {value:?}"
        );
        let md_path = value
            .pointer("/md_path")
            .and_then(|v| v.as_str())
            .unwrap_or_else(|| panic!("expected md_path in success result, got: {value:?}"));
        let markdown = std::fs::read_to_string(md_path).expect("failed to read markdown file");
        let size_chars = value
            .pointer("/size_chars")
            .and_then(|v| v.as_integer())
            .expect("expected size_chars in success result");
        assert_eq!(
            size_chars,
            markdown.chars().count() as i64,
            "size_chars should match markdown character count",
        );
        assert!(!markdown.trim().is_empty(), "markdown should not be empty");
    }
}
