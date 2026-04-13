use std::{
    io::Read as _,
    path::{Path, PathBuf},
    sync::Arc,
};

use anyhow::Context as _;

use crate::{
    agent::rt::tool::{
        ToolFunc, ToolKind, ToolRuntime,
        python_repl::{PythonReplConfig, PythonSourceToolConfig, build_python_source_tool},
    },
    datatype::Value,
    message::{Part, ToolDesc, ToolDescBuilder},
};

const TOOL_NAME: &str = "convert_pdf_to_md";
const DOCLING_PACKAGE: &str = "docling>=2,<3";
const PYTHON_VERSION: &str = "3.12";
const CONVERSION_TIMEOUT_SECS: u64 = 600;
const DOCLING_SOURCE: &str = r#"
import json
import logging
import os
import tempfile
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


def load_args():
    args_path = os.environ.get("AILOY_ARGS_JSON_PATH", "")
    if not args_path:
        raise ValueError("missing AILOY_ARGS_JSON_PATH")
    return json.loads(Path(args_path).read_text(encoding="utf-8"))


def main():
    args = load_args()
    raw_pdf_path = args.get("pdf_path")
    if not isinstance(raw_pdf_path, str) or not raw_pdf_path.strip():
        raise ValueError("`pdf_path` must be a non-empty string")

    pdf_path = Path(raw_pdf_path)
    markdown = build_converter().convert(pdf_path).document.export_to_markdown()
    size_chars = len(markdown)
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".md",
        prefix="convert_pdf_to_md-",
        delete=False,
        encoding="utf-8",
    ) as fp:
        fp.write(markdown)
        md_path = str(Path(fp.name).resolve())
    print(json.dumps({"md_path": md_path, "size_chars": size_chars}, ensure_ascii=False), end="")


if __name__ == "__main__":
    main()
"#;

pub async fn build_convert_pdf_to_md_tool() -> anyhow::Result<ToolRuntime> {
    let python_tool = Arc::new(build_convert_pdf_to_md_python_tool().await?);
    let desc = convert_pdf_to_md_tool_desc();

    let f: Arc<ToolFunc> = Arc::new(move |args: Value| {
        let python_tool = python_tool.clone();
        Box::pin(async move {
            let pdf_path = match validate_pdf_path(&args) {
                Ok(path) => path,
                Err(err) => return err,
            };

            let tool_call = Part::function_with_id(
                "convert-pdf-to-md-internal",
                TOOL_NAME,
                crate::to_value!({
                    "pdf_path": pdf_path.to_string_lossy().to_string()
                }),
            );

            match python_tool.run(tool_call).await {
                Ok(msg) => {
                    let value = match msg.contents.first().and_then(|part| part.as_value()) {
                        Some(value) => value,
                        None => {
                            return error_value(
                                &pdf_path.to_string_lossy(),
                                "execution",
                                "python source tool returned no value payload",
                            );
                        }
                    };

                    if value.is_object() {
                        return value.clone();
                    }

                    error_value(
                        &pdf_path.to_string_lossy(),
                        "execution",
                        "python source tool returned unexpected payload type",
                    )
                }
                Err(err) => error_value(
                    &pdf_path.to_string_lossy(),
                    "execution",
                    &format!("failed to run python source tool: {err}"),
                ),
            }
        })
    });

    Ok(ToolRuntime::new_with_kind(desc, f, ToolKind::Builtin))
}

async fn build_convert_pdf_to_md_python_tool() -> anyhow::Result<ToolRuntime> {
    build_python_source_tool(
        convert_pdf_to_md_tool_desc(),
        convert_pdf_to_md_tool_config()?,
    )
    .await
}

fn convert_pdf_to_md_tool_config() -> anyhow::Result<PythonSourceToolConfig> {
    Ok(PythonSourceToolConfig::new(DOCLING_SOURCE)
        .python(PythonReplConfig {
            python_version: Some(PYTHON_VERSION.to_string()),
            venv_path: Some(docling_venv_path()?.to_string_lossy().into_owned()),
            packages: vec![DOCLING_PACKAGE.to_string()],
        })
        .timeout_secs(CONVERSION_TIMEOUT_SECS))
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

fn docling_venv_path() -> anyhow::Result<PathBuf> {
    Ok(dirs::cache_dir()
        .context("cannot determine cache directory")?
        .join("ailoy")
        .join("venvs")
        .join("docling"))
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
    use crate::{
        agent::{BuiltinToolProvider, ToolSet},
        message::Part,
    };

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
    fn test_convert_pdf_to_md_tool_config_sets_runtime_and_docling() {
        let config = convert_pdf_to_md_tool_config().expect("failed to build config");

        assert_eq!(config.timeout_secs, CONVERSION_TIMEOUT_SECS);
        assert_eq!(
            config.python.python_version.as_deref(),
            Some(PYTHON_VERSION)
        );
        assert_eq!(
            config.python.packages.as_slice(),
            &[DOCLING_PACKAGE.to_string()]
        );
        assert_eq!(
            config.python.venv_path.as_deref(),
            Some(docling_venv_path().unwrap().to_string_lossy().as_ref())
        );
        assert!(config.source.contains("DocumentConverter"));
        assert!(config.source.contains("export_to_markdown"));
        assert!(config.source.contains("NamedTemporaryFile"));
        assert!(config.source.contains("\"md_path\""));
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

    #[test_with::executable(uv)]
    #[tokio::test]
    #[ignore = "requires real uv + docling installation"]
    async fn test_builtin_provider_registers_convert_pdf_to_md() {
        let tool_set = ToolSet::new()
            .with_builtin(&BuiltinToolProvider::ConvertPdfToMd {})
            .await
            .expect("failed to build builtin tool");

        assert!(tool_set.get(TOOL_NAME).is_some());
    }

    #[test_with::executable(uv)]
    #[tokio::test]
    #[ignore = "requires docling installation and model artifacts"]
    async fn test_convert_pdf_to_md_smoke() {
        let tool = build_convert_pdf_to_md_tool()
            .await
            .expect("failed to build convert_pdf_to_md tool");

        let dir = tempfile::tempdir().expect("failed to create temp dir");
        let pdf_path = dir.path().join("hello.pdf");
        write_minimal_pdf(&pdf_path, "Hello Docling");

        let call = Part::function_with_id(
            "call-1",
            TOOL_NAME,
            crate::to_value!({
                "pdf_path": pdf_path.to_string_lossy().to_string()
            }),
        );
        let msg = tool.run(call).await.expect("tool call failed");
        let value = msg.contents[0].as_value().expect("expected value response");

        assert!(
            value.is_object(),
            "expected object response, got: {value:?}"
        );
        let md_path = value
            .pointer("/md_path")
            .and_then(|v| v.as_str())
            .expect("expected md_path in success result");
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
        assert!(
            markdown.to_lowercase().contains("hello")
                || markdown.to_lowercase().contains("docling"),
            "markdown should contain extracted text, got: {markdown:?}"
        );
    }
}
