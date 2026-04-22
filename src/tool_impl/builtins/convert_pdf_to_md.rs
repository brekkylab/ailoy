use std::{path::Path, sync::Arc};

use super::python_repl::PythonScriptRunner;
use crate::{
    datatype::Value,
    message::{ToolDesc, ToolDescBuilder},
    sandbox::Sandbox,
    tool::{Tool, ToolFunc},
};

const TOOL_NAME: &str = "convert_pdf_to_md";

const SETUP_SCRIPT: &str = "pip install 'docling>=2,<3' \
     && pip install --force-reinstall --no-deps opencv-python-headless";
const SETUP_TIMEOUT_SECS: u64 = 600;
const CONVERSION_TIMEOUT_SECS: u64 = 600;

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

// ---------------------------------------------------------------------------
// Tool factory
// ---------------------------------------------------------------------------

pub async fn build_convert_pdf_to_md_tool() -> anyhow::Result<Tool> {
    let runner = Arc::new(PythonScriptRunner::new(
        Some(SETUP_SCRIPT.to_string()),
        SETUP_TIMEOUT_SECS,
    ));
    let desc = convert_pdf_to_md_tool_desc();

    let f = ToolFunc::new(move |args: Value, sandbox: Option<Arc<Sandbox>>| {
        let runner = runner.clone();
        async move {
            let pdf_path = match validate_pdf_path(&args) {
                Ok(p) => p,
                Err(e) => return e,
            };
            let output_path = derive_output_path(
                &pdf_path,
                args.pointer("/output_path").and_then(|v| v.as_str()),
            );

            let sandbox = sandbox.as_ref();

            if let Err(e) = runner.ensure_setup(sandbox).await {
                return error_value(
                    "",
                    "initialization",
                    &format!("failed to set up docling: {e}"),
                );
            }

            match runner
                .run_with_timeout(
                    sandbox,
                    DOCLING_SOURCE,
                    &[
                        ("AILOY_PDF_PATH", pdf_path.as_str()),
                        ("AILOY_OUTPUT_PATH", output_path.as_str()),
                    ],
                    CONVERSION_TIMEOUT_SECS,
                )
                .await
            {
                Ok(r) if r.timed_out => error_value(
                    &pdf_path,
                    "execution",
                    &format!("timed out after {CONVERSION_TIMEOUT_SECS}s"),
                ),
                Ok(r) if r.exit_code != 0 => error_value(&pdf_path, "execution", r.stderr.trim()),
                Ok(_) => crate::to_value!({ "md_path": output_path }),
                Err(e) => error_value(&pdf_path, "execution", &e.to_string()),
            }
        }
    });

    Ok(Tool::new(desc, Arc::new(f)))
}

// ---------------------------------------------------------------------------
// Utility helpers
// ---------------------------------------------------------------------------

fn validate_pdf_path(args: &Value) -> Result<String, Value> {
    let raw = args
        .pointer("/pdf_path")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .to_string();
    if raw.is_empty() {
        return Err(error_value(
            "",
            "validation",
            "missing required parameter: pdf_path",
        ));
    }
    Ok(raw)
}

fn derive_output_path(pdf_path: &str, override_path: Option<&str>) -> String {
    if let Some(p) = override_path.filter(|s| !s.trim().is_empty()) {
        return p.to_string();
    }
    Path::new(pdf_path)
        .with_extension("md")
        .to_string_lossy()
        .into_owned()
}

fn convert_pdf_to_md_tool_desc() -> ToolDesc {
    let mut desc = ToolDescBuilder::new(TOOL_NAME)
        .parameters(crate::to_value!({
            "type": "object",
            "properties": {
                "pdf_path": {
                    "type": "string",
                    "description": "Path to the input PDF file. When a sandbox is active this must be a path inside the VM (e.g. `/workspace/doc.pdf`); otherwise it is a host filesystem path."
                },
                "output_path": {
                    "type": "string",
                    "description": "Path to write the output Markdown file. Uses the same path context as `pdf_path`. Defaults to the input path with the `.pdf` extension replaced by `.md`."
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
                "properties": { "md_path": { "type": "string" } },
                "required": ["md_path"]
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

fn error_value(pdf_path: &str, phase: &str, error: &str) -> Value {
    crate::to_value!({ "pdf_path": pdf_path, "error": error, "phase": phase })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

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
    fn test_derive_output_path_replaces_extension() {
        assert_eq!(
            derive_output_path("/workspace/doc.pdf", None),
            "/workspace/doc.md"
        );
    }

    #[test]
    fn test_derive_output_path_uses_override() {
        assert_eq!(
            derive_output_path("/workspace/doc.pdf", Some("/workspace/out.md")),
            "/workspace/out.md"
        );
    }

    #[test]
    fn test_docling_source_reads_from_env_vars() {
        assert!(DOCLING_SOURCE.contains("AILOY_PDF_PATH"));
        assert!(DOCLING_SOURCE.contains("AILOY_OUTPUT_PATH"));
        assert!(DOCLING_SOURCE.contains("DocumentConverter"));
        assert!(DOCLING_SOURCE.contains("export_to_markdown"));
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
    fn test_nonempty_pdf_path_passes_validation() {
        assert!(validate_pdf_path(&crate::to_value!({ "pdf_path": "/workspace/doc.pdf" })).is_ok());
    }

    #[cfg(feature = "sandbox")]
    #[tokio::test]
    #[ignore = "requires docling installation and model artifacts"]
    async fn test_convert_pdf_to_md_smoke() {
        use crate::{
            message::Part,
            sandbox::{Sandbox, SandboxConfig},
            tool::test_helpers::call_with_sandbox,
        };

        fn minimal_pdf_bytes() -> Vec<u8> {
            let stream = "BT\n/F1 24 Tf\n72 100 Td\n(Hello Docling) Tj\nET";
            let objects = vec![
                "<< /Type /Catalog /Pages 2 0 R >>".to_string(),
                "<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_string(),
                "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 200] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>".to_string(),
                format!("<< /Length {} >>\nstream\n{stream}\nendstream", stream.len()),
                "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>".to_string(),
            ];
            let mut pdf = Vec::from("%PDF-1.4\n".as_bytes());
            let mut offsets = Vec::with_capacity(objects.len());
            for (i, obj) in objects.iter().enumerate() {
                offsets.push(pdf.len());
                pdf.extend_from_slice(format!("{} 0 obj\n{}\nendobj\n", i + 1, obj).as_bytes());
            }
            let xref = pdf.len();
            pdf.extend_from_slice(
                format!("xref\n0 {}\n0000000000 65535 f \n", objects.len() + 1).as_bytes(),
            );
            for o in offsets {
                pdf.extend_from_slice(format!("{o:010} 00000 n \n").as_bytes());
            }
            pdf.extend_from_slice(
                format!(
                    "trailer\n<< /Root 1 0 R /Size {} >>\nstartxref\n{}\n%%EOF\n",
                    objects.len() + 1,
                    xref
                )
                .as_bytes(),
            );
            pdf
        }

        let sandbox = Arc::new(
            Sandbox::new(SandboxConfig {
                memory_mib: 4096,
                ..SandboxConfig::default()
            })
            .await
            .expect("failed to create sandbox"),
        );
        let tool = build_convert_pdf_to_md_tool()
            .await
            .expect("failed to build convert_pdf_to_md tool");

        sandbox
            .write_file("/workspace/hello.pdf", &minimal_pdf_bytes())
            .await
            .expect("failed to write PDF into sandbox");

        let args = Part::function(
            "call-1",
            TOOL_NAME,
            crate::to_value!({ "pdf_path": "/workspace/hello.pdf" }),
        );
        let msg = call_with_sandbox(&tool, args, sandbox.clone()).await;
        let value = msg.contents[0].as_value().expect("expected value response");

        assert!(
            value.is_object(),
            "expected object response, got: {value:?}"
        );
        let md_path = value
            .pointer("/md_path")
            .and_then(|v| v.as_str())
            .expect("expected md_path in success result");
        assert_eq!(md_path, "/workspace/hello.md");

        let markdown = sandbox
            .read_file(md_path)
            .await
            .expect("failed to read markdown from sandbox");
        assert!(!markdown.trim().is_empty(), "markdown should not be empty");
    }
}
