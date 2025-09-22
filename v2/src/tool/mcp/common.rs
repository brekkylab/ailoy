use base64::Engine;

use crate::value::{Part, PartImageColorspace, ToolDesc, Value};

/// Convert MCP tool description to ToolDesc
pub(super) fn map_mcp_tool_to_tool_description(value: rmcp::model::Tool) -> ToolDesc {
    ToolDesc {
        name: value.name.into(),
        description: value.description.map(|v| v.into()),
        parameters: value
            .input_schema
            .iter()
            .map(|(k, v)| {
                (
                    k.clone(),
                    <serde_json::Value as Into<Value>>::into(v.clone()),
                )
            })
            .collect(),
        returns: value.output_schema.map(|map| {
            map.iter()
                .map(|(k, v)| {
                    (
                        k.clone(),
                        <serde_json::Value as Into<Value>>::into(v.clone()),
                    )
                })
                .collect()
        }),
    }
}

/// Convert MCP result to parts
pub(super) fn call_tool_result_to_parts(
    value: rmcp::model::CallToolResult,
) -> Result<Vec<Part>, String> {
    use image::{ColorType, DynamicImage};

    fn detect_colorspace(img: &DynamicImage) -> Result<PartImageColorspace, String> {
        match img.color() {
            ColorType::L8 | ColorType::L16 => Ok(PartImageColorspace::Grayscale),
            ColorType::Rgb8 | ColorType::Rgb16 => Ok(PartImageColorspace::RGB),
            ColorType::Rgba8 | ColorType::Rgba16 => Ok(PartImageColorspace::RGBA),
            other => Err(format!("Unsupported color type: {:?}", other)),
        }
    }

    if let Some(result) = value.structured_content {
        Ok(vec![Part::TextContent(
            serde_json::to_string(&result).unwrap(),
        )])
    } else if let Some(content) = value.content {
        let mut rv = Vec::with_capacity(content.len());
        for raw_content in content {
            let v = match raw_content.raw {
                rmcp::model::RawContent::Text(raw_text_content) => {
                    Part::TextContent(raw_text_content.text)
                }
                rmcp::model::RawContent::Image(raw_image_content) => {
                    let buf = base64::engine::general_purpose::STANDARD
                        .decode(raw_image_content.data)
                        .map_err(|e| format!("Invalid base64: {}", e.to_string()))?;
                    let img = match raw_image_content.mime_type.as_str() {
                        "image/png" => {
                            image::load_from_memory_with_format(&buf, image::ImageFormat::Png)
                        }
                        "image/jpeg" => {
                            image::load_from_memory_with_format(&buf, image::ImageFormat::Jpeg)
                        }
                        "image/webp" => {
                            image::load_from_memory_with_format(&buf, image::ImageFormat::WebP)
                        }
                        media_type => {
                            panic!("Unsupported media type: {}", media_type)
                        }
                    }
                    .map_err(|e| format!("Invalid image: {}", e.to_string()))?;
                    let h = img.height() as usize;
                    let w = img.width() as usize;
                    let color_space = detect_colorspace(&img)?;
                    let nbytes = buf.len() / h / w / color_space.channel();
                    Part::ImageContent {
                        h,
                        w,
                        c: color_space,
                        nbytes,
                        buf,
                    }
                }
                rmcp::model::RawContent::Resource(_) => todo!(),
                rmcp::model::RawContent::Audio(_) => todo!(),
            };
            rv.push(v);
        }
        Ok(rv)
    } else {
        Ok(vec![Part::text_content("null")])
    }
}
