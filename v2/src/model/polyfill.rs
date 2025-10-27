use dedent::dedent;
use serde::{Deserialize, Serialize};

use crate::value::{Document, Message, Part, Role};

/// Provides a polyfill for LLMs that do not natively support the Document feature.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
#[cfg_attr(feature = "python", pyo3_stub_gen::derive::gen_stub_pyclass)]
#[cfg_attr(feature = "python", pyo3::pyclass(get_all, set_all))]
#[cfg_attr(feature = "nodejs", napi_derive::napi(object))]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
#[cfg_attr(feature = "wasm", tsify(from_wasm_abi, into_wasm_abi))]
pub struct DocumentPolyfill {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_message_template: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub query_message_template: Option<String>,
}

impl DocumentPolyfill {
    pub fn polyfill(
        &self,
        mut msgs: Vec<Message>,
        documents: Vec<Document>,
    ) -> anyhow::Result<Vec<Message>> {
        // Find part indices
        // @jhlee: Currently, templating applies only to the first text `Part` within a message.
        // If the text `Part` is splitted, it might lead to undefined behavior.
        // We need to find a more robust solution for handling multiple Parts and non-text cases.
        fn get_part_idx(msgs: &[Message], msg_idx: Option<usize>) -> Option<usize> {
            if let Some(msg_idx) = msg_idx {
                msgs.get(msg_idx)
                    .unwrap()
                    .contents
                    .iter()
                    .position(|p| p.is_text())
            } else {
                None
            }
        }

        fn parse_original_text(
            msgs: &[Message],
            msg_idx: Option<usize>,
            part_idx: Option<usize>,
        ) -> String {
            if let Some(msg_idx) = msg_idx
                && let Some(part_idx) = part_idx
            {
                msgs.get(msg_idx)
                    .unwrap()
                    .contents
                    .get(part_idx)
                    .unwrap()
                    .as_text()
                    .unwrap()
                    .to_owned()
            } else {
                String::new()
            }
        }

        // Apply template
        fn apply_template(
            tmpl: &str,
            text: String,
            documents: &Vec<Document>,
        ) -> anyhow::Result<String> {
            let mut env = minijinja::Environment::new();
            env.add_template("polyfill", &tmpl)?;
            let res = env
                .get_template("polyfill")
                .unwrap()
                .render(minijinja::context! {
                    text => text,
                    documents => documents,
                })?;
            Ok(res)
        }

        // system message handling
        if let Some(tmpl) = &self.system_message_template {
            // Find first system message (index 0 in most cases, but we won't assume).
            let msg_idx = msgs.iter().position(|m| m.role == Role::System);
            let part_idx = get_part_idx(&msgs, msg_idx);

            // Get original message text part
            let orig_text = parse_original_text(&msgs, msg_idx, part_idx);

            // Apply template
            let rendered_text = apply_template(tmpl, orig_text, &documents)?;

            // Insert or replace rendered message
            if let Some(msg_idx) = msg_idx
                && let Some(part_idx) = part_idx
            {
                *msgs
                    .get_mut(msg_idx)
                    .unwrap()
                    .contents
                    .get_mut(part_idx)
                    .unwrap()
                    .as_text_mut()
                    .unwrap() = rendered_text;
            } else if let Some(msg_idx) = msg_idx {
                msgs.get_mut(msg_idx)
                    .unwrap()
                    .contents
                    .push(Part::text(rendered_text));
            } else {
                msgs.insert(
                    0,
                    Message::new(Role::System).with_contents([Part::text(rendered_text)]),
                );
            }
        }

        // query (last user) message handling
        if let Some(tmpl) = &self.query_message_template {
            // Find last user message (otherwise, it'll do nothing)
            if let Some(msg_idx) = msgs.iter().rposition(|m| m.role == Role::User) {
                let part_idx = get_part_idx(&msgs, Some(msg_idx));

                // Get original message text part
                let orig_text = parse_original_text(&msgs, Some(msg_idx), part_idx);

                // Apply template
                let rendered_text = apply_template(tmpl, orig_text, &documents)?;

                // Insert or replace rendered message
                if let Some(part_idx) = part_idx {
                    *msgs
                        .get_mut(msg_idx)
                        .unwrap()
                        .contents
                        .get_mut(part_idx)
                        .unwrap()
                        .as_text_mut()
                        .unwrap() = rendered_text;
                } else {
                    msgs.get_mut(msg_idx)
                        .unwrap()
                        .contents
                        .push(Part::text(rendered_text));
                }
            };
        }

        Ok(msgs)
    }
}

pub fn create_simple_document_polyfill() -> DocumentPolyfill {
    DocumentPolyfill {
        system_message_template: Some(dedent!(r#"
            {{- text }}
            # Knowledges
            After the user’s question, a list of documents retrieved from the knowledge base may appear. Try to answer the user’s question based on the provided knowledges.
            "#
        ).to_owned()),
        query_message_template: Some(dedent!(r#"
            {{- text }}
            {%- if documents %}
                {{- "<documents>\n" }}
                {%- for doc in documents %}
                {{- "<document>\n" }}
                    {{- doc.text + '\n' }}
                {{- "</document>\n" }}
                {%- endfor %}
                {{- "</documents>\n" }}
            {%- endif %}
            "#
        ).to_owned()),
    }
}

#[cfg(feature = "python")]
mod py {
    use pyo3::pymethods;
    use pyo3_stub_gen_derive::gen_stub_pymethods;

    use crate::model::DocumentPolyfill;

    #[gen_stub_pymethods]
    #[pymethods]
    impl DocumentPolyfill {
        #[new]
        fn __new__() -> Self {
            Self::default()
        }
    }
}
