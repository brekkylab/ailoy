use anyhow::Context;
use dedent::dedent;
use minijinja::{Environment, context};
use minijinja_contrib::{add_to_environment, pycompat::unknown_method_callback};

use crate::value::Document;

pub struct SystemMessageRenderer {
    template: String,
    mj_env: Environment<'static>,
}

impl SystemMessageRenderer {
    pub fn new() -> Self {
        let default_template = dedent!(r#"
        {%- if content %}
            {{- content }}
        {%- endif %}
        {%- if knowledge_results %}
            {{- '\n\n# Knowledges\n\nBelow is a list of documents retrieved from knowledge bases. Try to answer user\'s question based on the provided knowledges.\n' }}
            {{- "<documents>\n" }}
            {%- for item in knowledge_results %}
            {{- "<document>\n" }}
                {{- item.document + '\n' }}
            {{- "</document>\n" }}
            {%- endfor %}
            {{- "</documents>\n" }}
        {%- endif %}
        "#).to_string();
        let default_mj_env = Self::_create_mj_env(default_template.clone());

        Self {
            template: default_template,
            mj_env: default_mj_env,
        }
    }

    pub fn with_template(self, template: String) -> Self {
        let mj_env = Self::_create_mj_env(template.clone());
        Self {
            template,
            mj_env,
            ..self
        }
    }

    pub fn template(&self) -> &String {
        &self.template
    }

    pub fn _create_mj_env(template: String) -> Environment<'static> {
        let mut e = Environment::new();
        add_to_environment(&mut e);
        e.set_unknown_method_callback(unknown_method_callback);
        e.add_template_owned("template", template).unwrap();
        e
    }

    pub fn render(
        &self,
        content: String,
        knowledge_results: Option<Vec<Document>>,
    ) -> anyhow::Result<String> {
        let ctx = context!(content => content, knowledge_results => knowledge_results);
        let rendered = self
            .mj_env
            .get_template("template")
            .unwrap()
            .render(ctx)
            .context("minijinja::render failed")?;

        Ok(rendered)
    }
}
