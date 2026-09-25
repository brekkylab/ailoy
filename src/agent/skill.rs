//! Skills: directories in the console, each with a `SKILL.md` saying how to use it.

use anyhow::Context as _;
use cortex::console::Console;
use tokio::sync::Mutex;

/// The skills in `dirs`, as the section of the system message that lists them.
///
/// The catalog the Agent Skills integration guide describes: each skill's `name`,
/// `description` and `location` (its `SKILL.md`), after a few lines on how to load one.
/// Only the frontmatter goes in; the body is the agent's to read when a task calls for
/// the skill.
///
/// In the shape `model` reads best: an `<available_skills>` block for Claude, which is
/// trained on XML-tagged prompts, and a Markdown list for every other model.
pub(super) async fn render_skills(
    dirs: &[String],
    console: &Mutex<Option<Console>>,
    model: &str,
) -> anyhow::Result<String> {
    let mut guard = console.lock().await;
    let console = guard
        .as_mut()
        .context("an agent with skills needs a console to read them from")?;

    let mut skills = Vec::with_capacity(dirs.len());
    for dir in dirs {
        let location = format!("{}/SKILL.md", dir.trim_end_matches('/'));
        let text = read_to_string(console, &location)
            .await
            .with_context(|| format!("reading the skill {location}"))?;
        let (name, description) =
            frontmatter(&text).with_context(|| format!("reading the frontmatter of {location}"))?;
        skills.push(Skill {
            name,
            description,
            location,
        });
    }
    Ok(if is_claude(model) {
        render_xml(&skills)
    } else {
        render_markdown(&skills)
    })
}

/// What the catalog says about one skill.
struct Skill {
    name: String,
    description: String,
    /// The skill's `SKILL.md`, as the console spells it.
    location: String,
}

const PREAMBLE: &str = "# Skills\n\n\
    The following skills provide specialized instructions for specific tasks. When a task \
    matches a skill's description, read the SKILL.md at its location in full before \
    proceeding. When a skill refers to a relative path, resolve it against the skill's \
    directory, the one that holds its SKILL.md.\n\n";

/// Whether `model` is a Claude model, whichever provider serves it: `anthropic/*` by
/// name, and on Bedrock and the like by the model id, `bedrock/global.anthropic.claude-*`.
fn is_claude(model: &str) -> bool {
    crate::lang_model::model_family(model).starts_with("anthropic/") || model.to_ascii_lowercase().contains("claude")
}

fn render_xml(skills: &[Skill]) -> String {
    let mut out = format!("{PREAMBLE}<available_skills>\n");
    for skill in skills {
        out.push_str(&format!(
            "  <skill>\n    <name>{}</name>\n    <description>{}</description>\n    \
            <location>{}</location>\n  </skill>\n",
            escape(&skill.name),
            escape(&skill.description),
            escape(&skill.location),
        ));
    }
    out.push_str("</available_skills>");
    out
}

fn render_markdown(skills: &[Skill]) -> String {
    let mut out = PREAMBLE.to_string();
    for skill in skills {
        out.push_str(&format!(
            "- `{}` at {}. {}\n",
            skill.name, skill.location, skill.description
        ));
    }
    out.truncate(out.trim_end().len());
    out
}

/// `text` with what would end or open an XML element in it escaped.
fn escape(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

/// The whole of the file at `path` in the console, asked for again from where the last
/// piece ended for as long as the file is longer than what has arrived.
async fn read_to_string(console: &mut Console, path: &str) -> anyhow::Result<String> {
    let mut data = Vec::new();
    loop {
        let resp = console.read(path, Some(data.len() as u64), None).await?;
        let got = resp.data.len();
        data.extend(resp.data);
        if got == 0 || data.len() as u64 >= resp.size {
            break;
        }
    }
    String::from_utf8(data).context("the file is not UTF-8")
}

/// The `name` and `description` in the YAML frontmatter that opens a `SKILL.md`.
///
/// Only one line per field, which is all a skill's frontmatter holds, so no YAML parser:
/// a value is taken as it stands, with quotes around it dropped.
fn frontmatter(text: &str) -> anyhow::Result<(String, String)> {
    let mut lines = text.lines();
    anyhow::ensure!(
        lines.next().map(str::trim) == Some("---"),
        "a SKILL.md opens with a `---` line"
    );
    let (mut name, mut description) = (None, None);
    for line in lines.by_ref() {
        if line.trim() == "---" {
            break;
        }
        let Some((key, value)) = line.split_once(':') else {
            continue;
        };
        let value = value
            .trim()
            .trim_matches(|c| c == '"' || c == '\'')
            .to_string();
        match key.trim() {
            "name" => name = Some(value),
            "description" => description = Some(value),
            _ => {}
        }
    }
    Ok((
        name.context("the frontmatter has no `name`")?,
        description.context("the frontmatter has no `description`")?,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_skill_frontmatter() {
        let text = "---\nname: laya\ndescription: \"Decide: with Laya.\"\n---\n\n# Laya\n";
        let (name, description) = frontmatter(text).unwrap();
        assert_eq!(name, "laya");
        // Split at the first colon only, so one in the value stays.
        assert_eq!(description, "Decide: with Laya.");

        assert!(frontmatter("# Laya\n").is_err(), "no frontmatter");
        assert!(
            frontmatter("---\nname: laya\n---\n").is_err(),
            "no description"
        );
    }

    #[test]
    fn test_skill_escape() {
        assert_eq!(escape("a <b> & c"), "a &lt;b&gt; &amp; c");
    }

    #[test]
    fn test_skill_format_follows_the_model() {
        assert!(is_claude("anthropic/claude-sonnet-5"));
        assert!(is_claude("bedrock/global.anthropic.claude-sonnet-5"));
        assert!(!is_claude("openai/gpt-5"));
        assert!(!is_claude("google/gemini-3-pro"));

        let skills = [Skill {
            name: "laya".into(),
            description: "Decide with Laya.".into(),
            location: "/skills/laya/SKILL.md".into(),
        }];
        let xml = render_xml(&skills);
        assert!(xml.contains("<name>laya</name>"), "{xml}");
        assert!(
            xml.contains("<location>/skills/laya/SKILL.md</location>"),
            "{xml}"
        );
        assert!(xml.ends_with("</available_skills>"), "{xml}");

        let md = render_markdown(&skills);
        assert!(
            md.ends_with("- `laya` at /skills/laya/SKILL.md. Decide with Laya."),
            "{md}"
        );
        assert!(!md.contains('<'), "{md}");
    }
}
