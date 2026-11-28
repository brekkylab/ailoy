//! Skills: directories in the sandbox, each with a `SKILL.md` saying how to use it.

use anyhow::Context as _;

use crate::console::Sandbox;

/// The skills in `dirs` and those of the sandbox's abins, as the section of the system message
/// that lists them; `None` when there are none.
///
/// Per the Agent Skills integration guide: a short preamble, then each skill's `name`,
/// `description` and `location` (its `SKILL.md`). Only frontmatter is included; the agent
/// reads the body when it uses the skill.
///
/// An abin's `SKILL.md` is at `/abin/skills/{name}/`; an abin without one is left out.
///
/// An `<available_skills>` XML block for Claude (trained on XML-tagged prompts), a
/// Markdown list otherwise; `org` is the spec's model org, e.g. `"anthropic"`.
pub(super) async fn render_skills(
    dirs: &[String],
    sandbox: Option<&Sandbox>,
    org: &str,
) -> anyhow::Result<Option<String>> {
    let abins = sandbox.map_or(&[][..], |sandbox| sandbox.info().abins.as_slice());
    if dirs.is_empty() && abins.is_empty() {
        return Ok(None);
    }
    let sandbox = sandbox.context("an agent with skills needs a sandbox to read them from")?;

    let mut locations: Vec<(String, bool)> = dirs
        .iter()
        .map(|dir| (format!("{}/SKILL.md", dir.trim_end_matches('/')), true))
        .collect();
    for abin in abins {
        let location = abin_skill(abin);
        if !locations.iter().any(|(l, _)| *l == location) {
            locations.push((location, false));
        }
    }

    let mut skills = Vec::with_capacity(locations.len());
    for (location, required) in locations {
        let text = match crate::console::read(sandbox, &location).await {
            Err(e) if !required && e.status() == Some(404) => continue,
            read => read
                .map_err(anyhow::Error::from)
                .and_then(|data| String::from_utf8(data).context("the file is not UTF-8"))
                .with_context(|| format!("reading the skill {location}"))?,
        };
        let (name, description) =
            frontmatter(&text).with_context(|| format!("reading the frontmatter of {location}"))?;
        skills.push(Skill {
            name,
            description,
            location,
        });
    }
    if skills.is_empty() {
        return Ok(None);
    }
    Ok(Some(if org == "anthropic" {
        render_xml(&skills)
    } else {
        render_markdown(&skills)
    }))
}

/// Where virtx puts the `SKILL.md` of the abin `name`.
fn abin_skill(name: &str) -> String {
    format!("/abin/skills/{name}/SKILL.md")
}

/// What the catalog says about one skill.
struct Skill {
    name: String,
    description: String,
    /// The skill's `SKILL.md`, as the sandbox spells it.
    location: String,
}

const PREAMBLE: &str = "# Skills\n\n\
    The following skills provide specialized instructions for specific tasks. When a task \
    matches a skill's description, read the SKILL.md at its location in full before \
    proceeding. When a skill refers to a relative path, resolve it against the skill's \
    directory, the one that holds its SKILL.md.\n\n";

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

/// The `name` and `description` in the YAML frontmatter that opens a `SKILL.md`.
///
/// No YAML parser: skill frontmatter fields are single-line, so a value is taken verbatim
/// minus surrounding quotes.
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
