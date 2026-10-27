# Skills

A skill is a folder of instructions for one kind of task, with the files those instructions use, in the [Agent Skills](https://agentskills.io/specification) format. The agent is told which skills it has and what each is for, and to read a skill's instructions when a task matches it.

## What a skill folder holds

A `SKILL.md`, and anything it refers to: scripts, references, data. `SKILL.md` opens with YAML frontmatter giving the skill's `name` and a `description` of what it does and when to use it, and the instructions follow. This is the start of the one in `examples/laya/shared`, whose instructions tell the agent to run `run_laya.py` from the same folder:

````markdown
---
name: laya
description: Answer typed questions about a text with Laya, a machine learning model for decisions. Use it to classify, score or answer yes or no about a message, an email or a ticket, with calibrated probabilities.
---

# Laya

Laya is a machine learning model for decisions. Given a text, such as a message, an email or a
ticket, it answers typed questions about it with calibrated probabilities. It picks one of
several options, rates on an ordered scale, or answers yes or no, and it generates no text.
You put what is to be decided to it as questions, and write the reply yourself from its
answers.

## Running it

Run `run_laya.py` from this directory with the mode and the request as its two arguments.

```sh
python3 run_laya.py bf16 '{"state": "...", "questions": {...}}'
```
````

Ailoy reads `name` and `description` and nothing else from the frontmatter, each as one line: keep each on a single line, without YAML's `>` or `|`.

## Giving the agent a skill

`skill(dir)` takes the folder's path in the agent's VM, not on your machine, so mount the folder into the VM, with whatever its instructions expect to find there: Laya's reads its model from `/models`, which `examples/laya` mounts too. Call `skill` once for each skill.

::: code-group

```python [Python]
console = await (
    ConsoleClient.builder()
    .image(Recipe("python:3.12-slim-trixie"))
    .mount_readonly("./skills/laya", "/skills/laya")
    .build()
)

agent = await (
    AgentBuilder("anthropic/claude-haiku-5-5")
    .system_tools()
    .console(console)
    .skill("/skills/laya")
    .build()
)
```

```js [Node.js]
const console_ = await ConsoleClient.builder()
  .image(new Recipe('python:3.12-slim-trixie'))
  .mountReadonly('./skills/laya', '/skills/laya')
  .build()

const agent = await new AgentBuilder('anthropic/claude-haiku-5-5')
  .systemTools()
  .console(console_)
  .skill('/skills/laya')
  .build()
```

```rust [Rust]
let console = ConsoleClient::builder()
    .image(Recipe::new("python:3.12-slim-trixie"))
    .mount_readonly(std::path::absolute("./skills/laya")?, "/skills/laya")
    .build()
    .await?;

let agent = AgentBuilder::new("anthropic/claude-haiku-5-5")
    .system_tools()
    .console(console)
    .skill("/skills/laya")
    .build()
    .await?;
```

:::

When the agent writes the system message itself, `build()` reads the frontmatter of each `SKILL.md` through the console, and fails if a `SKILL.md` is missing or has no `name` or `description`. The system message gets a list of the skills after the instruction: each one's name, description and location, and a line telling the model to read a skill's `SKILL.md` in full before using it. The rest of `SKILL.md` stays out of the system message, and the agent reads it with its own tools, so an agent with skills needs a console and a tool that reads files, such as those of `system_tools()`.

For `anthropic` models the list is an XML block:

```text
# Skills

The following skills provide specialized instructions for specific tasks. When a task matches a skill's description, read the SKILL.md at its location in full before proceeding. When a skill refers to a relative path, resolve it against the skill's directory, the one that holds its SKILL.md.

<available_skills>
  <skill>
    <name>laya</name>
    <description>Answer typed questions about a text with Laya, a machine learning model for decisions. Use it to classify, score or answer yes or no about a message, an email or a ticket, with calibrated probabilities.</description>
    <location>/skills/laya/SKILL.md</location>
  </skill>
</available_skills>
```

For any other org, the same preamble is followed by a Markdown list, one line per skill:

```text
- `laya` at /skills/laya/SKILL.md. Answer typed questions about a text with Laya, a machine learning model for decisions. Use it to classify, score or answer yes or no about a message, an email or a ticket, with calibrated probabilities.
```

Skills are added only to the system message the agent writes itself. A `history(...)` that brings a system message keeps it: the skills are not read or added, though the `# Mounts` section is, as [Message format](./message-format) describes.
