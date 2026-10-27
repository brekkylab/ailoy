# Memory

An agent with a memory remembers a user from one session to the next. It saves every turn it finishes to a folder in its console, word for word, and searches that folder with the `search_history` tool when an answer depends on the user. With the profile on, it also keeps short notes about the user, which go into its system message when a conversation starts. This page is about that memory; the VM's RAM is a console setting, in [Building VM](./building-vm).

## Giving the agent a memory

`memory(dir)` takes a folder's path inside the VM, where the memory commands run, so the agent needs a console. The folder must already exist there: the agent never makes it, so a mistyped path does not become an empty memory. Mount a folder of your machine at that path, since what a session writes anywhere else in the VM is gone when it ends. The folder on your machine has to exist too, or the VM does not boot.

::: code-group

```python [Python]
console = await (
    ConsoleClient.builder()
    .image(Recipe("python:3.12-slim-trixie"))
    .mount("./memory", "/memory")
    .build()
)

agent = await (
    AgentBuilder("anthropic/claude-haiku-5-5")
    .console(console)
    .memory("/memory")
    .build()
)
```

```js [Node.js]
const console_ = await ConsoleClient.builder()
  .image(new Recipe('python:3.12-slim-trixie'))
  .mount('./memory', '/memory')
  .build()

const agent = await new AgentBuilder('anthropic/claude-haiku-5-5')
  .console(console_)
  .memory('/memory')
  .build()
```

```rust [Rust]
let console = ConsoleClient::builder()
    .image(Recipe::new("python:3.12-slim-trixie"))
    .mount(std::path::absolute("./memory")?, "/memory")
    .build()
    .await?;

let agent = AgentBuilder::new("anthropic/claude-haiku-5-5")
    .console(console)
    .memory("/memory")
    .build()
    .await?;
```

:::

The agent gets the `search_history` tool, and `profile` too when the [profile](#the-profile) is on. Without the profile, a memory adds nothing to the system message.

## What is saved

At the end of each turn, the agent adds two messages to the history, the file `mem.sqlite` in the folder, which the first save makes:

- what the user asked, the text of the query;
- what the model answered: the text of every message it gave in the turn, including text it gave beside a tool call, joined by a blank line.

Each is saved as a line that records who said it (`user` or `assistant`), the [conversation](#conversations) it was said in, and when, in UTC. Thinking, tool calls and tool results are not saved, nor anything in a message that is not text. A message longer than 1,500 characters is saved as several lines, whole paragraphs where they fit, since a search answers with whole lines.

The turn is saved before its last message (or, with `run_stream`, its last delta) reaches you, so a program that stops reading there has it saved already. A save that fails, as it does when the folder is missing, does not fail the turn: the Rust crate logs a warning through the `log` crate, and the Python and Node bindings report nothing. After the save the agent stops the console's VM, and the next command boots it again; files in the VM are kept.

## Searching the history

The model calls `search_history` with a few words. Its arguments:

| Argument | Default | What it is |
| --- | --- | --- |
| `query` | | The words to find. |
| `limit` | 10 | How many lines to answer with, at most 15. |
| `sort` | `relevance` | `relevance`, best match first, or `newest`, the same lines most recent first. |
| `role` | both | `user` for only the user's lines, `assistant` for only the agent's. |
| `around` | | Instead of searching, read the lines around this result id. |
| `context` | 2 | With `around`, how many lines to read on each side, at most 5. |

Matching is by words, not meaning: a line matches if it has any of the query's words, and lines with more of them, and rarer ones, come first. The lines of the conversation in progress are left out, since the model already has them. The best match comes with up to two lines said before and after it in its own conversation. After a user said "I switched to oat milk last week. Just answer OK." and the agent answered "OK", a later conversation's search for `milk coffee` answered:

```json
{
  "count": 1,
  "results": [
    {
      "id": "d470ddf9-1a55-4618-8729-ce1524eeb028",
      "date": "2026-10-27 15:25 UTC",
      "role": "user",
      "text": "I switched to oat milk last week. Just answer OK.",
      "before": [],
      "after": [
        {
          "id": "4e2bade1-4d7d-42ba-9937-c76d3022154a",
          "date": "2026-10-27 15:25 UTC",
          "role": "assistant",
          "text": "OK"
        }
      ]
    }
  ]
}
```

No match is `{"count": 0, "results": []}`. The tool's description tells the model when to search: before answering anything that depends on the user, and to check exact numbers, dates and facts that change.

## The profile

The profile is off unless you turn it on:

::: code-group

```python [Python]
agent = await (
    AgentBuilder("anthropic/claude-haiku-5-5")
    .console(console)
    .memory("/memory", profile=True, user_limit=1_000)
    .build()
)
```

```js [Node.js]
const agent = await new AgentBuilder('anthropic/claude-haiku-5-5')
  .console(console_)
  .memory('/memory', { profile: true, userLimit: 1000 })
  .build()
```

```rust [Rust]
use ailoy::memory::Memory;

let agent = AgentBuilder::new("anthropic/claude-haiku-5-5")
    .console(console)
    .memory(Memory::new("/memory").with_profile(true).with_limits(1_000, 3_000))
    .build()
    .await?;
```

:::

With it, the folder also holds two Markdown files, each a list of entries with a blank line between them. Each file is made when the model first writes to it.

| File | What the tool asks the model to keep there | Limit | Setting |
| --- | --- | --- | --- |
| `USER.md` | Who the user is: name, work, lasting preferences, how they like to be helped. | 2,000 characters | `user_limit` (`userLimit`) |
| `NOTES.md` | Its own notes: agreements, what to avoid, lessons learned. | 3,000 characters | `notes_limit` (`notesLimit`) |

In Rust, `with_limits(user, notes)` sets both limits. The `profile` tool adds an entry, replaces or removes the entry that contains a piece of text, or makes several such changes at once, all of them or none. A change that would take a file past its limit is refused and changes nothing. A file already past its limit, after you edited it by hand, can still be shortened.

When the agent writes its system message, `build()` reads both files through the console and adds a `# Memory` section after the instruction and the skills:

```text
# Memory

Your profile of this user, kept with the `profile` tool across conversations. It is a lossy summary loaded when this conversation started; your changes show up next time. Check exact numbers, dates and changed facts with `search_history` even if the profile mentions them.

## About the user [36/1000 chars]
Name is Nuri. Prefers short answers.

## Your notes [0/3000 chars]
(empty)
```

The section shows the files as `build()` read them. The tool writes the files at once, but what the model changes shows up in the system message of the next agent you build. A system message that a `history(...)` brings is kept as it is, without the section, and a profile that cannot be read is shown empty.

## Conversations

Each agent has a conversation id: a new UUID, unless you give it one. The lines it saves carry that id, `search_history` leaves out the lines that have it, and `around` reads only within one conversation. Read it as `agent.conversation_id` (`agent.conversationId`, `agent.conversation_id()`).

Give the id when a new agent carries a conversation on, as a server does that builds an agent for each request from the history it stored. Without it, the new agent's searches find the conversation's own earlier lines as if they came from a past one.

::: code-group

```python [Python]
agent = await (
    AgentBuilder("anthropic/claude-haiku-5-5")
    .console(console)
    .memory("/memory")
    .history(saved_history)
    .conversation_id(saved_id)
    .build()
)
# After the turn, store agent.history and agent.conversation_id for the next request.
```

```js [Node.js]
const agent = await new AgentBuilder('anthropic/claude-haiku-5-5')
  .console(console_)
  .memory('/memory')
  .history(savedHistory)
  .conversationId(savedId)
  .build()
// After the turn, store agent.history and agent.conversationId for the next request.
```

```rust [Rust]
let agent = AgentBuilder::new("anthropic/claude-haiku-5-5")
    .console(console)
    .memory("/memory")
    .history(saved_history)
    .conversation_id(saved_id)
    .build()
    .await?;
// After the turn, store agent.history() and agent.conversation_id() for the next request.
```

:::

`Agent.from_spec` (`Agent.fromSpec`) takes `memory` and `conversation_id` (`conversationId`) as well. A memory given there is a folder without a profile.

## The `mem` command

The agent reads and writes the history with `mem`, run in the console, and the profile files with `sh`, `cat`, `printf` and `mv`, so the folder can be anywhere a command in the VM reaches. virtx puts `mem` in the VM at `/abin/mem`, and `/abin` comes first on the `PATH`.

Saving needs a `mem` that records who said a line and in which conversation, one whose `mem insert --help` lists `--role`. With an older `mem`, a turn saves nothing, and `search_history` searches what the folder already holds, without leaving out the conversation in progress and with `role` ignored.
