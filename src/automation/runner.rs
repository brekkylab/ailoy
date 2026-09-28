//! Turning one event into one run of the workflow, and collecting what the run left.

use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    sync::Arc,
    time::{Duration, Instant},
};

use anyhow::Context as _;
use futures::StreamExt as _;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use tokio::sync::Mutex;
use virtx::{console::ConsoleClient, protocol::Error as ConsoleError};

use super::{
    AutomationDef, Event, Outcome, RunRecord, TaskDetail, TaskRecord,
    console_spec::{GUEST_ARTIFACTS, GUEST_SCRATCH},
    event::unix_now,
    render::render,
    workflow::{EVENT_INPUT, Task, TaskKind, builtin_tool_desc},
};
use crate::{
    agent::{Agent, AgentState},
    lang_model::ResponseFormat,
    message::{Message, Part, Role},
};

/// Under the work directory: one directory per run.
pub const RUNS_DIR: &str = "runs";
/// In a run directory: the audit record.
pub const RECORD_FILE: &str = "record.json";
/// Under the scratch tree: one directory per python task with its code and I/O files.
pub const TASKS_SCRATCH_DIR: &str = ".ailoy/tasks";

/// Kept in a record; enough to read, not enough to bloat it.
const MAX_CAPTURED_CHARS: usize = 30_000;

/// System message for an agent task, ahead of the spec's own instruction. The trees'
/// paths are appended by the agent itself, so this speaks of them by role.
const INSTRUCTION: &str = "\
You are running one step of an automation. This turn began with an event, not with a \
person, and there is no one to ask: decide what the step needs and do it.

The user message is the whole task; everything it needs is already in it. Files a \
person should receive go in the artifacts directory. Your final message is this step's \
output, handed to the next step or kept as the run's result, so end by stating the \
result plainly rather than with a question.";

/// A run's console, shared by the runner's python tasks and every agent task's tools.
type ConsoleSlot = Arc<Mutex<Option<ConsoleClient>>>;

/// Keep the head and the tail of a long capture, with the middle elided.
fn middle_truncate(s: String, max_chars: usize) -> String {
    if s.chars().count() <= max_chars {
        return s;
    }
    let keep = max_chars / 2;
    let head: String = s.chars().take(keep).collect();
    let tail: String = s
        .chars()
        .rev()
        .take(max_chars - keep)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    format!("{head}\n… elided …\n{tail}")
}

/// Runs one automation: `(definition, event) → record`. Holds nothing between runs.
#[derive(Clone, Debug)]
pub struct Runner {
    def: AutomationDef,
    work_dir: PathBuf,
    /// The console server to start, when it is not the one virtx finds for itself.
    console_cmd: Option<Vec<String>>,
    provider: String,
}

/// A run laid out but not started: directories made, order settled. What a person
/// sees before approving one, and what a daemon keeps across a restart.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PreparedRun {
    pub run_id: String,
    pub dir: PathBuf,
    pub event: Event,
    /// Task names in execution order.
    pub order: Vec<String>,
}

impl Runner {
    /// Runs on the console server virtx finds for itself.
    pub fn new(def: AutomationDef, work_dir: impl Into<PathBuf>) -> Self {
        Self {
            def,
            work_dir: work_dir.into(),
            console_cmd: None,
            provider: "default".to_string(),
        }
    }

    /// Start `argv` as the console server instead. What it enforces of the console spec
    /// is its own business: a server that swaps no image and takes no network away
    /// leaves the spec unenforced.
    pub fn with_console_program(
        mut self,
        argv: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        self.console_cmd = Some(argv.into_iter().map(Into::into).collect());
        self
    }

    /// Name in [`get_agent_providers`](crate::agent::get_agent_providers).
    pub fn with_provider(mut self, provider: impl Into<String>) -> Self {
        self.provider = provider.into();
        self
    }

    pub fn def(&self) -> &AutomationDef {
        &self.def
    }

    /// [`prepare`](Self::prepare) then [`execute`](Self::execute).
    pub async fn run(&self, event: Event) -> anyhow::Result<RunRecord> {
        let prepared = self.prepare(event)?;
        self.execute(prepared).await
    }

    /// Lay out `runs/<id>/{scratch,artifacts}` and settle the order. Nothing talks to
    /// a model or a console yet.
    pub fn prepare(&self, event: Event) -> anyhow::Result<PreparedRun> {
        self.prepare_with_id(uuid::Uuid::new_v4().simple().to_string(), event)
    }

    /// [`prepare`](Self::prepare) under an id the caller chose, so a daemon's run and
    /// the runner's directory share one name. The id becomes a directory name.
    pub fn prepare_with_id(
        &self,
        run_id: impl Into<String>,
        event: Event,
    ) -> anyhow::Result<PreparedRun> {
        let run_id = run_id.into();
        anyhow::ensure!(
            !run_id.is_empty()
                && run_id
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_')),
            "run id `{run_id}` is not a plain identifier"
        );
        let dir = self.work_dir.join(RUNS_DIR).join(&run_id);
        let (scratch, artifacts) = trees(&dir);
        fs::create_dir_all(&scratch)?;
        fs::create_dir_all(&artifacts)?;
        Ok(PreparedRun {
            order: self
                .def
                .workflow
                .order()?
                .into_iter()
                .map(|t| t.name.clone())
                .collect(),
            run_id,
            dir,
            event,
        })
    }

    /// Open the console, run the tasks in order, record the run.
    pub async fn execute(&self, prepared: PreparedRun) -> anyhow::Result<RunRecord> {
        let (scratch, artifacts) = trees(&prepared.dir);
        fs::create_dir_all(&scratch)?;
        fs::create_dir_all(&artifacts)?;
        let started_at = unix_now();
        let mut completed: BTreeMap<String, Value> = BTreeMap::new();
        let mut tasks = Vec::new();

        let mut builder = self
            .def
            .console
            .console_builder(&self.def.workflow.packages())
            .mount(scratch.clone(), GUEST_SCRATCH)
            .mount(artifacts.clone(), GUEST_ARTIFACTS);
        if let Some(cmd) = &self.console_cmd {
            builder = builder.cmd(cmd);
        }
        let mut console = builder
            .build()
            .await
            .with_context(|| format!("console server {:?}", self.console_cmd))?;
        console
            .start()
            .await
            .map_err(|e| anyhow::anyhow!("starting the console: {e}"))?;
        let slot: ConsoleSlot = Arc::new(Mutex::new(Some(console)));

        let result = self
            .run_tasks(&prepared, &mut completed, &slot, &scratch, &mut tasks)
            .await;

        // Whatever happened above, the guest comes down here: a turn cut short by a
        // timeout leaves it up, and an error left the slot untouched.
        if let Some(console) = slot.lock().await.as_mut() {
            let _ = console.stop().await;
        }
        drop(slot);

        let outcome = result?;
        let output = if outcome.is_completed() {
            self.def
                .workflow
                .output_task()
                .and_then(|t| completed.get(&t.name).cloned())
        } else {
            None
        };
        let record = RunRecord {
            run_id: prepared.run_id,
            event: prepared.event,
            started_at,
            finished_at: unix_now(),
            outcome,
            output,
            tasks,
        };
        write_json(&prepared.dir.join(RECORD_FILE), &record)?;
        Ok(record)
    }

    /// The tasks in order, until one does not complete. Returns the run's outcome and
    /// leaves every completed task's output in `completed`.
    async fn run_tasks(
        &self,
        prepared: &PreparedRun,
        completed: &mut BTreeMap<String, Value>,
        slot: &ConsoleSlot,
        scratch: &Path,
        tasks: &mut Vec<TaskRecord>,
    ) -> anyhow::Result<Outcome> {
        let wf = &self.def.workflow;
        let deadline = Instant::now() + Duration::from_secs(wf.timeout_secs);
        let mut run_outcome = Outcome::Completed;

        for name in &prepared.order {
            let task = wf
                .task(name)
                .with_context(|| format!("prepared run names unknown task `{name}`"))?;
            if !run_outcome.is_completed() {
                let now = unix_now();
                tasks.push(TaskRecord {
                    name: name.clone(),
                    started_at: now,
                    finished_at: now,
                    input: Value::Null,
                    output: None,
                    outcome: Outcome::Skipped,
                    detail: TaskDetail::None,
                });
                continue;
            }

            let input = assemble_input(task, completed, &prepared.event.payload);
            let started_at = unix_now();
            let remaining = deadline.saturating_duration_since(Instant::now());
            let (outcome, output, detail) = if remaining.is_zero() {
                (Outcome::TimedOut, None, TaskDetail::None)
            } else {
                let budget = match task.timeout_secs {
                    Some(s) => remaining.min(Duration::from_secs(s)),
                    None => remaining,
                };
                match &task.kind {
                    TaskKind::Python { .. } => {
                        self.run_python(task, &input, slot, scratch, budget).await?
                    }
                    TaskKind::Agent { .. } => self.run_agent(task, &input, slot, budget).await?,
                }
            };

            if outcome.is_completed() {
                completed.insert(name.clone(), output.clone().unwrap_or(Value::Null));
            } else {
                run_outcome = outcome.clone();
            }
            tasks.push(TaskRecord {
                name: name.clone(),
                started_at,
                finished_at: unix_now(),
                input,
                output,
                outcome,
                detail,
            });
        }
        Ok(run_outcome)
    }

    /// `python3 main.py input.json output.json` in the console, files written on the
    /// host side of the scratch tree and named to the guest by its own paths.
    async fn run_python(
        &self,
        task: &Task,
        input: &Value,
        slot: &ConsoleSlot,
        scratch: &Path,
        budget: Duration,
    ) -> anyhow::Result<(Outcome, Option<Value>, TaskDetail)> {
        let code = self.def.task_code(task)?;
        let host_dir = scratch.join(TASKS_SCRATCH_DIR).join(&task.name);
        fs::create_dir_all(&host_dir)?;
        fs::write(host_dir.join("main.py"), &code)?;
        write_json(&host_dir.join("input.json"), input)?;
        // Not removed if it somehow exists: the console shares this tree with a guest,
        // and a name whose file was unlinked and remade reads back there as the old one
        // or as nothing. A task runs once per run, so the directory is its own.
        let output_path = host_dir.join("output.json");

        let guest_dir = Path::new(GUEST_SCRATCH)
            .join(TASKS_SCRATCH_DIR)
            .join(&task.name);
        let guest_file = |name: &str| guest_dir.join(name).to_string_lossy().into_owned();
        let argv = [
            "python3".to_string(),
            guest_file("main.py"),
            guest_file("input.json"),
            guest_file("output.json"),
        ];

        let detail = |exit_code, stdout: String, stderr: String| TaskDetail::Python {
            code: code.clone(),
            exit_code,
            stdout: middle_truncate(stdout, MAX_CAPTURED_CHARS),
            stderr: middle_truncate(stderr, MAX_CAPTURED_CHARS),
        };

        let mut guard = slot.lock().await;
        let console = guard.as_mut().context("the run's console is gone")?;
        let resp = match console
            .exec(argv, Some(budget.as_millis().min(u64::MAX as u128) as u64))
            .await
        {
            Ok(resp) => resp,
            Err(e) if e.code() == Some(ConsoleError::TIMED_OUT) => {
                return Ok((
                    Outcome::TimedOut,
                    None,
                    detail(None, String::new(), String::new()),
                ));
            }
            Err(e) => {
                return Ok((
                    Outcome::Failed {
                        error: e.to_string(),
                    },
                    None,
                    detail(None, String::new(), String::new()),
                ));
            }
        };
        drop(guard);

        let stdout = String::from_utf8_lossy(&resp.stdout).into_owned();
        let stderr = String::from_utf8_lossy(&resp.stderr).into_owned();
        if resp.code != 0 {
            let tail: String = stderr
                .chars()
                .rev()
                .take(2000)
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
                .collect();
            return Ok((
                Outcome::Failed {
                    error: format!("python exited {}: {}", resp.code, tail.trim()),
                },
                None,
                detail(Some(resp.code), stdout, stderr),
            ));
        }

        let output = match fs::read(&output_path) {
            Ok(bytes) => match serde_json::from_slice::<Value>(&bytes) {
                Ok(v) => v,
                Err(e) => {
                    return Ok((
                        Outcome::Failed {
                            error: format!("output.json is not JSON: {e}"),
                        },
                        None,
                        detail(Some(resp.code), stdout, stderr),
                    ));
                }
            },
            Err(_) => Value::Null,
        };
        Ok((
            Outcome::Completed,
            Some(output),
            detail(Some(resp.code), stdout, stderr),
        ))
    }

    /// One agent turn on the shared console. The user message is the rendered prompt
    /// and nothing else.
    async fn run_agent(
        &self,
        task: &Task,
        input: &Value,
        slot: &ConsoleSlot,
        budget: Duration,
    ) -> anyhow::Result<(Outcome, Option<Value>, TaskDetail)> {
        let TaskKind::Agent {
            agent,
            tools,
            output_schema,
            ..
        } = &task.kind
        else {
            anyhow::bail!("{}: not an agent task", task.name);
        };
        let prompt = render(&self.def.task_prompt(task)?, input);

        let mut spec = (**agent).clone();
        spec.instruction = Some(match &spec.instruction {
            Some(own) => format!("{INSTRUCTION}\n\n{own}"),
            None => INSTRUCTION.to_string(),
        });
        spec.tools = tools
            .iter()
            .map(|n| builtin_tool_desc(n))
            .collect::<anyhow::Result<Vec<_>>>()?;
        if let Some(schema) = output_schema {
            spec = spec.response_format(ResponseFormat::json_schema(schema.clone().into())?);
        }
        let state = AgentState::new().with_console_slot(slot.clone());
        let mut agent = Agent::try_with_provider_and_state(spec, &self.provider, state).await?;

        let query = Message::new(Role::User).with_contents([Part::text(prompt.clone())]);
        let mut outcome = {
            let mut stream = agent.run(query);
            let drive = async {
                while let Some(item) = stream.next().await {
                    item?;
                }
                anyhow::Ok(())
            };
            match tokio::time::timeout(budget, drive).await {
                Ok(Ok(())) => Outcome::Completed,
                Ok(Err(e)) => Outcome::Failed {
                    error: format!("{e:#}"),
                },
                Err(_) => Outcome::TimedOut,
            }
        };

        let transcript = agent.get_history().to_vec();
        let text = final_text(&transcript);
        let output = if !outcome.is_completed() {
            None
        } else if output_schema.is_some() {
            match serde_json::from_str::<Value>(text.trim()) {
                Ok(v) => Some(v),
                Err(e) => {
                    outcome = Outcome::Failed {
                        error: format!(
                            "the final answer is not the JSON output_schema asks for: {e}"
                        ),
                    };
                    None
                }
            }
        } else {
            Some(Value::String(text))
        };
        Ok((outcome, output, TaskDetail::Agent { prompt, transcript }))
    }
}

/// The task's `inputs` as one JSON object: each name → the named task's output, or
/// the event payload for `event`.
fn assemble_input(task: &Task, completed: &BTreeMap<String, Value>, event: &Value) -> Value {
    let mut map = Map::new();
    for (name, source) in &task.inputs {
        let value = if source == EVENT_INPUT {
            event.clone()
        } else {
            completed.get(source).cloned().unwrap_or(Value::Null)
        };
        map.insert(name.clone(), value);
    }
    Value::Object(map)
}

/// The text of the last assistant message: what the step answered.
fn final_text(history: &[Message]) -> String {
    history
        .iter()
        .rev()
        .find(|m| m.role == Role::Assistant)
        .map(|m| {
            m.contents
                .iter()
                .filter_map(Part::as_text)
                .collect::<Vec<_>>()
                .join("")
        })
        .unwrap_or_default()
}

fn trees(run_dir: &Path) -> (PathBuf, PathBuf) {
    (run_dir.join("scratch"), run_dir.join("artifacts"))
}

#[cfg(test)]
fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> anyhow::Result<Option<T>> {
    match fs::read(path) {
        Ok(bytes) => Ok(Some(
            serde_json::from_slice(&bytes)
                .with_context(|| format!("parsing {}", path.display()))?,
        )),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(anyhow::anyhow!("reading {}: {e}", path.display())),
    }
}

fn write_json<T: Serialize>(path: &Path, value: &T) -> anyhow::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, serde_json::to_vec_pretty(value)?)
        .with_context(|| format!("writing {}", path.display()))
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::automation::definition::write_def;

    /// `a` doubles the event's `n`; `b` adds one to `a`'s answer.
    fn two_python_tasks(root: &Path, b_code: &str) -> AutomationDef {
        write_def(
            root,
            &json!({"tasks": [
                {"name": "a", "kind": "python", "inputs": {"event": "event"},
                 "code": "import json,sys\nev=json.load(open(sys.argv[1]))['event']\n\
                          json.dump({'n': ev['n']*2}, open(sys.argv[2],'w'))\n"},
                {"name": "b", "kind": "python", "inputs": {"a": "a"}, "code": b_code},
            ]}),
        );
        AutomationDef::load(root).unwrap()
    }

    const B_OK: &str = "import json,sys\na=json.load(open(sys.argv[1]))['a']\n\
                        json.dump({'n': a['n']+1}, open(sys.argv[2],'w'))\n";

    #[test]
    fn prepare_lays_out_the_run_and_roundtrips() {
        let root = tempfile::tempdir().unwrap();
        let work = tempfile::tempdir().unwrap();
        let def = two_python_tasks(root.path(), B_OK);
        let runner = Runner::new(def, work.path());

        let p = runner.prepare(Event::new(json!({"n": 1}))).unwrap();
        assert_eq!(p.order, ["a", "b"]);
        assert!(p.dir.join("scratch").is_dir() && p.dir.join("artifacts").is_dir());
        assert!(p.dir.starts_with(work.path().join(RUNS_DIR)));

        let back: PreparedRun = serde_json::from_str(&serde_json::to_string(&p).unwrap()).unwrap();
        assert_eq!(back, p);
    }

    #[test]
    fn inputs_are_assembled_by_name() {
        let task: Task = serde_json::from_value(
            json!({"name": "c", "kind": "python", "inputs": {"x": "a", "ev": "event", "gone": "z"}}),
        )
        .unwrap();
        let mut completed = BTreeMap::new();
        completed.insert("a".to_string(), json!(1));
        let v = assemble_input(&task, &completed, &json!({"id": 9}));
        assert_eq!(v, json!({"x": 1, "ev": {"id": 9}, "gone": null}));
    }

    #[test]
    fn final_text_is_the_last_assistant_message() {
        let history = vec![
            Message::new(Role::System).with_contents([Part::text("s")]),
            Message::new(Role::User).with_contents([Part::text("u")]),
            Message::new(Role::Assistant).with_contents([Part::text("first")]),
            Message::new(Role::Tool).with_contents([Part::text("t")]),
            Message::new(Role::Assistant).with_contents([Part::text("a"), Part::text("b")]),
        ];
        assert_eq!(final_text(&history), "ab");
        assert_eq!(final_text(&[]), "");
    }

    /// A console server to run these against, when it is not the one virtx finds for
    /// itself: `$AILOY_CONSOLE_SERVER`, as argv split on spaces.
    fn console_program() -> Option<Vec<String>> {
        dotenvy::dotenv().ok();
        std::env::var("AILOY_CONSOLE_SERVER")
            .ok()
            .map(|s| s.split_whitespace().map(str::to_string).collect())
    }

    /// `Runner::new`, pointed at `$AILOY_CONSOLE_SERVER` when it names one.
    fn runner_for(def: AutomationDef, work: &Path) -> Runner {
        let runner = Runner::new(def, work);
        match console_program() {
            Some(argv) => runner.with_console_program(argv),
            None => runner,
        }
    }

    #[tokio::test]
    #[ignore = "needs a console server"]
    async fn python_tasks_chain_through_json() {
        let root = tempfile::tempdir().unwrap();
        let work = tempfile::tempdir().unwrap();
        let def = two_python_tasks(root.path(), B_OK);
        let runner = runner_for(def, work.path());

        let record = runner.run(Event::new(json!({"n": 5}))).await.unwrap();
        assert_eq!(record.outcome, Outcome::Completed, "{record:#?}");
        assert_eq!(record.output, Some(json!({"n": 11})));
        assert_eq!(record.tasks.len(), 2);
        assert_eq!(record.tasks[0].output, Some(json!({"n": 10})));

        let dir = work.path().join(RUNS_DIR).join(&record.run_id);
        let on_disk: RunRecord = read_json(&dir.join(RECORD_FILE)).unwrap().unwrap();
        assert_eq!(on_disk.run_id, record.run_id);
        assert!(
            dir.join("scratch")
                .join(TASKS_SCRATCH_DIR)
                .join("a/output.json")
                .exists()
        );
    }

    #[tokio::test]
    #[ignore = "needs a console server"]
    async fn a_failed_task_ends_the_run_and_skips_the_rest() {
        let root = tempfile::tempdir().unwrap();
        let work = tempfile::tempdir().unwrap();
        write_def(
            root.path(),
            &json!({"tasks": [
                {"name": "a", "kind": "python", "inputs": {"event": "event"},
                 "code": "import json,sys\nev=json.load(open(sys.argv[1]))['event']\n\
                          json.dump({'n': ev['n']*2}, open(sys.argv[2],'w'))\n"},
                {"name": "b", "kind": "python", "inputs": {"a": "a"}, "code": "import sys\nsys.exit('no')\n"},
                {"name": "c", "kind": "python", "inputs": {"b": "b"},
                 "code": "import json,sys\njson.dump('done', open(sys.argv[2],'w'))\n"},
            ]}),
        );
        let runner = runner_for(AutomationDef::load(root.path()).unwrap(), work.path());

        let record = runner.run(Event::new(json!({"n": 1}))).await.unwrap();
        assert!(
            matches!(record.outcome, Outcome::Failed { .. }),
            "{record:#?}"
        );
        let outcomes: Vec<_> = record
            .tasks
            .iter()
            .map(|t| (t.name.as_str(), t.outcome.clone()))
            .collect();
        assert_eq!(outcomes[0], ("a", Outcome::Completed));
        assert!(matches!(outcomes[1], ("b", Outcome::Failed { .. })));
        assert_eq!(outcomes[2], ("c", Outcome::Skipped));
        assert!(record.output.is_none());
    }

    /// Run a definition on disk by hand:
    ///
    /// ```text
    /// AILOY_AUTOMATION_DEF=<automation dir> AILOY_AUTOMATION_WORK=<work dir> \
    /// AILOY_AUTOMATION_EVENT=<event json file> \
    /// cargo test --lib run_definition_from_env -- --ignored --nocapture
    /// ```
    ///
    /// `$AILOY_CONSOLE_SERVER` names a console server to use instead of the one virtx
    /// finds for itself.
    #[tokio::test]
    #[ignore = "runs whatever definition the environment names"]
    async fn run_definition_from_env() {
        dotenvy::dotenv().ok();
        let var = |name: &str| std::env::var(name).unwrap_or_else(|_| panic!("set {name}"));
        let def = AutomationDef::load(var("AILOY_AUTOMATION_DEF")).unwrap();
        let mut runner = Runner::new(def, var("AILOY_AUTOMATION_WORK"));
        if let Some(argv) = console_program() {
            runner = runner.with_console_program(argv);
        }
        let payload: Value =
            serde_json::from_slice(&fs::read(var("AILOY_AUTOMATION_EVENT")).unwrap()).unwrap();

        let record = runner.run(Event::new(payload)).await.unwrap();
        println!("run {}: {:?}", record.run_id, record.outcome);
        println!(
            "output: {}",
            serde_json::to_string_pretty(&record.output).unwrap()
        );
    }
}
