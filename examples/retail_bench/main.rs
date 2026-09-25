//! Run [RetailBench](https://github.com/linghuazhang01/RetailBench) against an agent: one
//! supermarket, one day at a time, for as many days as the store survives.
//!
//! ```sh
//! cargo run --example retail_bench -- --smoke            # eight SKUs, two days: the machinery
//! cargo run --example retail_bench -- --days 3           # three days of the real store
//! cargo run --release --example retail_bench             # the benchmark: 96 SKUs, 180 days
//! ```
//!
//! The store is RetailBench's own simulator, run as a local REST server by
//! `simulator/sim` — see `simulator/README.md`. Its code is restored from the upstream
//! repository at a pinned commit and its data fetched from the same one, both on the first
//! run (about 700 MB, once; eight SKUs for `--smoke`).
//!
//! # A day is a turn
//!
//! Every morning the store writes what the agent may read into `runs/<slug>/context/`,
//! mounted read-only at `/context`. The agent reads it with its file tools and its shell,
//! calls an action when it wants to change something, and closes the day with `end_today`.
//! Then the store writes tomorrow's tree and the agent is asked again, **with an empty
//! history**.
//!
//! Fifteen of RetailBench's nineteen tools only look, and here looking is reading a file.
//! Three stayed tools because they can be *refused*, and a file cannot say no:
//!
//! * `place_order` — buy from one named supplier; refused for a supplier not quoting that
//!   SKU today, or for more than the till holds.
//! * `modify_sku_price` — set a shelf price.
//! * `end_today` — settle the day and move the clock.
//!
//! Their schemas are the environment's own, read from the store at startup rather than
//! written here. Upstream's `add_note` is a directory instead: the agent writes one file a
//! day into `/artifacts/notes/` and reads the earlier ones back, and that is the whole of
//! what survives the night.
//!
//! What the agent is told is in `system.md`, `user.md` (every morning) and `nudge.md` (when
//! a turn ends without closing the day).
//!
//! # What a run leaves behind
//!
//! ```text
//! runs/<slug>/
//!   metrics.json      run_days, final_networth, total_sales, and the ratios
//!   days.jsonl        one line per closed day: funds, net worth, sales
//!   tool_calls.jsonl  one line per action, in RetailBench's own field names
//!   days/NNN.json     the messages of one day's turn
//!   context/          the tree as it stood when the run ended
//!   artifacts/        what the agent wrote, notes/<date>.md among it
//! ```
//!
//! The store outlives nothing by default; `--keep-store` leaves it up to be asked
//! (`simulator/sim view view_inventory --run runs/<slug>`).
//!
//! Environment:
//!
//! * `AILOY_CORTEX_CONSOLE` — the console server binary, `cortex-krun` by default.
//! * `AILOY_MODEL` — the agent's model, `openai/gpt-6-astra` by default; its provider's API
//!   key has to be set (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, …).
//!
//! Read from `.env` as well.

use std::{
    io::Write as _,
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::{Instant, SystemTime, UNIX_EPOCH},
};

use ailoy::{
    agent::{AgentBuilder, AgentProvider, get_agent_providers_mut},
    console::Console,
    message::{Message, Part, Role},
    tool::{ToolDesc, ToolDescBuilder, get_tool_providers, get_tool_providers_mut},
    tool_func,
};
use anyhow::{Context as _, bail};
use cortex::{console::NetworkAccess, image::Image};
use futures::StreamExt as _;
use serde_json::{Value as Json, json};

const SYSTEM: &str = include_str!("system.md");
const USER: &str = include_str!("user.md");
const NUDGE: &str = include_str!("nudge.md");

/// The tools the agent may call, `end_today` last because that is where it belongs in a day.
const ACTIONS: [&str; 3] = ["place_order", "modify_sku_price", "end_today"];

/// The name the tool and agent providers carrying the three actions are registered under.
const PROVIDER: &str = "retail_bench";

/// How many times a day is asked before the harness closes it itself.
///
/// Upstream forces the day shut after twenty steps of its own loop; this is the same idea a
/// level up. A day left open would stop the clock, and the run would never end.
const NUDGES: usize = 3;

const HELP: &str = "\
cargo run --example retail_bench -- [flags]

  --days N         the horizon (default: 180)
  --config NAME    dynamic_hard | dynamic_middle | still_hard | still_middle
  --smoke          eight SKUs and two days: the machinery, not a result
  --keep-store     leave the store's server running when the run ends
";

struct Args {
    days: u32,
    config: String,
    smoke: bool,
    keep_store: bool,
}

fn parse_args() -> anyhow::Result<Args> {
    let mut args = Args {
        days: 180,
        config: "dynamic_hard".into(),
        smoke: false,
        keep_store: false,
    };
    let mut days = None;
    let mut rest = std::env::args().skip(1);
    while let Some(flag) = rest.next() {
        match flag.as_str() {
            "--days" => days = Some(rest.next().context("--days needs a number")?.parse()?),
            "--config" => args.config = rest.next().context("--config needs a name")?,
            "--smoke" => args.smoke = true,
            "--keep-store" => args.keep_store = true,
            "-h" | "--help" => {
                print!("{HELP}");
                std::process::exit(0);
            }
            other => bail!("unknown flag {other}\n\n{HELP}"),
        }
    }
    args.days = days.unwrap_or(if args.smoke { 2 } else { 180 });
    Ok(args)
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenvy::dotenv().ok();
    let args = parse_args()?;
    let model = std::env::var("AILOY_MODEL").unwrap_or_else(|_| "openai/gpt-6-astra".to_string());

    // Absolute, because a mount is named to the console server as a `file://` URL.
    let project_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/retail_bench");
    let sim = project_path.join("simulator/sim");
    prepare(&sim, args.smoke).await?;

    let run_dir = project_path.join("runs").join(slug(&model, &args.config));
    let context = run_dir.join("context");
    let artifacts = run_dir.join("artifacts");
    for dir in [&context, &artifacts.join("notes"), &run_dir.join("days")] {
        std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    }
    let record = Record(run_dir.clone());

    // ── the store
    let store = Store::start(&sim, &run_dir, &context, &args.config, args.days).await?;

    // ── the agent
    let day_over = Arc::new(AtomicBool::new(false));
    let tools = register_actions(&store, &record, &day_over).await?;
    let mut agent = AgentBuilder::new(&model)
        .agent_provider(PROVIDER)
        .instruction(SYSTEM)
        .system_tools()
        .tools(tools)
        .console(
            Console::builder()
                .stdio_client(&[&std::env::var("AILOY_CORTEX_CONSOLE")
                    .unwrap_or_else(|_| "cortex-krun".to_string())])
                .image(Image::new().base("python:3.12-slim-trixie"))
                .mount_readonly(context.clone(), "/context")
                .mount(artifacts.clone(), "/artifacts")
                .network(NetworkAccess::none())
                .vcpus(2)
                .memory_mib(2048)
                .build()
                .await
                .with_context(|| "starting the console")?,
        )
        .build()
        .await?;

    println!("  model  {model}");
    println!("  store  {}", store.base);
    println!("  run    {}\n", run_dir.display());

    // ── the days
    let mut failure: Option<String> = None;
    for day in 1..=args.days {
        match run_day(&mut agent, &store, &record, &day_over, day, args.days).await {
            Ok(None) => {}
            Ok(Some(reason)) => {
                println!("\nthe run ended on day {day}: {reason}");
                break;
            }
            // A failed day ends the run and not the process: the days before it are a
            // result, and on day 150 that is hours of work.
            Err(e) => {
                println!("\nday {day} failed: {e:#}");
                failure = Some(format!("day {day}: {e:#}"));
                break;
            }
        }
    }

    let metrics = store.get("/metrics").await.unwrap_or(Json::Null);
    record.write(
        "metrics.json",
        &json!({"model": model, "config": args.config, "horizon": args.days,
                "failed": failure, "metrics": metrics}),
    )?;
    println!("\n{}", serde_json::to_string_pretty(&metrics)?);
    println!("\nwritten to {}", run_dir.display());

    drop(agent);
    if args.keep_store {
        println!(
            "the store is still up at {} — `sim stop --run {}` ends it",
            store.base,
            run_dir.display()
        );
    } else {
        store.post("/stop", json!({})).await.ok();
    }
    Ok(())
}

/// One day: write its tree, ask the agent until it closes the day, and write down what the day
/// came to. Returns why the run ended, if this day ended it.
async fn run_day(
    agent: &mut ailoy::agent::Agent,
    store: &Store,
    record: &Record,
    day_over: &AtomicBool,
    day: u32,
    max_days: u32,
) -> anyhow::Result<Option<String>> {
    // Written before the turn, so the agent's first read is of a tree whose date matches the
    // prompt. Day one's was written when the store booted.
    if day > 1 {
        store.post("/context", json!({"market": true})).await?;
    }
    let state = store.get("/state").await?;
    let fill = |template: &str| {
        template
            .replace("{{day}}", &day.to_string())
            .replace("{{max_days}}", &max_days.to_string())
            .replace("{{date}}", state["date"].as_str().unwrap_or_default())
            .replace("{{funds}}", &number(&state["funds"]))
            .replace("{{net_worth}}", &number(&state["net_worth"]))
    };

    // An empty history every morning: the system message, and nothing the agent did yesterday
    // except what it wrote into its notes.
    agent.state.history.truncate(1);
    day_over.store(false, Ordering::SeqCst);

    // Whether the model answered at all today. A day where it never did is a model that could
    // not be reached, not one that decided to do nothing, and is not played as an empty day.
    let mut heard = false;
    let mut error = None;
    for nudge in 0..NUDGES {
        let text = fill(if nudge == 0 { USER } else { NUDGE });
        let mut stream = agent.run(Message::new(Role::User).with_contents([Part::text(text)]));
        while let Some(output) = stream.next().await {
            let message = match output {
                Ok(output) => output.message,
                Err(e) => {
                    error = Some(format!("{e:#}"));
                    break;
                }
            };
            heard = true;
            show(day, &message);
            // The day closes inside a tool call, so this is checked after every message: what
            // the model would say next belongs to a day that has not been written yet.
            if day_over.load(Ordering::SeqCst) {
                break;
            }
        }
        if day_over.load(Ordering::SeqCst) {
            break;
        }
    }
    record.write(
        &format!("days/{day:03}.json"),
        &json!({"day": day, "state": state, "error": error, "messages": agent.state.history}),
    )?;

    let mut ended = None;
    if !day_over.load(Ordering::SeqCst) {
        if !heard {
            bail!("the model never answered: {}", error.unwrap_or_default());
        }
        println!("{day:>3}   closed by the harness: the agent did not call end_today");
        let call = store.act("end_today", json!({})).await?;
        record
            .tool_call("end_today", &json!({"forced": true}), &call, store, 0.0)
            .await;
        ended = call.terminated;
    }

    let metrics = store.get("/metrics").await?;
    record.append(
        "days.jsonl",
        &json!({
            "day": day,
            "date": state["date"],
            "funds": metrics["final_funds"],
            "net_worth": metrics["final_networth"],
            "total_sales": metrics["total_sales"],
            "refusals": metrics["refusals"],
        }),
    );
    println!(
        "day {day:>3}  funds {:>12}  net worth {:>12}  sales {:>8}",
        number(&metrics["final_funds"]),
        number(&metrics["final_networth"]),
        number(&metrics["total_sales"]),
    );
    Ok(ended
        .filter(|reason| !reason.is_empty())
        .or_else(|| metrics["terminated"].as_str().map(str::to_string)))
}

/// Register the three actions under [`PROVIDER`] and return their descriptions.
///
/// A `ToolDesc` in a spec is resolved by name against a tool provider, so the default one is
/// cloned, keeping the built-in tools, and the three are inserted into the copy.
async fn register_actions(
    store: &Store,
    record: &Record,
    day_over: &Arc<AtomicBool>,
) -> anyhow::Result<Vec<ToolDesc>> {
    let declared = store.get("/actions").await?;
    let declared = declared["actions"].as_array().cloned().unwrap_or_default();

    let mut provider = get_tool_providers()
        .get("default")
        .cloned()
        .context("ailoy's default tool provider is not registered")?;
    let mut descs = Vec::new();
    for name in ACTIONS {
        // A name the store does not declare means this example and the simulator have drifted
        // apart, which is worth stopping for rather than running without the tool.
        let Some(spec) = declared.iter().find(|spec| spec["name"] == name) else {
            bail!("the store does not declare '{name}'");
        };
        let mut description = spec["description"].as_str().unwrap_or_default().to_string();
        if name == "end_today" {
            // True here and not in the runners upstream wrote the description for.
            description.push_str(
                " This ends your turn: call it once, when you are done for the day, and expect \
                 no further instructions afterwards.",
            );
        }
        descs.push(
            ToolDescBuilder::new(name)
                .description(description)
                .parameters(spec["input_schema"].clone())
                .build(),
        );

        let (store, record, day_over) = (store.clone(), record.clone(), day_over.clone());
        provider.insert_func(
            name,
            tool_func!(async |args: Value| -> Value
                with [store = store.clone(), record = record.clone(), day_over = day_over.clone()]
                {
                    let args: Json = args.into();
                    let started = Instant::now();
                    let text = match store.act(name, args.clone()).await {
                        Ok(call) => {
                            let elapsed = started.elapsed().as_secs_f64();
                            record.tool_call(name, &args, &call, &store, elapsed).await;
                            if call.day_over {
                                day_over.store(true, Ordering::SeqCst);
                            }
                            said(&call)
                        }
                        // Nothing the model can fix, but saying so beats a silent failure: the
                        // day never closes, and the harness closes it.
                        Err(e) => format!("The store could not be reached: {e:#}"),
                    };
                    text.into()
                }
            ),
        );
    }
    get_tool_providers_mut().insert(PROVIDER.to_string(), provider);
    get_agent_providers_mut().insert(
        PROVIDER.to_string(),
        AgentProvider::new("default", PROVIDER),
    );
    Ok(descs)
}

/// What the model is shown for one action: the store's own words, and a line saying so when the
/// day just closed. Without it, a model that was just told yesterday's sales tends to keep going.
fn said(call: &Call) -> String {
    if !call.ok {
        return format!("Refused: {}", call.said);
    }
    let mut text = call.said.clone();
    if call.day_over {
        text.push_str("\n\n---\nThe day is over. Stop here — do not call anything else.");
        match &call.terminated {
            Some(reason) if !reason.is_empty() => {
                text.push_str(&format!(" The run has ended: {reason}."))
            }
            _ => text.push_str(" Tomorrow's files will be waiting when you are asked again."),
        }
    }
    text
}

/// Set the store up if it is not: the upstream code, an interpreter for it, and the dataset.
async fn prepare(sim: &Path, smoke: bool) -> anyhow::Result<()> {
    let ready = !smoke
        && tokio::process::Command::new(sim)
            .arg("status")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .await
            .is_ok_and(|status| status.success());
    if ready {
        return Ok(());
    }
    let mut command = tokio::process::Command::new(sim);
    command.args(["setup", "--no-serve"]);
    if smoke {
        command.args(["--skus", "8"]);
    }
    let status = command
        .status()
        .await
        .with_context(|| format!("running {}. It needs `python3` on PATH.", sim.display()))?;
    anyhow::ensure!(status.success(), "setting the store up: {status}");
    Ok(())
}

/// The store, over HTTP. Cheap to clone: each action tool holds one.
#[derive(Clone)]
struct Store {
    http: reqwest::Client,
    base: String,
}

/// What one action came back as. A refusal is an answer, not an error.
struct Call {
    said: String,
    ok: bool,
    day_over: bool,
    terminated: Option<String>,
    body: Json,
}

impl Store {
    /// Start this run's server — booting reads about 600 MB, and `sim serve` waits for it.
    async fn start(
        sim: &Path,
        run: &Path,
        context: &Path,
        config: &str,
        days: u32,
    ) -> anyhow::Result<Self> {
        let status = tokio::process::Command::new(sim)
            .arg("serve")
            .arg("--run")
            .arg(run)
            .arg("--context")
            .arg(context)
            .args(["--config", config, "--days", &days.to_string()])
            .status()
            .await?;
        anyhow::ensure!(status.success(), "`sim serve` failed: {status}");
        let port = std::fs::read_to_string(run.join("sim.port")).context("reading sim.port")?;
        let store = Self {
            http: reqwest::Client::new(),
            base: format!("http://127.0.0.1:{}", port.trim()),
        };
        store
            .get("/state")
            .await
            .context("the store did not answer")?;
        Ok(store)
    }

    async fn get(&self, path: &str) -> anyhow::Result<Json> {
        let response = self.http.get(format!("{}{path}", self.base)).send().await?;
        let status = response.status();
        let body: Json = response.json().await?;
        anyhow::ensure!(status.is_success(), "GET {path} failed ({status}): {body}");
        Ok(body)
    }

    async fn post(&self, path: &str, payload: Json) -> anyhow::Result<Json> {
        let response = self
            .http
            .post(format!("{}{path}", self.base))
            .json(&payload)
            .send()
            .await?;
        let status = response.status();
        let body: Json = response.json().await?;
        anyhow::ensure!(status.is_success(), "POST {path} failed ({status}): {body}");
        Ok(body)
    }

    /// One action. `409` is the store refusing a well-formed request, which the agent is meant
    /// to read and try differently; anything else that is not `200` is this side being wrong.
    async fn act(&self, name: &str, arguments: Json) -> anyhow::Result<Call> {
        let response = self
            .http
            .post(format!("{}/actions/{name}", self.base))
            .json(&arguments)
            .send()
            .await?;
        let status = response.status().as_u16();
        let body: Json = response.json().await?;
        if status != 200 && status != 409 {
            bail!("{name} failed ({status}): {}", body["error"]);
        }
        let ok = status == 200;
        let said = body[if ok { "formatted" } else { "error" }]
            .as_str()
            .unwrap_or_default()
            .to_string();
        Ok(Call {
            said,
            ok,
            day_over: body["day_over"].as_bool().unwrap_or(false),
            terminated: body["terminated"].as_str().map(str::to_string),
            body,
        })
    }
}

/// The run's directory, written to as the run goes, so a run stopped on day 90 has ninety days
/// on disk.
#[derive(Clone)]
struct Record(PathBuf);

impl Record {
    fn write(&self, name: &str, value: &Json) -> anyhow::Result<()> {
        let path = self.0.join(name);
        std::fs::write(&path, serde_json::to_vec_pretty(value)?)
            .with_context(|| format!("writing {}", path.display()))
    }

    fn append(&self, name: &str, value: &Json) {
        let path = self.0.join(name);
        let appended = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .and_then(|mut file| writeln!(file, "{value}"));
        if let Err(e) = appended {
            eprintln!("could not append to {}: {e}", path.display());
        }
    }

    /// One action, in the field names RetailBench's own `analysis/` scripts read.
    async fn tool_call(&self, tool: &str, args: &Json, call: &Call, store: &Store, elapsed: f64) {
        let state = store.get("/state").await.unwrap_or_default();
        self.append(
            "tool_calls.jsonl",
            &json!({
                "ts": unix_now(),
                "tool": tool,
                "args": args,
                "ok": call.ok,
                "result": call.body.get("result"),
                "formatted": call.said,
                "funds": state["funds"],
                "net_worth": state["net_worth"],
                "current_date": state["current_date"],
                "day": state["day"],
                "elapsed_time": (elapsed * 10_000.0).round() / 10_000.0,
            }),
        );
    }
}

/// One line per thing the agent says or calls, so a long run can be watched.
fn show(day: u32, message: &Message) {
    for text in message.contents.iter().filter_map(Part::as_text) {
        if let Some(line) = text.trim().lines().next() {
            println!("{day:>3}   {}", truncate(line, 140));
        }
    }
    for call in message.tool_calls.iter().flatten() {
        if let Some((_, name, args)) = call.as_function() {
            let args = serde_json::to_string(args).unwrap_or_default();
            println!("{day:>3} → {name}({})", truncate(&args, 120));
        }
    }
    std::io::stdout().flush().ok();
}

fn truncate(text: &str, at: usize) -> String {
    if text.chars().count() <= at {
        return text.to_string();
    }
    text.chars().take(at).collect::<String>() + "…"
}

fn number(value: &Json) -> String {
    match value.as_f64() {
        Some(f) => format!("{f:.2}"),
        None => "-".to_string(),
    }
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or_default()
}

/// A sortable name for this run: when it started, which model, which configuration.
fn slug(model: &str, config: &str) -> String {
    format!(
        "{}_{}_{config}",
        unix_now(),
        model.replace(['/', ':', ' '], "-")
    )
}
