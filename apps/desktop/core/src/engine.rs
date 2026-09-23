//! The engine: everything the window can ask for, behind one `Arc`.
//!
//! Every other module in this crate answers one question — where the sessions are, what the
//! workspace serves, which providers ailoy knows about. This is the object that holds them
//! together and is the only one the Tauri layer ever sees: one `Arc<Engine>` in managed
//! state, one method per command.

use std::{
    path::{Path, PathBuf},
    sync::Arc,
};

use ailoy::message::Part;

use cortex::fs::FileSystem;

use crate::{
    catalog::{self, Catalog, split_model_id},
    config::EngineConfig,
    error::{EngineError, Result},
    providers,
    run::{RunDeps, RunHandle, RunManager},
    store::{MountRow, Store},
    types::*,
    workspace::{WorkspaceManager, connectors, fsops},
};

pub struct Engine {
    cfg: EngineConfig,
    store: Arc<Store>,
    workspace: Arc<WorkspaceManager>,
    catalog: Arc<Catalog>,
    /// Where the fetched catalog is kept between starts.
    catalog_cache: PathBuf,
    /// Wakes the background refresh loop early: the setting was just turned on.
    catalog_wake: Arc<tokio::sync::Notify>,
    runs: RunManager,
    /// The exclusive lock on `<data_dir>/engine.lock`, held for the engine's whole life.
    /// Never read — the value *is* the lock, and the OS releases it when this file closes
    /// (on drop, or when the process dies however it dies).
    _instance_lock: std::fs::File,
    /// Flipped to `true` when the background connector restore has finished. A `watch`
    /// rather than a `Notify` because it latches: a waiter that arrives after the restore
    /// already ended must return immediately, not block for a notification it missed.
    restored: Arc<tokio::sync::watch::Sender<bool>>,
    /// Held across the whole of `settings_set`. The store is transactional per statement and
    /// `providers::apply` reads it before touching ailoy's registry, so two settings commands
    /// arriving together could otherwise interleave as write(A) → write(B) → apply(B) →
    /// apply(A) and leave the registry holding a key the store no longer has.
    settings_lock: tokio::sync::Mutex<()>,
}

impl Engine {
    pub async fn start(cfg: EngineConfig) -> Result<Arc<Engine>> {
        std::fs::create_dir_all(&cfg.data_dir)?;
        let instance_lock = lock_data_dir(&cfg.data_dir)?;
        let store = Arc::new(Store::open(&cfg.db_path())?);
        // The chosen root, or the default when nothing has been chosen yet. Read before the
        // workspace is built rather than applied after: starting on one directory and
        // swapping to another would mount the wrong tree for as long as that took, and a
        // console spawned in between would have been handed it.
        let files_root = store
            .setting_get(crate::config::ROOT_SETTING)?
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| cfg.default_files_root());
        let workspace = Arc::new(
            WorkspaceManager::start(
                files_root,
                cfg.artifacts_root(),
                cfg.mountpoint(),
                cfg.mount_workspace,
            )
            .await,
        );

        let catalog_cache = cfg.cache_dir().join("models.json");
        let catalog = Arc::new(Catalog::load(Some(&catalog_cache)));
        let catalog_wake = Arc::new(tokio::sync::Notify::new());
        if cfg.catalog_refresh {
            tokio::spawn(refresh_catalog(
                catalog.clone(),
                store.clone(),
                catalog_cache.clone(),
                catalog_wake.clone(),
            ));
        }

        // Not fatal: a legacy or hand-edited settings row that fails to parse would otherwise
        // block startup, and the settings pane that could fix it is behind a started engine.
        // `settings_set` still surfaces the same error when the user edits.
        if let Err(e) = providers::apply(&store) {
            tracing::warn!("applying provider settings at start: {e}");
        }

        // Nothing is started here: the server is written out and run by the first run that
        // needs a console, so a window with no run yet has spent nothing on one.
        let console = cfg
            .console
            .then(|| cortex::console::Backend::local().home(cfg.cortex_home()));

        let runs = RunManager::new(RunDeps {
            store: store.clone(),
            console,
            workspace: workspace.clone(),
            catalog: catalog.clone(),
            scratch_root: cfg.scratch_root(),
        });
        let engine = Arc::new(Engine {
            cfg,
            store,
            workspace,
            catalog,
            catalog_cache,
            catalog_wake,
            runs,
            _instance_lock: instance_lock,
            restored: Arc::new(tokio::sync::watch::channel(false).0),
            settings_lock: tokio::sync::Mutex::new(()),
        });

        // Restoring connectors is not on the path to a usable window. `build_and_probe`
        // gives a remote store 15 seconds to answer, and a laptop that woke up off-network
        // has every one of them time out — awaiting that here is the one startup wait
        // nothing the user did can bound, with the window still dark for all of it. So the
        // engine is handed back with the root mount alone and the connectors arrive in the
        // list as they come up; `mount_list()` says so, and `wait_restored` is for a test
        // (or a caller) that needs the settled answer.
        {
            let store = engine.store.clone();
            let workspace = engine.workspace.clone();
            let restored = engine.restored.clone();
            let cache_dir = engine.cfg.cache_dir();
            tokio::spawn(async move {
                restore_connectors(&store, &workspace, &cache_dir).await;
                let _ = restored.send(true);
            });
        }
        Ok(engine)
    }

    /// Ask every run to stop, wait for them to finish writing, then take the workspace down.
    ///
    /// `cancel_all` only signals: a run ends on its own terms — it still flushes the
    /// assembler's trailing message into SQLite, which is the partial answer the user was
    /// watching. Joining the run tasks is what makes that write land before the process
    /// exits. Three seconds is the cap: a run wedged in a tool call must not hold the
    /// window open, and the workspace comes down either way.
    pub async fn shutdown(&self) {
        self.runs.cancel_all().await;
        if !self.runs.wait_idle(std::time::Duration::from_secs(3)).await {
            tracing::warn!("a run did not finish within the shutdown grace period");
        }
        self.workspace.shutdown().await;
    }

    /// Wait until the background connector restore has finished — every stored connector
    /// either attached or listed with its error. Returns immediately once it has.
    pub async fn wait_restored(&self) {
        let mut rx = self.restored.subscribe();
        while !*rx.borrow_and_update() {
            // The sender lives in this `Engine`, so `changed()` can only fail if the engine
            // is being dropped — at which point there is nothing left to wait for.
            if rx.changed().await.is_err() {
                return;
            }
        }
    }

    // ── sessions ────────────────────────────────────────────────────────────

    pub async fn session_list(&self) -> Result<Vec<SessionSummary>> {
        let mut out = Vec::new();
        for r in self.store.session_list()? {
            let running = self.runs.is_running(&r.id).await;
            out.push(SessionSummary {
                id: r.id,
                title: r.title,
                model: r.model,
                created_at: r.created_at,
                updated_at: r.updated_at,
                running,
            });
        }
        Ok(out)
    }

    pub async fn session_create(&self, model: Option<String>) -> Result<SessionSummary> {
        let model = match model {
            // Checked the way `session_set_model` checks it: a session whose model is not a
            // `provider/model` id can be created but never run, and the refusal the user
            // would eventually see says nothing about the model picker they used.
            Some(m) if !m.trim().is_empty() => {
                if split_model_id(&m).is_none() {
                    return Err(EngineError::Invalid(format!(
                        "A model id looks like provider/model: {m}"
                    )));
                }
                m
            }
            _ => providers::read_settings(&self.store)?.default_model,
        };
        let id = uuid::Uuid::new_v4().to_string();
        let r = self.store.session_create(&id, "New chat", &model)?;
        Ok(SessionSummary {
            id: r.id,
            title: r.title,
            model: r.model,
            created_at: r.created_at,
            updated_at: r.updated_at,
            running: false,
        })
    }

    pub async fn session_rename(&self, id: &str, title: &str) -> Result<()> {
        let title = title.trim();
        if title.is_empty() {
            return Err(EngineError::Invalid("Enter a title".into()));
        }
        self.store.session_rename(id, title)
    }

    pub async fn session_set_model(&self, id: &str, model: &str) -> Result<()> {
        if split_model_id(model).is_none() {
            return Err(EngineError::Invalid(format!(
                "A model id looks like provider/model: {model}"
            )));
        }
        self.store.session_set_model(id, model)
    }

    pub async fn session_delete(&self, id: &str) -> Result<()> {
        self.runs.cancel(id).await;
        self.store.session_delete(id)
    }

    pub async fn message_list(&self, session_id: &str) -> Result<Vec<StoredMessage>> {
        self.store.session_get(session_id)?;
        self.store.message_list(session_id)
    }

    // ── runs ────────────────────────────────────────────────────────────────

    pub async fn run_start(&self, session_id: &str, parts: Vec<Part>) -> Result<RunHandle> {
        if parts.is_empty() {
            return Err(EngineError::Invalid("The message is empty".into()));
        }
        self.runs.start(session_id, parts).await
    }

    /// Re-subscribe to a run already in flight. The `String` is the assistant text streamed
    /// so far: a window that attaches mid-answer has to paint it before the first delta it
    /// receives, or the bubble starts in the middle of a sentence.
    pub async fn run_attach(&self, session_id: &str) -> Option<(RunHandle, String)> {
        self.runs.attach(session_id).await
    }

    pub async fn run_cancel(&self, session_id: &str) -> bool {
        self.runs.cancel(session_id).await
    }

    // ── workspace ───────────────────────────────────────────────────────────

    pub fn workspace_info(&self) -> WorkspaceInfo {
        self.workspace.info()
    }

    /// Tell the stores to ask their sources again.
    ///
    /// What a reader means by Refresh. A remote store keeps what it has read — a Notion page
    /// render answers a listing without a request, which is what makes a tree quick — and
    /// nothing about a page *deleted* at the source reaches it on its own. This is the one
    /// way to say so, and it is a person saying it.
    ///
    /// Nothing is fetched here: the stores drop what they kept, and the next read pays for
    /// what the reader is actually looking at.
    pub async fn workspace_refresh(&self) {
        self.workspace.fs().forget().await;
    }

    pub async fn fs_list(&self, path: &str) -> Result<Vec<Entry>> {
        fsops::list(&self.workspace.fs(), path).await
    }

    pub async fn fs_read(&self, path: &str) -> Result<FileContent> {
        fsops::read(&self.workspace.fs(), path).await
    }

    /// A file's bytes, for the viewers that open a format rather than read characters.
    ///
    /// Not a Tauri command: a PDF crossing the IPC bridge would be base64 in a JSON string,
    /// which is a third again in size and a copy at each end. The window reaches this
    /// through the app's own URI scheme instead, which streams and gives the webview
    /// something it can hand straight to an `<img>` or an `<object>`.
    pub async fn fs_read_bytes(&self, path: &str) -> Result<Vec<u8>> {
        fsops::read_bytes(&self.workspace.fs(), path).await
    }

    pub async fn fs_write(&self, path: &str, text: &str) -> Result<()> {
        fsops::write(&self.workspace.fs(), path, text).await
    }

    pub async fn fs_mkdir(&self, path: &str) -> Result<()> {
        fsops::mkdir(&self.workspace.fs(), path).await
    }

    pub async fn fs_delete(&self, path: &str) -> Result<()> {
        fsops::delete(&self.workspace.fs(), path).await
    }

    pub async fn fs_rename(&self, from: &str, to: &str) -> Result<()> {
        fsops::rename(&self.workspace.fs(), from, to).await
    }

    pub async fn fs_import(&self, dest: &str, sources: Vec<PathBuf>) -> Result<ImportReport> {
        fsops::import(&self.workspace.fs(), dest, sources).await
    }

    /// The mounts as they stand *right now*.
    ///
    /// Right after `start` that is the root alone: the stored connectors are probed on a
    /// background task and appear here one at a time as they come up, each either `Ok` or
    /// carrying its error. A client that wants to show them should poll this (or call
    /// [`Engine::wait_restored`] once) rather than assume the first answer is final.
    pub async fn mount_list(&self) -> Vec<MountInfo> {
        self.workspace.mounts().await
    }

    /// Probe, mount, then record. The store row is written last because a connector that is
    /// remembered but could not be mounted would come back on every launch as an error the
    /// user never asked for; when the row fails to write, the mount is undone.
    pub async fn mount_add(&self, req: MountRequest) -> Result<MountInfo> {
        let path = connectors::normalize_mount_path(&req.path)?;
        let (kind, detail, writable) = connectors::describe(&req.config);
        let label = req
            .label
            .map(|l| l.trim().to_string())
            .filter(|l| !l.is_empty())
            .unwrap_or_else(|| path.rsplit('/').next().unwrap_or(&path).to_string());
        let fs = connectors::build_and_probe(&req.config, &self.cfg.cache_dir()).await?;
        let info = MountInfo {
            id: uuid::Uuid::new_v4().to_string(),
            path: path.clone(),
            kind: kind.clone(),
            label: label.clone(),
            detail,
            writable,
            status: MountStatus::Ok,
        };
        self.workspace.attach(info.clone(), fs).await?;
        let row = MountRow {
            id: info.id.clone(),
            path,
            kind,
            label,
            config: req.config,
            writable,
            created_at: now_ms(),
        };
        if let Err(e) = self.store.mount_insert(&row) {
            let _ = self.workspace.detach(&row.path).await;
            return Err(e);
        }
        Ok(info)
    }

    /// Unmount, then forget. Both halves act on the same normal form: `detach` normalizes
    /// what it is given (and refuses the root in its own words), so the row has to be deleted
    /// under the normalized path too — deleting the caller's spelling would unmount `/docs`
    /// and leave a `"/docs/"` row in the database to come back on the next launch.
    pub async fn mount_remove(&self, path: &str) -> Result<()> {
        self.workspace.detach(path).await?;
        let path = connectors::normalize_mount_path(path)?;
        self.store.mount_delete(&path)
    }

    /// Point the workspace root at a different directory on the host.
    ///
    /// Swapped first, stored second: the store is what the next launch reads, so writing it
    /// for a directory the workspace refused would come back as a root that does not work
    /// and a window with no way to say why.
    pub async fn workspace_set_root(&self, path: &str) -> Result<WorkspaceInfo> {
        self.workspace
            .set_root(std::path::PathBuf::from(path))
            .await?;
        let info = self.workspace.info();
        self.store.setting_set(
            crate::config::ROOT_SETTING,
            &info.files_root.to_string_lossy(),
        )?;
        Ok(info)
    }

    // ── settings & catalog ──────────────────────────────────────────────────

    pub async fn settings_get(&self) -> Result<Settings> {
        let mut settings = providers::read_settings(&self.store)?;
        self.fill_routings(&mut settings);
        Ok(settings)
    }

    /// What the settings pane offers as routings is the catalog's answer, not the store's:
    /// `read_settings` reads what the user chose, and which profiles exist to choose from
    /// is a fact about the provider's model list. See `catalog::region_routings`.
    fn fill_routings(&self, settings: &mut Settings) {
        for p in &mut settings.providers {
            if p.routing.is_some() {
                let Some(def) = providers::PROVIDERS.iter().find(|d| d.key == p.key) else {
                    continue;
                };
                p.routings = catalog::region_routings(&self.catalog.models_for(def.ailoy_prefix));
            }
        }
    }

    /// Write, register, read back — all three under one lock, so the `Settings` returned is
    /// the one the registry was built from.
    pub async fn settings_set(&self, patch: SettingsPatch) -> Result<Settings> {
        let _guard = self.settings_lock.lock().await;
        providers::write_settings(&self.store, &patch)?;
        providers::apply(&self.store)?;
        // Turned on after a long while off, the list may be months old: look now rather
        // than when the loop's last sleep happens to end.
        if patch.catalog_refresh == Some(true) {
            self.catalog_wake.notify_one();
        }
        let mut settings = providers::read_settings(&self.store)?;
        self.fill_routings(&mut settings);
        Ok(settings)
    }

    pub fn catalog_status(&self) -> CatalogStatus {
        self.catalog.status()
    }

    /// Every change to [`Engine::catalog_status`], for the Tauri layer to pass on.
    pub fn catalog_subscribe(&self) -> tokio::sync::watch::Receiver<CatalogStatus> {
        self.catalog.subscribe()
    }

    /// Fetch the model list now, whether or not refreshing is on — this is the user asking.
    /// Answers with the status either way: a failure is the list staying as it was, and
    /// `error` says why.
    pub async fn models_refresh(&self) -> CatalogStatus {
        if let Err(e) = self.catalog.refresh(&self.catalog_cache).await {
            tracing::warn!("catalog refresh failed: {e:#}");
        }
        self.catalog.status()
    }

    /// Every model the catalog knows about, with the ones whose provider has a key first.
    /// `available` is about the key, not the model: a model listed here without one is shown
    /// so the settings pane can say what a key would buy.
    pub fn models_list(&self) -> Result<Vec<ModelInfo>> {
        let settings = providers::read_settings(&self.store)?;
        let mut out = Vec::new();
        for def in providers::PROVIDERS {
            let available = settings
                .providers
                .iter()
                .any(|p| p.key == def.key && p.has_key);
            let mut models = self.catalog.models_for(def.ailoy_prefix);
            // Bedrock lists a model once per inference profile. One row, reached through
            // the profile this provider's settings name.
            if let Some(routing) = settings
                .providers
                .iter()
                .find(|p| p.key == def.key)
                .and_then(|p| p.routing.as_deref())
            {
                models = catalog::fold_region_profiles(&models, routing);
            }
            for m in models {
                out.push(ModelInfo {
                    id: format!("{}/{}", def.ailoy_prefix, m.id),
                    provider: def.label.into(),
                    name: m.name,
                    context: m.context,
                    output: m.output,
                    cost: m.cost,
                    reasoning: m.reasoning,
                    tool_call: m.tool_call,
                    available,
                });
            }
        }
        out.sort_by(|a, b| b.available.cmp(&a.available).then(a.id.cmp(&b.id)));
        Ok(out)
    }

    pub async fn session_usage(&self, session_id: &str) -> Result<SessionUsage> {
        let session = self.store.session_get(session_id)?;
        let usages = self.store.message_usages(session_id)?;
        Ok(crate::usage::session_usage(
            &usages,
            self.catalog.lookup(&session.model).as_ref(),
        ))
    }

    pub fn config(&self) -> &EngineConfig {
        &self.cfg
    }
}

/// Keeps the model list fresh for as long as the engine runs.
///
/// Fetches when the list is older than [`catalog::REFRESH_EVERY`] — or missing, which on a
/// first start is right away — and then sleeps until it next will be, sooner after a
/// failure. The setting is read on every pass rather than once, so switching it off stops
/// the next fetch and switching it on (see `settings_set`) wakes the loop.
async fn refresh_catalog(
    catalog: Arc<Catalog>,
    store: Arc<Store>,
    cache: PathBuf,
    wake: Arc<tokio::sync::Notify>,
) {
    loop {
        // Read through `read_settings` rather than off the raw row, so "is the catalog
        // allowed to refresh" has exactly one definition — the one the settings pane shows.
        let enabled = providers::read_settings(&store).map_or(true, |s| s.catalog_refresh);
        let mut failed = false;
        if enabled
            && catalog::is_due(catalog.age())
            && let Err(e) = catalog.refresh(&cache).await
        {
            tracing::warn!("catalog refresh failed: {e:#}");
            failed = true;
        }
        match catalog::next_check(enabled, catalog.age(), failed) {
            Some(wait) => {
                tokio::select! {
                    _ = tokio::time::sleep(wait) => {}
                    _ = wake.notified() => {}
                }
            }
            None => wake.notified().await,
        }
    }
}

/// Take the exclusive advisory lock that makes one data directory mean one engine.
///
/// Two processes over one SQLite file would each hold their own WAL connection and their
/// own workspace, and the second would try to mount over the first's mount point; ailoy's
/// provider registry is process-wide, so they would also disagree about which keys are
/// registered. The lock is on a file of its own rather than on the database, because the
/// database is opened and closed by tooling (`sqlite3`, a backup) that has no business
/// being locked out.
///
/// It is an `flock` (`LockFileEx` on Windows), so it is tied to this open file description:
/// the kernel drops it when the file closes, which covers a clean shutdown, a panic, and a
/// kill alike. No stale lock file to clean up, and no PID to second-guess.
///
/// `std::fs::File::try_lock` is what takes it — stable since Rust 1.89, well under this
/// workspace's 1.95 MSRV, so the `fs4` crate the review suggested would only shadow an
/// inherent method that already does the same thing on the same syscalls.
fn lock_data_dir(data_dir: &std::path::Path) -> Result<std::fs::File> {
    let path = data_dir.join("engine.lock");
    let file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(&path)?;
    file.try_lock().map_err(|_| {
        EngineError::Invalid("Another Ailoy instance is using this data directory".into())
    })?;
    Ok(file)
}

/// Rebuild every stored connector into the live workspace.
///
/// One that cannot be rebuilt shows its error in the mount list instead of taking the
/// workspace down — the sidebar is where the user finds out, and the root mount keeps
/// serving files either way.
///
/// Probed concurrently, attached in row order: the probes are what take 15 seconds each
/// when a store does not answer, and the attaching takes the workspace's write lock, where
/// order decides which of two rows on the same path wins.
async fn restore_connectors(store: &Store, workspace: &WorkspaceManager, cache_dir: &Path) {
    let rows = match store.mount_list() {
        Ok(rows) => rows,
        Err(e) => {
            tracing::error!("reading the stored connectors: {e}");
            return;
        }
    };
    let probes = futures::future::join_all(
        rows.iter()
            .map(|r| connectors::build_and_probe(&r.config, cache_dir)),
    )
    .await;
    for (row, probe) in rows.into_iter().zip(probes) {
        let (kind, detail, writable) = connectors::describe(&row.config);
        let mut info = MountInfo {
            id: row.id.clone(),
            path: row.path.clone(),
            kind,
            label: row.label.clone(),
            detail,
            writable,
            status: MountStatus::Ok,
        };
        // The stored path goes back through the normalizer on the way in. `mount_add`
        // writes the normal form, but a row from an older build — or a hand-edited
        // database — can hold a spelling `ContextFs` collapses to something else, and
        // mounting under it would key the tree by one path and the sidebar by another,
        // leaving a connector that cannot be removed. Such a row is listed with its
        // error, the same as one whose store did not answer.
        let failure = match connectors::normalize_mount_path(&row.path) {
            Ok(path) => {
                info.path = path;
                match probe {
                    Ok(fs) => {
                        // The probes took up to a connector timeout; the user may have
                        // removed this row (or re-added the path) meanwhile. Attaching
                        // anyway would resurrect a connector with no row behind it.
                        let current = store.mount_list().unwrap_or_default();
                        let mounted = workspace.mounts().await;
                        if !still_wanted(&current, &mounted, &row.id, &info.path) {
                            tracing::debug!(
                                "skipping restore of {}: removed or re-added meanwhile",
                                info.path
                            );
                            continue;
                        }
                        workspace.attach(info.clone(), fs).await.err()
                    }
                    Err(e) => Some(e),
                }
            }
            Err(e) => Some(e),
        };
        if let Some(e) = failure {
            info.status = MountStatus::Error {
                message: e.to_string(),
            };
            workspace.remember_failed(info).await;
        }
    }
}

/// Whether a stored connector should still be attached once its probe has answered: its
/// row must still exist (a `mount_remove` during the probe deletes it) and nothing may
/// already be mounted at the path (the user re-added it while the restore was running).
fn still_wanted(current: &[MountRow], mounted: &[MountInfo], row_id: &str, path: &str) -> bool {
    current.iter().any(|r| r.id == row_id) && !mounted.iter().any(|m| m.path == path)
}

/// Tauri's managed state is shared across the command threads, so an `Engine` that is not
/// `Send + Sync` does not compile in the app — a long way from here, in another crate, with
/// the failure pointing at `app.manage(..)` rather than at whatever field lost the bound.
/// Asserted here so the field that breaks it is the thing that fails to compile.
const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Engine>();
};

#[cfg(test)]
mod tests {
    use super::*;

    /// An engine with nothing outside the temp directory: no FUSE-T mount, no catalog
    /// refresh over the network, no console process. Its model list is the catalog
    /// fixture, put where a fetch would have cached it — a test build embeds none.
    async fn engine() -> (tempfile::TempDir, Arc<Engine>) {
        let dir = tempfile::tempdir().unwrap();
        let cfg = config(dir.path());
        std::fs::create_dir_all(cfg.cache_dir()).unwrap();
        std::fs::write(
            cfg.cache_dir().join("models.json"),
            include_str!("../testdata/catalog.json"),
        )
        .unwrap();
        let e = Engine::start(cfg).await.unwrap();
        (dir, e)
    }

    fn config(data_dir: &std::path::Path) -> EngineConfig {
        let mut cfg = EngineConfig::new(data_dir);
        cfg.mount_workspace = false;
        cfg.catalog_refresh = false;
        cfg.console = false;
        cfg
    }

    // `Engine::start` and `settings_set` both call `providers::apply`, which writes ailoy's
    // process-wide `"default"` registry. The std guard is deliberate: it is what serializes
    // these against every other registry-touching test in the binary, and a `#[tokio::test]`
    // current-thread runtime cannot deadlock on it.
    #[allow(clippy::await_holding_lock)]
    #[tokio::test]
    async fn sessions_settings_and_mounts_round_trip_through_the_engine() {
        let _g = crate::providers::REGISTRY_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let (_dir, e) = engine().await;
        let s = e.session_create(None).await.unwrap();
        assert_eq!(s.model, providers::DEFAULT_MODEL);
        assert_eq!(e.session_list().await.unwrap().len(), 1);
        e.session_rename(&s.id, "renamed").await.unwrap();
        assert_eq!(e.session_list().await.unwrap()[0].title, "renamed");

        let mut patch = SettingsPatch::default();
        patch
            .provider_keys
            .insert("anthropic".into(), Some("sk-ant-test-1234".into()));
        let settings = e.settings_set(patch).await.unwrap();
        assert!(
            settings
                .providers
                .iter()
                .any(|p| p.key == "anthropic" && p.has_key)
        );
        let models = e.models_list().unwrap();
        assert!(
            models
                .iter()
                .any(|m| m.id.starts_with("anthropic/") && m.available)
        );
        assert!(
            models
                .iter()
                .filter(|m| m.id.starts_with("openai/"))
                .all(|m| !m.available)
        );

        let local = tempfile::tempdir().unwrap();
        std::fs::write(local.path().join("f.txt"), b"hey").unwrap();
        let info = e
            .mount_add(MountRequest {
                path: "docs".into(),
                label: None,
                config: MountConfig::Local {
                    host_root: local.path().to_path_buf(),
                },
            })
            .await
            .unwrap();
        assert_eq!(info.path, "/docs");
        assert_eq!(info.label, "docs");
        assert_eq!(
            e.fs_read("/docs/f.txt").await.unwrap().text.as_deref(),
            Some("hey")
        );
        assert_eq!(e.mount_list().await.len(), 2);
        e.mount_remove("/docs").await.unwrap();
        assert_eq!(e.mount_list().await.len(), 1);
        e.shutdown().await;
    }

    #[allow(clippy::await_holding_lock)]
    #[tokio::test]
    async fn mounts_are_restored_on_restart() {
        let _g = crate::providers::REGISTRY_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let dir = tempfile::tempdir().unwrap();
        let local = tempfile::tempdir().unwrap();
        {
            let e = Engine::start(config(dir.path())).await.unwrap();
            e.mount_add(MountRequest {
                path: "/data".into(),
                label: Some("data".into()),
                config: MountConfig::Local {
                    host_root: local.path().to_path_buf(),
                },
            })
            .await
            .unwrap();
            e.shutdown().await;
        }
        let e = Engine::start(config(dir.path())).await.unwrap();
        // The restore runs off the startup path now, so the settled list is what this
        // asserts against — `mount_list()` right after `start` is allowed to be short.
        e.wait_restored().await;
        let mounts = e.mount_list().await;
        assert!(
            mounts.iter().any(|m| m.path == "/data"
                && m.label == "data"
                && matches!(m.status, MountStatus::Ok)),
            "{mounts:?}"
        );

        // Removed by a spelling that is not the stored one: the row is keyed by the normal
        // form, so this has to delete it and not just unmount it for this run.
        e.mount_remove("/data/").await.unwrap();
        assert_eq!(e.mount_list().await.len(), 1, "the root row");
        e.shutdown().await;
        drop(e);

        let e = Engine::start(config(dir.path())).await.unwrap();
        e.wait_restored().await;
        assert_eq!(
            e.mount_list().await.len(),
            1,
            "a removed connector came back from the database"
        );
        e.shutdown().await;
    }

    /// Every refusal the facade can reach without a model call. These are the arguments a
    /// webview can send — a session id from a stale list, a model id typed into a picker, a
    /// composer submitted empty — and each has to come back as its own error rather than as
    /// a run that starts and then fails somewhere the user cannot read.
    #[allow(clippy::await_holding_lock)]
    #[tokio::test]
    async fn bad_arguments_are_refused_at_the_facade() {
        let _g = crate::providers::REGISTRY_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let (_dir, e) = engine().await;

        let err = e.message_list("no-such-session").await.unwrap_err();
        assert!(matches!(err, EngineError::NotFound(_)), "{err:?}");

        let s = e.session_create(None).await.unwrap();
        let err = e.session_set_model(&s.id, "garbage").await.unwrap_err();
        assert!(matches!(err, EngineError::Invalid(_)), "{err:?}");
        assert_eq!(
            e.session_list().await.unwrap()[0].model,
            providers::DEFAULT_MODEL,
            "a rejected model must not have been written"
        );

        let err = e.session_create(Some("garbage".into())).await.unwrap_err();
        assert!(matches!(err, EngineError::Invalid(_)), "{err:?}");
        assert_eq!(
            e.session_list().await.unwrap().len(),
            1,
            "a rejected model must not have created a session"
        );

        // `matches!` rather than `unwrap_err`: a `RunHandle` is a live subscription, not a
        // value to format, and it does not implement `Debug`.
        assert!(
            matches!(
                e.run_start(&s.id, Vec::new()).await,
                Err(EngineError::Invalid(_))
            ),
            "an empty message should have been refused"
        );
        assert!(!e.session_list().await.unwrap()[0].running);

        e.shutdown().await;
    }

    /// One data directory, one engine. Two would fight over the SQLite WAL, the mount
    /// point and ailoy's process-wide registry — and the second one is what the user sees,
    /// so it is the one that has to say what is wrong.
    #[allow(clippy::await_holding_lock)]
    #[tokio::test]
    async fn a_second_engine_on_the_same_data_directory_is_refused() {
        let _g = crate::providers::REGISTRY_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let dir = tempfile::tempdir().unwrap();
        let first = Engine::start(config(dir.path())).await.unwrap();

        let err = Engine::start(config(dir.path()))
            .await
            .err()
            .expect("a second engine on the same data directory");
        assert!(matches!(err, EngineError::Invalid(_)), "{err:?}");
        assert!(err.to_string().contains("Another Ailoy instance"), "{err}");

        // And the lock is the file handle, not a marker to clean up: once the first engine
        // is gone the directory is free again.
        first.shutdown().await;
        drop(first);
        let third = Engine::start(config(dir.path())).await.unwrap();
        third.shutdown().await;
    }

    /// A connector whose store is gone comes back as a row with its error, not as a missing
    /// mount and not as a failed start — the sidebar is where the user finds out.
    #[allow(clippy::await_holding_lock)]
    #[tokio::test]
    async fn a_connector_that_cannot_be_rebuilt_is_listed_with_its_error() {
        let _g = crate::providers::REGISTRY_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let dir = tempfile::tempdir().unwrap();
        let local = tempfile::tempdir().unwrap();
        {
            let e = Engine::start(config(dir.path())).await.unwrap();
            e.mount_add(MountRequest {
                path: "/gone".into(),
                label: None,
                config: MountConfig::Local {
                    host_root: local.path().to_path_buf(),
                },
            })
            .await
            .unwrap();
            e.shutdown().await;
        }
        drop(local); // the directory the connector points at is deleted

        let e = Engine::start(config(dir.path())).await.unwrap();
        e.wait_restored().await;
        let mounts = e.mount_list().await;
        let row = mounts.iter().find(|m| m.path == "/gone").expect("the row");
        assert!(
            matches!(&row.status, MountStatus::Error { message } if !message.is_empty()),
            "{:?}",
            row.status
        );
        // The workspace itself is untouched: its own root still serves files.
        e.fs_write("/still-here.txt", "root intact").await.unwrap();
        assert_eq!(
            e.fs_read("/still-here.txt").await.unwrap().text.as_deref(),
            Some("root intact")
        );
        e.shutdown().await;
    }

    #[test]
    fn a_restore_skips_rows_removed_or_re_added_meanwhile() {
        let row = |id: &str, path: &str| MountRow {
            id: id.into(),
            path: path.into(),
            kind: MountKind::Local,
            label: "l".into(),
            config: MountConfig::Local {
                host_root: "/tmp".into(),
            },
            writable: true,
            created_at: 0,
        };
        let mounted = |path: &str| MountInfo {
            id: "x".into(),
            path: path.into(),
            kind: MountKind::Local,
            label: "l".into(),
            detail: String::new(),
            writable: true,
            status: MountStatus::Ok,
        };
        let root = mounted("/");
        // Still stored, nothing at the path: attach.
        assert!(still_wanted(
            &[row("a", "/docs")],
            std::slice::from_ref(&root),
            "a",
            "/docs"
        ));
        // Row deleted during the probe: skip.
        assert!(!still_wanted(
            &[],
            std::slice::from_ref(&root),
            "a",
            "/docs"
        ));
        // Re-added at the same path while probing: skip.
        assert!(!still_wanted(
            &[row("a", "/docs")],
            &[root, mounted("/docs")],
            "a",
            "/docs"
        ));
    }
}
