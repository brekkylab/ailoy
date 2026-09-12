//! The engine: everything the window can ask for, behind one `Arc`.
//!
//! Every other module in this crate answers one question — where the sessions are, what the
//! workspace serves, which providers ailoy knows about. This is the object that holds them
//! together and is the only one the Tauri layer ever sees: one `Arc<Engine>` in managed
//! state, one method per command.

use std::{path::PathBuf, sync::Arc};

use ailoy::message::Part;

use crate::{
    catalog::{Catalog, split_model_id},
    config::EngineConfig,
    console::{ConsoleFactory, resolve_console_bin},
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
    runs: RunManager,
    /// Held across the whole of `settings_set`. The store is transactional per statement and
    /// `providers::apply` reads it before touching ailoy's registry, so two settings commands
    /// arriving together could otherwise interleave as write(A) → write(B) → apply(B) →
    /// apply(A) and leave the registry holding a key the store no longer has.
    settings_lock: tokio::sync::Mutex<()>,
}

impl Engine {
    pub async fn start(cfg: EngineConfig) -> Result<Arc<Engine>> {
        std::fs::create_dir_all(&cfg.data_dir)?;
        let store = Arc::new(Store::open(&cfg.db_path())?);
        let workspace = Arc::new(
            WorkspaceManager::start(cfg.files_root(), cfg.mountpoint(), cfg.mount_workspace).await,
        );

        // Connectors come back from the database; one that cannot be rebuilt shows its error
        // in the list instead of taking the workspace down.
        for row in store.mount_list()? {
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
            match connectors::build_and_probe(&row.config).await {
                Ok(fs) => {
                    if let Err(e) = workspace.attach(info.clone(), fs).await {
                        info.status = MountStatus::Error {
                            message: e.to_string(),
                        };
                        workspace.remember_failed(info).await;
                    }
                }
                Err(e) => {
                    info.status = MountStatus::Error {
                        message: e.to_string(),
                    };
                    workspace.remember_failed(info).await;
                }
            }
        }

        let cache = cfg.cache_dir().join("models.json");
        let catalog = Arc::new(Catalog::load(Some(&cache)));
        let refresh = cfg.catalog_refresh
            && store
                .setting_get("catalog_refresh")?
                .map(|v| v != "false")
                .unwrap_or(true);
        if refresh {
            let catalog = catalog.clone();
            tokio::spawn(async move {
                if let Err(e) = catalog.refresh(&cache).await {
                    tracing::warn!("catalog refresh failed: {e}");
                }
            });
        }

        // Not fatal: a legacy or hand-edited settings row that fails to parse would otherwise
        // block startup, and the settings pane that could fix it is behind a started engine.
        // `settings_set` still surfaces the same error when the user edits.
        if let Err(e) = providers::apply(&store) {
            tracing::warn!("applying provider settings at start: {e}");
        }

        // An empty `console_bin` is the caller saying "no console" — that is how the tests and
        // a headless run ask for it. Otherwise the binary is looked for, and *not* finding one
        // is still a start: a window with no console can list sessions, browse the workspace
        // and change settings, and only the tools that need a shell fail, saying so.
        let console = Arc::new(match cfg.console_bin.as_deref() {
            Some(p) if p.as_os_str().is_empty() => ConsoleFactory::disabled(),
            explicit => match resolve_console_bin(explicit) {
                Ok(bin) => ConsoleFactory::new(bin),
                Err(e) => {
                    tracing::warn!("{e}");
                    ConsoleFactory::disabled()
                }
            },
        });

        let runs = RunManager::new(RunDeps {
            store: store.clone(),
            console,
            workspace: workspace.clone(),
            catalog: catalog.clone(),
        });
        Ok(Arc::new(Engine {
            cfg,
            store,
            workspace,
            catalog,
            runs,
            settings_lock: tokio::sync::Mutex::new(()),
        }))
    }

    /// Ask every run to stop, give them a moment to write what they have, then take the
    /// workspace down. `cancel_all` only signals — a run ends on its own terms, and the
    /// grace period is what lets the partial answer reach SQLite before the process exits.
    pub async fn shutdown(&self) {
        self.runs.cancel_all().await;
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
        self.workspace.shutdown().await;
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
            Some(m) if !m.trim().is_empty() => m,
            _ => providers::read_settings(&self.store)?.default_model,
        };
        let id = uuid::Uuid::new_v4().to_string();
        let r = self.store.session_create(&id, "새 대화", &model)?;
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
            return Err(EngineError::Invalid("제목을 입력해 주세요".into()));
        }
        self.store.session_rename(id, title)
    }

    pub async fn session_set_model(&self, id: &str, model: &str) -> Result<()> {
        if split_model_id(model).is_none() {
            return Err(EngineError::Invalid(format!(
                "모델 ID 형식은 provider/model 입니다: {model}"
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
            return Err(EngineError::Invalid("메시지가 비어 있습니다".into()));
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

    pub async fn fs_list(&self, path: &str) -> Result<Vec<Entry>> {
        fsops::list(&self.workspace.fs(), path).await
    }

    pub async fn fs_read(&self, path: &str) -> Result<FileContent> {
        fsops::read(&self.workspace.fs(), path).await
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
        let fs = connectors::build_and_probe(&req.config).await?;
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

    // ── settings & catalog ──────────────────────────────────────────────────

    pub async fn settings_get(&self) -> Result<Settings> {
        providers::read_settings(&self.store)
    }

    /// Write, register, read back — all three under one lock, so the `Settings` returned is
    /// the one the registry was built from.
    pub async fn settings_set(&self, patch: SettingsPatch) -> Result<Settings> {
        let _guard = self.settings_lock.lock().await;
        providers::write_settings(&self.store, &patch)?;
        providers::apply(&self.store)?;
        providers::read_settings(&self.store)
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
            for m in self.catalog.models_for(def.ailoy_prefix) {
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

#[cfg(test)]
mod tests {
    use super::*;

    /// An engine with nothing outside the temp directory: no FUSE-T mount, no catalog
    /// refresh over the network, no console process.
    async fn engine() -> (tempfile::TempDir, Arc<Engine>) {
        let dir = tempfile::tempdir().unwrap();
        let e = Engine::start(config(dir.path())).await.unwrap();
        (dir, e)
    }

    fn config(data_dir: &std::path::Path) -> EngineConfig {
        let mut cfg = EngineConfig::new(data_dir);
        cfg.mount_workspace = false;
        cfg.catalog_refresh = false;
        cfg.console_bin = Some(PathBuf::new()); // disabled console
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
                label: Some("데이터".into()),
                config: MountConfig::Local {
                    host_root: local.path().to_path_buf(),
                },
            })
            .await
            .unwrap();
            e.shutdown().await;
        }
        let e = Engine::start(config(dir.path())).await.unwrap();
        let mounts = e.mount_list().await;
        assert!(
            mounts.iter().any(|m| m.path == "/data"
                && m.label == "데이터"
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
        assert_eq!(
            e.mount_list().await.len(),
            1,
            "a removed connector came back from the database"
        );
        e.shutdown().await;
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
}
