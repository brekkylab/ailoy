//! A context: a directory of files an agent reads and never writes — mounted read-only.
//!
//! The tree is mounted as it stands, so what we know about it lives beside it rather than
//! in it: `contexts/{id}/` is the tree, `contexts/{id}.json` its name.
//!
//! One context is always there: [`DEFAULT_ID`], made at every start it is missing from, so
//! the default agent has somewhere to look before anyone has made a context of their own.
//! It is not to be deleted — anything that deletes a context refuses it (see
//! [`Context::is_default`]).

use std::{
    fs, io,
    path::{Path, PathBuf},
    time::UNIX_EPOCH,
};

use cortex::console::{InvalidMount, MountSpec};
use serde::{Deserialize, Serialize};
use tauri::State;

use crate::cache::Cache;

/// The id of the context that is always there.
pub const DEFAULT_ID: &str = "default";
const DEFAULT_NAME: &str = "Default";

#[derive(Clone, Debug, Serialize)]
pub struct Context {
    pub id: String,
    pub name: String,
    /// The tree itself, absolute.
    pub dir: PathBuf,
    /// Whether this is [`DEFAULT_ID`], so the frontend need not know the id.
    pub default: bool,
}

/// One entry of a directory in a context's tree, as it is on disk.
#[derive(Debug, Serialize)]
pub struct FileEntry {
    pub name: String,
    pub kind: FileKind,
    /// Bytes; 0 for a folder.
    pub size: u64,
    /// Milliseconds since the epoch, when the platform says.
    pub modified: Option<u64>,
}

#[derive(Debug, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum FileKind {
    Folder,
    File,
    /// Shown as what it is, not followed: the tree is displayed as it stands.
    Link,
}

/// What `{id}.json` holds.
#[derive(Serialize, Deserialize)]
struct Meta {
    name: String,
}

impl Context {
    /// The context whose tree is `dir`, read from the `.json` beside it. `None` when either
    /// half is missing; an error when the file is there but is not one.
    pub fn load(dir: &Path) -> io::Result<Option<Self>> {
        let Some(id) = dir.file_name().and_then(|n| n.to_str()) else {
            return Ok(None);
        };
        if !dir.is_dir() {
            return Ok(None);
        }
        let bytes = match fs::read(dir.with_extension("json")) {
            Ok(bytes) => bytes,
            Err(err) if err.kind() == io::ErrorKind::NotFound => return Ok(None),
            Err(err) => return Err(err),
        };
        let Meta { name } = serde_json::from_slice(&bytes)?;
        Ok(Some(Self {
            default: id == DEFAULT_ID,
            id: id.to_owned(),
            name,
            dir: dir.to_owned(),
        }))
    }

    /// The one context nothing may delete.
    pub fn is_default(&self) -> bool {
        self.default
    }

    /// A new, empty context named `name` under `contexts`. The tree is made first and the
    /// name last, so one cut short is a bare directory that `load` passes over.
    pub fn create(contexts: &Path, name: &str) -> io::Result<Self> {
        Self::create_as(contexts, uuid::Uuid::new_v4().to_string(), name)
    }

    /// The default context, made if it is not there — whether this is the first start or
    /// its tree or name went missing since. What is there is left as it is.
    pub fn seed(contexts: &Path) -> io::Result<Self> {
        let dir = contexts.join(DEFAULT_ID);
        if let Some(context) = Self::load(&dir)? {
            return Ok(context);
        }
        // A tree whose name went missing keeps its files: only the name is written again.
        if !dir.is_dir() {
            return Self::create_as(contexts, DEFAULT_ID.to_owned(), DEFAULT_NAME);
        }
        Self::name(&dir, DEFAULT_NAME)?;
        Ok(Self::load(&dir)?.expect("the context just written"))
    }

    fn create_as(contexts: &Path, id: String, name: &str) -> io::Result<Self> {
        let dir = contexts.join(&id);
        fs::create_dir(&dir)?;
        Self::name(&dir, name)?;
        Ok(Self {
            default: id == DEFAULT_ID,
            id,
            name: name.to_owned(),
            dir,
        })
    }

    /// Writes the `.json` beside the tree at `dir`.
    fn name(dir: &Path, name: &str) -> io::Result<()> {
        let meta = dir.with_extension("json");
        // Beside the target and renamed over it, so a reader never sees half a file.
        let tmp = dir.with_extension("json.tmp");
        fs::write(
            &tmp,
            serde_json::to_vec_pretty(&Meta {
                name: name.to_owned(),
            })?,
        )?;
        fs::rename(&tmp, &meta)
    }

    /// The path under the tree that `path` names, one segment at a time. A segment that is
    /// empty, `.`, `..` or carries a separator could reach outside the tree, so it is
    /// refused rather than resolved.
    fn resolve(&self, path: &[String]) -> io::Result<PathBuf> {
        let mut at = self.dir.clone();
        for segment in path {
            let plain = !segment.is_empty()
                && segment != "."
                && segment != ".."
                && !segment.contains(['/', '\\', '\0']);
            if !plain {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("not a file name: {segment:?}"),
                ));
            }
            at.push(segment);
        }
        Ok(at)
    }

    /// The directory at `path`, folders first and then by name.
    pub fn list(&self, path: &[String]) -> io::Result<Vec<FileEntry>> {
        let mut entries = Vec::new();
        for item in fs::read_dir(self.resolve(path)?)? {
            let item = item?;
            // Not `metadata`: a link is listed as a link, never as whatever it points at.
            let meta = item.metadata()?;
            let kind = if meta.is_symlink() {
                FileKind::Link
            } else if meta.is_dir() {
                FileKind::Folder
            } else {
                FileKind::File
            };
            entries.push(FileEntry {
                name: item.file_name().to_string_lossy().into_owned(),
                size: if kind == FileKind::File {
                    meta.len()
                } else {
                    0
                },
                modified: meta
                    .modified()
                    .ok()
                    .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
                    .map(|d| d.as_millis() as u64),
                kind,
            });
        }
        entries.sort_by(|a, b| {
            (a.kind != FileKind::Folder)
                .cmp(&(b.kind != FileKind::Folder))
                .then_with(|| a.name.cmp(&b.name))
        });
        Ok(entries)
    }

    /// A new, empty folder at `path`.
    pub fn make_dir(&self, path: &[String]) -> io::Result<()> {
        fs::create_dir(self.resolve(path)?)
    }

    /// Copies each of `sources` — files or whole folders, from anywhere on this machine —
    /// into the directory at `path`, under its own name. Nothing is overwritten: a name
    /// already taken stops the copy there.
    pub fn add(&self, path: &[String], sources: &[PathBuf]) -> io::Result<()> {
        let into = self.resolve(path)?;
        for source in sources {
            let name = source.file_name().ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("no file name: {}", source.display()),
                )
            })?;
            copy_new(source, &into.join(name))?;
        }
        Ok(())
    }

    /// The file or folder at `path`, and everything under it. A link goes, not what it
    /// points at.
    pub fn remove(&self, path: &[String]) -> io::Result<()> {
        if path.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "the tree itself cannot be removed here",
            ));
        }
        let at = self.resolve(path)?;
        if fs::symlink_metadata(&at)?.is_dir() {
            fs::remove_dir_all(at)
        } else {
            fs::remove_file(at)
        }
    }

    /// The tree written to `to` as a gzipped tar, under one folder named after the context.
    /// Links are archived as links, never followed out of the tree. Written beside `to` and
    /// renamed over it, so a failed export leaves no half an archive behind.
    pub fn export(&self, to: &Path) -> io::Result<()> {
        let root: String = self
            .name
            .chars()
            .map(|c| {
                if matches!(c, '/' | '\\' | '\0') {
                    '_'
                } else {
                    c
                }
            })
            .collect();
        let root = match root.trim() {
            "" | "." | ".." => self.id.clone(),
            name => name.to_owned(),
        };
        let mut tmp = to.as_os_str().to_owned();
        tmp.push(".partial");
        let tmp = PathBuf::from(tmp);
        let result = (|| {
            let gz = flate2::write::GzEncoder::new(
                fs::File::create(&tmp)?,
                flate2::Compression::default(),
            );
            let mut tar = tar::Builder::new(gz);
            tar.follow_symlinks(false);
            tar.append_dir_all(&root, &self.dir)?;
            tar.into_inner()?.finish()?.sync_all()
        })();
        match result {
            Ok(()) => fs::rename(&tmp, to),
            Err(err) => {
                let _ = fs::remove_file(&tmp);
                Err(err)
            }
        }
    }

    /// The tree mounted read-only at `at` in the guest.
    pub fn mount(&self, at: &str) -> Result<MountSpec, InvalidMount> {
        MountSpec::new(format!("file://{}", self.dir.display()), at).map(MountSpec::read_only)
    }
}

/// `from` copied to `to`, which must not exist yet. A folder is copied whole; a link
/// inside one is left behind, since following it could copy half the disk or loop.
fn copy_new(from: &Path, to: &Path) -> io::Result<()> {
    if to.symlink_metadata().is_ok() {
        return Err(io::Error::new(
            io::ErrorKind::AlreadyExists,
            format!(
                "{} is already there",
                to.file_name().unwrap_or_default().to_string_lossy()
            ),
        ));
    }
    if fs::metadata(from)?.is_dir() {
        fs::create_dir(to)?;
        for item in fs::read_dir(from)? {
            let item = item?;
            if item.file_type()?.is_symlink() {
                continue;
            }
            copy_new(&item.path(), &to.join(item.file_name()))?;
        }
        Ok(())
    } else {
        fs::copy(from, to).map(|_| ())
    }
}

#[tauri::command]
pub fn list_contexts(cache: State<'_, Cache>) -> Vec<Context> {
    cache.contexts()
}

#[tauri::command]
pub fn create_context(cache: State<'_, Cache>, name: String) -> Result<Context, String> {
    let name = name.trim();
    if name.is_empty() {
        return Err("a context needs a name".into());
    }
    cache.create_context(name).map_err(|err| err.to_string())
}

/// The context `id`, or the sentence the frontend shows when there is none.
fn find(cache: &Cache, id: &str) -> Result<Context, String> {
    cache.context(id).ok_or_else(|| format!("no context {id}"))
}

#[tauri::command]
pub fn list_context_files(
    cache: State<'_, Cache>,
    id: String,
    path: Vec<String>,
) -> Result<Vec<FileEntry>, String> {
    find(&cache, &id)?
        .list(&path)
        .map_err(|err| err.to_string())
}

#[tauri::command]
pub fn make_context_dir(
    cache: State<'_, Cache>,
    id: String,
    path: Vec<String>,
) -> Result<(), String> {
    find(&cache, &id)?
        .make_dir(&path)
        .map_err(|err| err.to_string())
}

#[tauri::command]
pub fn add_context_files(
    cache: State<'_, Cache>,
    id: String,
    path: Vec<String>,
    sources: Vec<PathBuf>,
) -> Result<(), String> {
    find(&cache, &id)?
        .add(&path, &sources)
        .map_err(|err| err.to_string())
}

#[tauri::command]
pub fn remove_context_file(
    cache: State<'_, Cache>,
    id: String,
    path: Vec<String>,
) -> Result<(), String> {
    find(&cache, &id)?
        .remove(&path)
        .map_err(|err| err.to_string())
}

#[tauri::command]
pub fn export_context(cache: State<'_, Cache>, id: String, to: PathBuf) -> Result<(), String> {
    find(&cache, &id)?
        .export(&to)
        .map_err(|err| err.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("ailoy-context-{name}-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn seed_makes_the_default_once_and_keeps_its_files() {
        let root = scratch("seed");
        let made = Context::seed(&root).unwrap();
        assert!(made.is_default());
        assert_eq!(made.name, DEFAULT_NAME);
        fs::write(made.dir.join("notes.md"), "kept").unwrap();

        // Seeded again, it is left as it is.
        Context::seed(&root).unwrap();
        assert!(made.dir.join("notes.md").exists());

        // Its name gone, the name comes back and the files stay.
        fs::remove_file(made.dir.with_extension("json")).unwrap();
        let again = Context::seed(&root).unwrap();
        assert!(again.is_default());
        assert!(again.dir.join("notes.md").exists());

        // Another context is not the default.
        assert!(!Context::create(&root, "Other").unwrap().is_default());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn export_archives_the_tree_under_its_name() {
        let root = scratch("export");
        let context = Context::create(&root, "Notes/2026").unwrap();
        fs::create_dir(context.dir.join("sub")).unwrap();
        fs::write(context.dir.join("sub/a.md"), "hello").unwrap();

        let to = root.join("out.tar.gz");
        context.export(&to).unwrap();
        assert!(!root.join("out.tar.gz.partial").exists());

        let gz = flate2::read::GzDecoder::new(fs::File::open(&to).unwrap());
        let mut names: Vec<String> = tar::Archive::new(gz)
            .entries()
            .unwrap()
            .map(|e| e.unwrap().path().unwrap().to_string_lossy().into_owned())
            .collect();
        names.sort();
        assert_eq!(names, ["Notes_2026/", "Notes_2026/sub", "Notes_2026/sub/a.md"]);
        fs::remove_dir_all(root).unwrap();
    }
}
