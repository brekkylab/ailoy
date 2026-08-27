//! GLEIF's LEI API projected onto a read-only [`FileSystem`].
//!
//! ```text
//! /by-lei/<LEI>/record.json              the LEI record
//! /by-lei/<LEI>/<resource>.json          whatever that record's `relationships` links to
//! /search/<field>/<value>/…/           a filtered query, narrowed by descending
//! /search/<field>/<value>/…/pages/     its results, one file per page
//! ```
//!
//! # Why the search path is nested
//!
//! GLEIF combines filters with repeated `filter[..]` parameters and reads a comma
//! inside one value as OR. So a comma is already taken, and AND has to be spelled
//! some other way — here it is directory nesting, which leaves the comma meaning
//! exactly what it means upstream:
//!
//! ```text
//! /search/entity.legalAddress.country/KR,JP/entity.status/ACTIVE/
//!         └─ OR inside the value ─┘  └─ AND by descending ─┘
//! ```
//!
//! Splitting `field` and `value` into two segments is what makes the tree
//! self-describing: listing `/search` yields the field names, which is the part a
//! caller cannot guess. The attribute paths that look plausible and answer 400
//! (`entity.legalAddress.city`, every `headquartersAddress`, every date) are simply
//! absent, so a wrong field is a missing directory rather than a request.
//!
//! Values are not listable — they are unbounded — so a value directory is reached by
//! naming it. That is a lookup with the key already in hand: countries, statuses and
//! LEIs all come from a file the caller has read.
//!
//! # What costs a request
//!
//! Narrowing does not. The fields are known here and the field listing is a constant,
//! so walking down `country/KR/entity.status/ACTIVE/…` asks the API nothing — which
//! matters because narrowing is the step a caller repeats. The results sit one level
//! further down, behind `pages`, and that listing is the one that has to ask: how many
//! pages a query has is something only the API knows. It answers with page 1, which is
//! cached, so reading `pages/page-001.json` afterwards is free.
//!
//! # Sizes
//!
//! A kernel asks for a size before it opens anything, and an API answers only when
//! asked. So a fetch fills a cache and the reads that follow are served from it —
//! otherwise the file would appear to end wherever the first read stopped. The cache
//! lasts as long as the store, which is one run: a citation written at turn 5 has to
//! still mean the same bytes at turn 40.

use std::{collections::HashMap, io, path::{Component, Path}, sync::Arc};

use cortex::{
    BoxFuture,
    fs::{Dirent, DirentKind, FileSystem, Stat},
};

use crate::apifs::{Fetcher, io_err, percent_decode, urlencode};

const API: &str = "https://api.gleif.org/api/v1/lei-records";

/// GLEIF's ceiling. 250 is a 400, so this is the fewest round trips available.
const PAGE_SIZE: usize = 200;

/// The filter fields this tree offers.
///
/// A subset of what the API accepts, confirmed by asking it. Kept as a list rather than
/// a guess because it is also the directory listing for `/search`: a field that is not
/// here is a path that does not exist rather than a 400 waiting to happen.
///
/// What is left out is vendor cross-references and the registry's own bookkeeping —
/// real fields, but ones that answer questions about GLEIF rather than about a company.
const FILTERS: &[&str] = &[
    "entity.category",
    "entity.jurisdiction",
    "entity.legalAddress.country",
    "entity.legalName",
    "entity.otherNames",
    "entity.registeredAs",
    "entity.status",
    "fulltext",
    "isin",
    "lei",
    "owns",
    "ownedBy",
];

/// The tree's entry point, served by the tree itself.
///
/// A store that answers over the network has no place to leave a README on disk, and
/// a caller landing at the root has no other way to learn that a listing here is not
/// a promise of completeness. So the catalogue is a node like any other.
const CATALOG: &str = r#"# GLEIF, as a filesystem

Every file here is one request to the Global LEI Index, answered live. Nothing is
stored; a listing is what the API would say now.

    /search/<field>/<value>/…/    narrow a query by descending
    /search/…/pages/              its results
    /by-lei/<LEI>/                one entity, addressed by its LEI

## Which way in

An entity is addressed, not found by listing: `ls /by-lei` names nothing but its own
note. An LEI comes from a search page here, or from `owns`/`ownedBy` on another record.

Identifiers are worth confirming rather than recalling: a wrong LEI answers with
somebody else's entity rather than with an error.

## Two rules that are not obvious

**AND is a directory, OR is a comma.** The API takes repeated filters as AND and reads
a comma inside one value as OR, so the comma is spoken for and nesting carries the
rest:

    /search/entity.legalAddress.country/KR,JP/entity.status/ACTIVE/pages/

**Descending is free; opening `pages` is not.** The fields are known here, so walking
down a query asks nothing. The first request happens when a `pages` directory is read,
because only the API knows how many results there are. Narrow first, then open.

## What you can filter on

`ls /search` names them. A field that is not listed is a path that does not exist, so
naming one costs no request. Fifteen plausible-looking ones (`entity.legalAddress.city`,
anything under `headquartersAddress`, any date) are absent because the API rejects them;
the rest of what it accepts is vendor cross-references and registry bookkeeping, left
out because no question about a company needs them.

Values are not listable. They come from data you have already read: a country code, a
status, an LEI from a search result or another registry.

## When an answer is an error

A rejected request is served as the file's contents, JSON and all, rather than as a
missing file. Read it — it says which parameter was wrong.
"#;

/// What a `pages` directory says about itself.
///
/// Only page 1 is listed, which needs saying: the missing names are not missing
/// results. A directory read is followed by a getattr for each entry, a page's size
/// cannot be known without fetching that page, and so naming every page would spend
/// one request per page before a caller had read anything.
const PAGES_README: &str = r#"# pages — page 1 is listed, the rest are named

    page-001.json   the first page of results
    page-002.json   the second, and so on — 1-based, three digits

Only `page-001.json` appears in the listing. The others exist and are read the same
way; they are left out because a directory read makes the kernel ask for every entry's
size, and a page's size is not known until that page has been fetched. Listing all of
them would spend one request per page before anything had been read.

## How many pages there are

The line above this note says so. `page-001.json` carries the same in
`meta.pagination` — `lastPage`, `total`, `perPage` (200 here, the API's ceiling) — and
`links.next` as a URL, but reading a page to count them costs two hundred records in
view. Either way, `page-002.json` is the path.

Paging is the expensive way to read a large result. The directory above this one still
lists fields to narrow by, and descending into them costs nothing.
"#;

/// What `/by-lei` says about itself.
///
/// A listing that comes back empty reads as "nothing here", which is the one thing a
/// caller must not conclude: the entries are all there, and none of them is
/// discoverable. So the directory holds one file that says so, and — the part that
/// makes it actionable — says where an LEI comes from.
const BY_LEI_README: &str = r#"# /by-lei — addressed, not listed

This directory has no listing. There are millions of LEIs and which ones matter
depends on the question, so an entity here is opened by **naming an LEI you already
have**.

    /by-lei/<LEI>/                what this entity has (use ls)
    /by-lei/<LEI>/record/         its GLEIF record

The 20 characters have to be right. A failed checksum means **the directory does not
exist** — not an empty answer, a missing path.

## Where an LEI comes from

- `/search/<field>/<value>/…/pages/page-001.json`
  Every item there carries its LEI in `id`, and the item is **byte for byte what
  `record.json` returns**. If the record is all you need, the search page is the end
  of the trip.
- A relationship resource answers with the other party's LEI. That is the path for
  walking ownership: LEI to LEI.
- Another registry's record, when it carries an LEI field, gives you one directly.

## Relationships read from two sides

    /by-lei/<LEI>/            what this record links to itself (issuer, parent, …)
    /search/ownedBy/<LEI>/    the entities this one owns
    /search/owns/<LEI>/       the entity that owns this one

`ownedBy` returns children and `owns` returns the parent. The names invite the
opposite reading, so check the direction before drawing a tree.
"#;

pub struct GleifFs {
    fetch: Fetcher,
}

/// What a path in this tree denotes.
#[derive(Debug, PartialEq, Eq)]
enum Node {
    Root,
    /// `/by-lei` — holds nothing but the note that explains itself.
    ByLeiRoot,
    /// `/by-lei/_README.md`
    ByLeiReadme,
    /// `…/pages/_README.md`
    PagesReadme(Vec<(String, String)>),
    /// `/CATALOG.md`
    Catalog,
    /// `/by-lei/<LEI>`
    Entity(String),
    /// `/by-lei/<LEI>/<resource>` — the wrapper, free to stat.
    ResourceDir(String, String),
    /// `/by-lei/<LEI>/<resource>/<resource>.json`, or the flat path beside it.
    EntityFile(String, String),
    /// `/search`
    SearchRoot,
    /// A query so far, plus a field named but not yet given a value — which is what
    /// every narrowing step passes through, since a query directory advertises the
    /// fields still available as directories of their own.
    Field(Vec<(String, String)>, String),
    /// `/search/<f>/<v>/…` — one or more complete pairs.
    Query(Vec<(String, String)>),
    /// `…/pages` — the results of the query above it.
    Pages(Vec<(String, String)>),
    /// `…/pages/page-NNN.json`
    Page(Vec<(String, String)>, usize),
}

impl GleifFs {
    pub fn new() -> Self {
        Self {
            fetch: Fetcher::new(concat!(
                "ailoy-company-analysis/",
                env!("CARGO_PKG_VERSION")
            )),
        }
    }

    /// How many requests this store has sent. A cache hit is not one.
    pub fn calls(&self) -> usize {
        self.fetch.calls()
    }

    pub fn breakdown(&self) -> Vec<(&'static str, usize)> {
        self.fetch.breakdown()
    }

    pub fn hot_keys(&self, n: usize) -> Vec<(String, usize)> {
        self.fetch.hot_keys(n)
    }

    pub fn distinct_keys(&self) -> usize {
        self.fetch.distinct_keys()
    }

    /// Parse a path into what it denotes, or `None` when it denotes nothing.
    ///
    /// Rejecting here rather than at the API is the difference between a missing
    /// directory and a failed request: an unknown filter field never becomes a call.
    fn parse(path: &Path) -> Option<Node> {
        let segs: Vec<&str> = path
            .components()
            .filter_map(|c| match c {
                Component::Normal(s) => s.to_str(),
                _ => None,
            })
            .collect();

        match segs.as_slice() {
            [] => Some(Node::Root),
            ["CATALOG.md"] => Some(Node::Catalog),
            ["by-lei"] => Some(Node::ByLeiRoot),
            ["by-lei", "_README.md"] => Some(Node::ByLeiReadme),
            ["by-lei", lei] if is_lei(lei) => Some(Node::Entity(lei.to_string())),
            // A resource is a directory holding one file of the same name, and the
            // flat form beside it also opens — it is simply never listed.
            ["by-lei", lei, res, file]
                if is_lei(lei) && file.trim_end_matches(".json") == *res =>
            {
                Some(Node::EntityFile(lei.to_string(), res.to_string()))
            }
            ["by-lei", lei, file] if is_lei(lei) && file.ends_with(".json") => Some(
                Node::EntityFile(lei.to_string(), file.trim_end_matches(".json").to_string()),
            ),
            ["by-lei", lei, res] if is_lei(lei) => {
                Some(Node::ResourceDir(lei.to_string(), res.to_string()))
            }
            ["search"] => Some(Node::SearchRoot),
            ["search", rest @ ..] => Self::parse_query(rest),
            _ => None,
        }
    }

    /// `(f/v)+`, optionally followed by `pages` or `pages/page-NNN.json`, or a lone
    /// field awaiting a value.
    fn parse_query(rest: &[&str]) -> Option<Node> {
        // Peel the results suffix off the end; what remains is the query itself.
        let (pairs, tail, readme) = match rest {
            [init @ .., "pages", page] if page.starts_with("page-") && page.ends_with(".json") => {
                let n: usize = page
                    .trim_start_matches("page-")
                    .trim_end_matches(".json")
                    .parse()
                    .ok()?;
                if n == 0 {
                    return None;
                }
                (init, Some(Some(n)), false)
            }
            // The note belongs to this query, not to `pages` in general: it reports the
            // query's own size, which the listing has already paid for.
            [init @ .., "pages", "_README.md"] => (init, Some(None), true),
            [init @ .., "pages"] => (init, Some(None), false),
            _ => (rest, None, false),
        };

        if pairs.is_empty() {
            return None;
        }
        // An odd tail is a field awaiting a value. It can appear at any depth, because
        // a query directory lists the fields still available — refusing it here would
        // advertise a directory nothing can enter.
        let (pairs, dangling) = if pairs.len() % 2 == 1 {
            // Results hang off a complete query, never off a field with no value.
            if tail.is_some() {
                return None;
            }
            (&pairs[..pairs.len() - 1], Some(pairs[pairs.len() - 1]))
        } else {
            (pairs, None)
        };

        let mut out: Vec<(String, String)> = Vec::with_capacity(pairs.len() / 2 + 1);
        for pair in pairs.chunks(2) {
            let (f, v) = (pair[0], pair[1]);
            if !FILTERS.contains(&f) {
                return None;
            }
            // The same field twice would silently drop one condition upstream — the API
            // takes the last `filter[..]` and ignores the earlier. A caller nesting them
            // meant AND, so refuse rather than answer a different question.
            if out.iter().any(|(prev, _): &(String, String)| prev == f) {
                return None;
            }
            out.push((f.to_string(), percent_decode(v)));
        }

        if let Some(field) = dangling {
            // The same constraints as a valued pair: a real field, and not one this
            // query already fixed.
            if !FILTERS.contains(&field) || out.iter().any(|(prev, _)| prev == field) {
                return None;
            }
            return Some(Node::Field(out, field.to_string()));
        }

        Some(match tail {
            Some(Some(n)) => Node::Page(out, n),
            Some(None) if readme => Node::PagesReadme(out),
            Some(None) => Node::Pages(out),
            None => Node::Query(out),
        })
    }

    /// The note for one query's `pages`, with that query's own size in it.
    ///
    /// The count comes from page 1, which the listing has already fetched, so reading it
    /// here is a cache hit. Without it the note can only say where the count is, and the
    /// caller pulls two hundred records into view to read one number.
    async fn pages_note(&self, pairs: &[(String, String)]) -> io::Result<Arc<Vec<u8>>> {
        let first = self
            .body(&Node::Page(pairs.to_vec(), 1), &page_key(pairs, 1))
            .await?;
        let doc: serde_json::Value = serde_json::from_slice(&first).map_err(io_err)?;
        let at = |k: &str| {
            doc.pointer(&format!("/meta/pagination/{k}"))
                .and_then(|v| v.as_u64())
        };
        let head = match (at("lastPage"), at("total")) {
            (Some(last), Some(total)) => {
                format!("This query has {last} page(s), {total} record(s).\n\n")
            }
            _ => String::new(),
        };
        Ok(Arc::new(format!("{head}{PAGES_README}").into_bytes()))
    }

    /// The URL a query's page comes from. `page` is 1-based, as the API counts.
    fn query_url(pairs: &[(String, String)], page: usize, size: usize) -> String {
        let mut url = format!("{API}?page%5Bnumber%5D={page}&page%5Bsize%5D={size}");
        for (f, v) in pairs {
            url.push_str(&format!(
                "&filter%5B{}%5D={}",
                urlencode(f),
                urlencode(v)
            ));
        }
        url
    }

    /// The record itself. Split out so [`resources`](Self::resources) can read it
    /// without going back through [`body`](Self::body), which would make the two
    /// mutually recursive for no gain.
    async fn record_bytes(&self, lei: &str) -> io::Result<Arc<Vec<u8>>> {
        self.fetch
            .get(
                "lei record",
                &format!("/by-lei/{lei}/record.json"),
                &format!("{API}/{lei}"),
            )
            .await
    }

    /// The bytes a file path serves.
    async fn body(&self, node: &Node, key: &str) -> io::Result<Arc<Vec<u8>>> {
        match node {
            Node::EntityFile(lei, file) if file == "record" => self.record_bytes(lei).await,
            Node::EntityFile(lei, file) => {
                let url = self
                    .resources(lei)
                    .await?
                    .get(file.as_str())
                    .cloned()
                    .ok_or_else(|| io::Error::from(io::ErrorKind::NotFound))?;
                self.fetch.get("linked resource", key, &url).await
            }
            Node::Page(pairs, n) => {
                self.fetch
                    .get("search page", key, &Self::query_url(pairs, *n, PAGE_SIZE))
                    .await
            }
            _ => Err(io::ErrorKind::IsADirectory.into()),
        }
    }

    /// The sub-resources one record links to: file name → URL.
    ///
    /// Taken from the record's own `relationships`, so the listing is the API's
    /// hypermedia rather than a table kept in step with it by hand. The file name is
    /// the URL's last segment, which is why a record with no parent shows
    /// `direct-parent-reporting-exception.json` and one with a parent shows
    /// `direct-parent.json` — the difference is a fact about the entity.
    async fn resources(&self, lei: &str) -> io::Result<HashMap<String, String>> {
        let bytes = self.record_bytes(lei).await?;
        let doc: serde_json::Value = serde_json::from_slice(&bytes).map_err(io_err)?;
        let rels = doc
            .pointer("/data/relationships")
            .and_then(|v| v.as_object())
            .ok_or_else(|| io::Error::from(io::ErrorKind::NotFound))?;

        let mut out = HashMap::new();
        for (_, entry) in rels {
            let Some(links) = entry.get("links").and_then(|v| v.as_object()) else {
                continue;
            };
            for (_, url) in links {
                if let Some(url) = url.as_str() {
                    if let Some(name) = url.rsplit('/').next() {
                        out.insert(name.to_string(), url.to_string());
                    }
                }
            }
        }
        Ok(out)
    }
}

impl Default for GleifFs {
    fn default() -> Self {
        Self::new()
    }
}

impl FileSystem for GleifFs {
    fn stat<'a>(&'a self, path: &'a Path) -> BoxFuture<'a, io::Result<Stat>> {
        Box::pin(async move {
            let node = Self::parse(path).ok_or(io::ErrorKind::NotFound)?;
            match &node {
                // Written here, so their sizes cost nothing to report.
                Node::ByLeiReadme => Ok(Stat::new(DirentKind::File, BY_LEI_README.len() as u64)),
                Node::PagesReadme(pairs) => {
                    let note = self.pages_note(pairs).await?;
                    Ok(Stat::new(DirentKind::File, note.len() as u64))
                }
                Node::Catalog => Ok(Stat::new(DirentKind::File, CATALOG.len() as u64)),
                Node::EntityFile(..) | Node::Page(..) => {
                    let bytes = self.body(&node, &node_key(&node)).await?;
                    Ok(Stat::new(DirentKind::File, bytes.len() as u64))
                }
                // Directories cost nothing to confirm: the path grammar already
                // decided, and a round trip here would make every `ls` pay for a
                // question it did not ask.
                _ => Ok(Stat::new(DirentKind::Dir, 0)),
            }
        })
    }

    fn list<'a>(&'a self, path: &'a Path) -> BoxFuture<'a, io::Result<Vec<Dirent>>> {
        Box::pin(async move {
            let node = Self::parse(path).ok_or(io::ErrorKind::NotFound)?;
            let dir = |n: &str| Dirent::new(n, DirentKind::Dir);
            Ok(match node {
                Node::Root => vec![
                    Dirent::with_stat(
                        "CATALOG.md",
                        Stat::new(DirentKind::File, CATALOG.len() as u64),
                    ),
                    dir("by-lei"),
                    dir("search"),
                ],
                Node::ByLeiRoot => vec![Dirent::with_stat(
                    "_README.md",
                    Stat::new(DirentKind::File, BY_LEI_README.len() as u64),
                )],
                // Directories, so this listing costs the one fetch it already needs —
                // which resources exist is a fact only the record carries — and not one
                // per resource on top of it. A file here would be fetched to be listed,
                // because a directory read asks for every entry's size.
                //
                // The names are the record's own links, so what appears is what exists
                // for this entity: a record with no parent shows a reporting exception
                // where one with a parent shows a parent.
                Node::Entity(lei) => {
                    let mut names: Vec<String> = self.resources(&lei).await?.into_keys().collect();
                    names.push("record".to_string());
                    names.sort();
                    names.into_iter().map(|n| dir(&n)).collect()
                }
                // Where the cost lands, once a caller has asked for the resource.
                Node::ResourceDir(lei, res) => {
                    let node = Node::EntityFile(lei.clone(), res.clone());
                    let b = self.body(&node, &node_key(&node)).await?;
                    vec![Dirent::with_stat(
                        format!("{res}.json"),
                        Stat::new(DirentKind::File, b.len() as u64),
                    )]
                }
                Node::ByLeiReadme
                | Node::PagesReadme(..)
                | Node::Catalog
                | Node::EntityFile(..)
                | Node::Page(..) => {
                    return Err(io::ErrorKind::NotADirectory.into());
                }
                Node::SearchRoot => FILTERS.iter().map(|f| dir(*f)).collect(),
                // Values are unbounded, so naming one is the only way in.
                Node::Field(..) => vec![],
                // Free. Narrowing is what a caller does most, and none of it is a
                // question for the API — the fields are known and the results are one
                // level down, behind `pages`. Listing the pages here instead would
                // charge a request for every step of a walk that wanted none.
                Node::Query(pairs) => {
                    let mut out = vec![dir("pages")];
                    out.extend(
                        FILTERS
                            .iter()
                            .filter(|f| !pairs.iter().any(|(used, _)| used == *f))
                            .map(|f| dir(*f)),
                    );
                    out
                }
                // Only page 1, and only because this is the one listing that has to
                // ask the API anyway. Naming all thirteen pages of a result would be
                // worse than useless: a directory read is followed by a getattr per
                // entry, a page's size is not knowable without fetching it, and so a
                // listing of N pages costs N requests before anything has been read.
                // Measured — an early version answered 13 for a caller that wanted 1.
                Node::Pages(pairs) => {
                    let key = page_key(&pairs, 1);
                    let first = self.body(&Node::Page(pairs.clone(), 1), &key).await?;
                    vec![
                        Dirent::with_stat(
                            "page-001.json",
                            Stat::new(DirentKind::File, first.len() as u64),
                        ),
                        Dirent::with_stat(
                            "_README.md",
                            Stat::new(DirentKind::File, self.pages_note(&pairs).await?.len() as u64),
                        ),
                    ]
                }
            })
        })
    }

    fn read_at<'a>(
        &'a self,
        path: &'a Path,
        buf: &'a mut [u8],
        offset: u64,
    ) -> BoxFuture<'a, io::Result<usize>> {
        Box::pin(async move {
            let node = Self::parse(path).ok_or(io::ErrorKind::NotFound)?;
            let bytes: Arc<Vec<u8>> = match &node {
                Node::ByLeiReadme => Arc::new(BY_LEI_README.as_bytes().to_vec()),
                Node::PagesReadme(pairs) => self.pages_note(pairs).await?,
                Node::Catalog => Arc::new(CATALOG.as_bytes().to_vec()),
                Node::EntityFile(..) | Node::Page(..) => {
                    self.body(&node, &node_key(&node)).await?
                }
                _ => return Err(io::ErrorKind::IsADirectory.into()),
            };
            let start = (offset as usize).min(bytes.len());
            let n = buf.len().min(bytes.len() - start);
            buf[..n].copy_from_slice(&bytes[start..start + n]);
            Ok(n)
        })
    }
}

/// Cache key for a node. Query pairs are sorted, so the two orders a caller might
/// descend in share one entry instead of fetching the same page twice.
fn node_key(node: &Node) -> String {
    match node {
        Node::EntityFile(lei, file) => format!("/by-lei/{lei}/{file}.json"),
        Node::Page(pairs, n) => page_key(pairs, *n),
        _ => String::new(),
    }
}

fn page_key(pairs: &[(String, String)], n: usize) -> String {
    let mut sorted = pairs.to_vec();
    sorted.sort();
    let joined: Vec<String> = sorted.iter().map(|(f, v)| format!("{f}={v}")).collect();
    format!("/search?{}#page-{n}", joined.join("&"))
}

/// ISO 17442: 18 alphanumerics and a mod 97-10 check pair.
///
/// Checked here so a typo is a missing directory rather than a request that comes
/// back empty and looks like an answer.
fn is_lei(s: &str) -> bool {
    if s.len() != 20 || !s.bytes().all(|b| b.is_ascii_alphanumeric()) {
        return false;
    }
    let mut rem: u32 = 0;
    for b in s.bytes() {
        let v = if b.is_ascii_digit() {
            (b - b'0') as u32
        } else {
            (b.to_ascii_uppercase() - b'A') as u32 + 10
        };
        // Two digits at a time for a letter, one for a digit — the same widening the
        // spec's "convert to numeric then mod 97" describes, without the big integer.
        rem = if v >= 10 {
            (rem * 100 + v) % 97
        } else {
            (rem * 10 + v) % 97
        };
    }
    rem == 1
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(p: &str) -> Option<Node> {
        GleifFs::parse(Path::new(p))
    }

    /// A real LEI, and the same string with one character moved.
    const SAMSUNG: &str = "9884007ER46L6N7EI764";

    #[test]
    fn lei_checksum_rejects_a_typo() {
        assert!(is_lei(SAMSUNG));
        assert!(is_lei("984500459FE6BED64E36"));
        // Right shape, wrong check digits — the case a length test would let through.
        assert!(!is_lei("9884007ER46L6N7EI765"));
        assert!(!is_lei("TOOSHORT"));
        assert!(!is_lei("9884007ER46L6N7EI76!"));
    }

    #[test]
    fn paths_denote_what_they_look_like() {
        assert_eq!(parse("/"), Some(Node::Root));
        assert_eq!(parse("/by-lei"), Some(Node::ByLeiRoot));
        assert_eq!(parse(&format!("/by-lei/{SAMSUNG}")), Some(Node::Entity(SAMSUNG.into())));
        assert_eq!(
            parse(&format!("/by-lei/{SAMSUNG}/record.json")),
            Some(Node::EntityFile(SAMSUNG.into(), "record".into()))
        );
        assert_eq!(parse("/search"), Some(Node::SearchRoot));
        // A bad LEI is not a path at all, so no request is ever made for it.
        assert_eq!(parse("/by-lei/NOTALEI"), None);
    }

    #[test]
    fn and_nests_while_or_stays_in_the_value() {
        let q = parse("/search/entity.legalAddress.country/KR,JP/entity.status/ACTIVE");
        assert_eq!(
            q,
            Some(Node::Query(vec![
                ("entity.legalAddress.country".into(), "KR,JP".into()),
                ("entity.status".into(), "ACTIVE".into()),
            ]))
        );
        // The comma survives into the request, because upstream reads it as OR.
        let Some(Node::Query(pairs)) = q else { unreachable!() };
        assert!(GleifFs::query_url(&pairs, 1, PAGE_SIZE).contains("=KR,JP"));
    }

    #[test]
    fn unknown_field_is_not_a_path() {
        // One of the 15 that answer 400. Rejected here, so it never becomes a request.
        assert_eq!(parse("/search/entity.legalAddress.city/Seoul"), None);
        assert_eq!(parse("/search/nosuchfield/x"), None);
        assert_eq!(parse("/search/entity.legalAddress.city"), None);
    }

    #[test]
    fn the_same_field_twice_is_refused() {
        // Upstream would keep the last and drop the first, answering a question the
        // caller did not ask. Better to have no such path.
        assert_eq!(
            parse("/search/entity.status/ACTIVE/entity.status/INACTIVE"),
            None
        );
    }

    #[test]
    fn a_field_may_dangle_at_any_depth() {
        assert_eq!(parse("/search/fulltext"), Some(Node::Field(vec![], "fulltext".into())));
        // The narrowing step every query directory advertises. Refusing it would list a
        // directory nothing can enter.
        assert_eq!(
            parse("/search/entity.status/ACTIVE/fulltext"),
            Some(Node::Field(
                vec![("entity.status".into(), "ACTIVE".into())],
                "fulltext".into()
            ))
        );
        // Still not a field the query already fixed, and still not an unknown one.
        assert_eq!(parse("/search/entity.status/ACTIVE/entity.status"), None);
        assert_eq!(parse("/search/entity.status/ACTIVE/nosuchfield"), None);
    }

    /// Whatever a query directory offers must be enterable. The two were allowed to
    /// disagree once, and a listing that advertises a dead end is worse than one that
    /// advertises nothing.
    /// The note under `pages` belongs to its query, so it can report that query's own
    /// size. Parsing has to keep the pairs for that; dropping them is how it silently
    /// becomes the same note everywhere.
    #[test]
    fn the_pages_note_keeps_its_query() {
        let Some(Node::PagesReadme(pairs)) = parse("/search/entity.legalName/NVIDIA/pages/_README.md") else {
            panic!("the note did not parse as a query's own")
        };
        assert!(!pairs.is_empty(), "the note lost the query it belongs to");
        // The same query with and without the note names one query, not two.
        let Some(Node::Pages(same)) = parse("/search/entity.legalName/NVIDIA/pages") else {
            panic!("pages")
        };
        assert_eq!(pairs, same);
    }

    #[tokio::test]
    async fn every_advertised_directory_can_be_entered() {
        let fs = GleifFs::new();
        let base = "/search/entity.legalAddress.country/KR";
        for e in fs.list(Path::new(base)).await.expect("list") {
            let child = format!("{base}/{}", e.name);
            if e.name == "pages" {
                continue; // that one would ask the API
            }
            assert!(
                GleifFs::parse(Path::new(&child)).is_some(),
                "listed but not enterable: {child}"
            );
        }
    }

    #[test]
    fn pages_are_one_based() {
        assert_eq!(
            parse("/search/fulltext/samsung/pages/page-001.json"),
            Some(Node::Page(vec![("fulltext".into(), "samsung".into())], 1))
        );
        assert_eq!(parse("/search/fulltext/samsung/pages/page-000.json"), None);
        assert_eq!(parse("/search/fulltext/pages/page-001.json"), None);
    }

    #[test]
    fn by_lei_explains_itself_instead_of_listing_nothing() {
        assert_eq!(parse("/by-lei/_README.md"), Some(Node::ByLeiReadme));
        // The two things a caller needs next: how to address an entity, and where an
        // LEI comes from. Pinned because an empty listing is the failure this replaces.
        assert!(BY_LEI_README.contains("/by-lei/<LEI>/"));
        assert!(BY_LEI_README.contains("/search/"));
        // `ownedBy` is the child query and reads backwards, so the note says so.
        assert!(BY_LEI_README.contains("ownedBy"));
    }

    #[test]
    fn results_hang_off_pages_not_off_the_query() {
        let q = "/search/entity.legalAddress.country/KR";
        assert!(matches!(parse(q), Some(Node::Query(_))));
        assert!(matches!(parse(&format!("{q}/pages")), Some(Node::Pages(_))));
        // `pages` only follows a complete query, never a field with no value.
        assert_eq!(parse("/search/fulltext/pages"), None);
        // And a page never hangs directly off the query.
        assert_eq!(parse(&format!("{q}/page-001.json")), None);
    }

    /// The design claim, checked rather than asserted in prose: walking down a
    /// narrowing chain asks the API nothing.
    ///
    /// No network and no mount, because none is needed — if any of these listings
    /// sent a request the counter would say so, and the test would fail offline too.
    #[tokio::test]
    async fn narrowing_sends_no_requests() {
        let fs = GleifFs::new();
        for p in [
            "/",
            "/by-lei",
            "/search",
            "/search/entity.legalAddress.country",
            "/search/entity.legalAddress.country/KR",
            "/search/entity.legalAddress.country/KR/entity.status",
            "/search/entity.legalAddress.country/KR/entity.status/ACTIVE",
            "/search/entity.legalAddress.country/KR/entity.status/ACTIVE/entity.category/FUND",
        ] {
            fs.list(Path::new(p))
                .await
                .unwrap_or_else(|e| panic!("list {p}: {e}"));
            fs.stat(Path::new(p)).await.unwrap_or_else(|e| panic!("stat {p}: {e}"));
        }
        assert_eq!(fs.calls(), 0, "narrowing sent a request");

        // The note is written here too, so reading it is also free.
        let mut buf = [0u8; 64];
        fs.read_at(Path::new("/by-lei/_README.md"), &mut buf, 0)
            .await
            .expect("read _README.md");
        assert_eq!(fs.calls(), 0, "reading a file written here sent a request");
    }

    /// The whole chain, for real: a tree served over FUSE, read with `std::fs`.
    ///
    /// Ignored by default — it mounts a filesystem and talks to GLEIF, neither of
    /// which belongs in a plain `cargo test`. Run it with
    /// `cargo test -p company_analysis --bins mounts_and_reads -- --ignored --nocapture`.
    ///
    /// Not a `#[tokio::test]`: the binding serves the store from its own thread on its
    /// own runtime, and the reads below go through the kernel. Blocking a test runtime
    /// on a mount that is waiting for that runtime is how this deadlocks.
    #[test]
    #[ignore = "needs a FUSE mount and network"]
    fn mounts_and_reads() {
        use cortex::fs::{FuseTMount, Mount};
        use std::fs;

        let fs_arc = Arc::new(GleifFs::new());
        let dir = std::env::temp_dir().join("ailoy-gleif-mount-test");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("mountpoint");

        let mount = FuseTMount::try_new(fs_arc.clone(), &dir).expect("mount");
        let root = mount.mountpoint().to_path_buf();

        let names = |p: &std::path::Path| {
            let mut v: Vec<String> = fs::read_dir(p)
                .unwrap_or_else(|e| panic!("read_dir {}: {e}", p.display()))
                .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
                .collect();
            v.sort();
            v
        };

        assert_eq!(names(&root), ["CATALOG.md", "by-lei", "search"]);
        // The tree carries its own entry point, so an agent told to read
        // `<data>/CATALOG.md` finds one whether the store is on disk or on the wire.
        let catalog = fs::read_to_string(root.join("CATALOG.md")).expect("catalog");
        assert!(catalog.contains("AND is a directory, OR is a comma"));
        assert_eq!(names(&root.join("by-lei")), ["_README.md"]);

        let note = fs::read_to_string(root.join("by-lei/_README.md")).expect("note");
        assert!(note.contains("addressed, not listed"));

        // Narrowing, through the kernel this time.
        let q = root.join("search/entity.legalAddress.country/KR/entity.status/ACTIVE");
        assert!(names(&q).contains(&"pages".to_string()));
        assert_eq!(fs_arc.calls(), 0, "narrowing sent a request");

        // The one listing that has to ask — and it asks exactly once, because it names
        // only the page it already holds.
        let pages = names(&q.join("pages"));
        assert_eq!(pages, ["_README.md", "page-001.json"]);
        assert_eq!(fs_arc.calls(), 1, "listing the pages took more than one request");

        // And the page it asked with is the page we read, so this is still one request.
        let page = q.join("pages/page-001.json");
        let size = fs::metadata(&page).expect("metadata").len();
        let body = fs::read(&page).expect("read page");
        assert_eq!(fs_arc.calls(), 1, "a page already held was fetched again");
        assert_eq!(
            body.len() as u64, size,
            "the size stat reported and the bytes read disagree — the cache did not hold"
        );

        let doc: serde_json::Value = serde_json::from_slice(&body).expect("json");
        assert!(doc["data"].as_array().is_some_and(|a| !a.is_empty()));
        // The page itself says how many there are, which is why the listing need not.
        let last = doc["meta"]["pagination"]["lastPage"].as_u64().unwrap_or(0);
        assert!(last > 1, "this check is only meaningful on a query with several pages");
        // A page that was never listed is still readable by name.
        let second = fs::read(q.join("pages/page-002.json")).expect("read page 2");
        assert!(!second.is_empty());
        assert_eq!(fs_arc.calls(), 2, "opening a page by name took more than one request");

        println!(
            "lastPage={last} first={} calls={}",
            doc["data"][0]["id"].as_str().unwrap_or("?"),
            fs_arc.calls()
        );

        drop(mount); // dropping unmounts, which is the contract
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn cache_key_is_order_insensitive() {
        let a = parse("/search/entity.status/ACTIVE/fulltext/x/pages/page-002.json").unwrap();
        let b = parse("/search/fulltext/x/entity.status/ACTIVE/pages/page-002.json").unwrap();
        assert_ne!(a, b, "the two paths differ");
        assert_eq!(node_key(&a), node_key(&b), "but they are the same request");
    }

    #[test]
    fn a_value_can_carry_a_slash_when_encoded() {
        let q = parse("/search/entity.legalName/A%2FB").unwrap();
        assert_eq!(
            q,
            Node::Query(vec![("entity.legalName".into(), "A/B".into())])
        );
        let Node::Query(pairs) = &q else { unreachable!() };
        assert!(GleifFs::query_url(pairs, 1, 1).contains("A%2FB"));
    }
}
