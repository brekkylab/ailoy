//! SEC EDGAR projected onto a read-only [`FileSystem`].
//!
//! ```text
//! /by-cik/<CIK>/submissions.json       one registrant: identifiers, addresses, filings
//! /by-cik/<CIK>/concept/<tax>/<tag>.json   one XBRL concept's history
//! /search/<field>/<value>/…/pages/      full-text search over filings
//! ```
//!
//! # The opposite of GLEIF
//!
//! GLEIF answers a search with the whole record, so a search page is the end of the
//! trip. EDGAR's list is a stub — cik, ticker, title — and everything else needs the
//! registrant fetched by CIK.
//!
//! # One address, one way to find it
//!
//! A registrant is addressed by CIK and nothing else. There is no door keyed by ticker
//! because nothing needs one: `search/entityName/` takes a symbol and answers with the
//! registrant — the SEC indexes a filer's display name with its ticker in it, so
//! `entityName/NVDA` lands on NVIDIA CORP more squarely than `entityName/NVIDIA`, which
//! returns individuals of that name first.
//!
//! A second door would also be a second listing, and a listing here is a place for a
//! `find` or a glob to start walking. Every registrant a walk reaches costs a
//! `submissions.json` and a four-megabyte `facts.json`, so an entry point that can be
//! enumerated is an entry point that can spend thousands of requests nobody asked for.
//!
//! # What a listing costs
//!
//! Measured, because the sizes differ by two orders of magnitude:
//!
//! ```text
//! submissions.json   161 KB
//! concept/<tag>       42 KB
//! facts.json           4 MB      ← every concept at once
//! ```
//!
//! A directory read is followed by a getattr per entry, and none of these can be
//! measured without being fetched. So `facts.json` is readable but not listed: naming
//! it would spend four megabytes on an `ls` that wanted the filing history.

use std::{
    io,
    path::{Component, Path},
    sync::Arc,
};

use cortex::{
    BoxFuture,
    fs::{Dirent, DirentKind, FileSystem, Stat},
};

use crate::apifs::{Fetcher, io_err, percent_decode, urlencode};

const DATA: &str = "https://data.sec.gov";
const EFTS: &str = "https://efts.sec.gov/LATEST/search-index";

/// Full-text search returns 100 hits a page and pages by offset, not by number.
const HITS_PER_PAGE: usize = 100;

/// The XBRL taxonomies a concept can come from. Fixed, so listing them is free.
const TAXONOMIES: &[&str] = &["us-gaap", "dei", "ifrs-full", "srt", "invest"];

/// The full-text search parameters that were confirmed to narrow a result set.
///
/// `locationCode` is absent on purpose: it is accepted and changes nothing, which is
/// worse than a rejection because the count looks like an answer.
const FILTERS: &[&str] = &["q", "forms", "ciks", "entityName", "startdt", "enddt"];

/// What a query's `pages` says about itself.
const PAGES_README: &str = r#"# pages — page 1 is listed, the rest are named

    page-001.json   the first page of hits
    page-002.json   the second, and so on — 1-based, three digits

Only `page-001.json` appears in the listing. The others exist and are read the same way;
they are left out because a directory read makes the kernel ask for every entry's size,
and a page's size is not known until that page has been fetched. Listing them all would
spend one request per page before anything had been read. A page past the last one is
`No such file`, not an empty result.

## A hit is a filing, not a company

Each hit names the document and the filers behind it. `ciks` is the list to follow —
zero-padded, so it drops straight into `/by-cik/<CIK>/`. `display_names` reads
`NVIDIA CORP  (NVDA)  (CIK 0001045810)`, which is for a person; the spacing is not a
contract.

A search for a company therefore returns filings that *mention* it as well as its own —
insiders, index funds, counterparties. Only rows carrying its CIK are the registrant.

Paging is the expensive way to read a large result. The directory above this one still
lists parameters to narrow by, and descending into them costs nothing.
"#;

/// What `/by-cik` says about itself.
const BY_CIK_README: &str = r#"# /by-cik — addressed, not listed

There is no listing here. If you know the number, name it.

Not because the set is unknowable — the SEC publishes every CIK it has — but because a
listing of them would be a million zero-padded numbers. Knowing one already lets you
address it; not knowing one, the numbers do not say which company is which.

To get one, search. `/search/entityName/<name>/pages/` searches filings by company name
and reports the CIK behind each hit. A ticker works there as well as a name, and often
better: the SEC writes the symbol into the filer's display name, so `entityName/NVDA`
lands on NVIDIA CORP while `entityName/NVIDIA` returns individuals of that name first.

    /by-cik/<CIK>/submissions.json          identifiers, addresses, filing history
    /by-cik/<CIK>/concept/<taxonomy>/<tag>.json   one concept over time
    /by-cik/<CIK>/facts.json                every concept at once — see below

## The CIK is not padded everywhere

The SEC's own URLs use `CIK0001045810`, while its ticker file writes `1045810`. Either
form works as a directory name here, and both mean the same registrant.

## What a registrant's directory holds

    submissions/    identity, addresses, filing history      (~160 KB)
    facts/          every XBRL concept it reports            (~4 MB)
    concept/<taxonomy>/<tag>.json   one concept over time    (~40 KB)

Each of the first two is a directory holding one file of the same name. That is not
decoration: a directory read asks for every entry's size, a document's size is not
knowable until it has been fetched, and a document sitting here directly would
therefore be fetched merely to be listed. Wrapping it means `ls` shows what exists
while the request waits until someone descends.

So `ls submissions` costs one request and `submissions/submissions.json` is then free.
The flat `submissions.json` beside it also works, for anyone who guesses it.

Start with `submissions`. `filings.recent` inside it is column-oriented: parallel
arrays, one per field, rather than a list of filings — zip them by index before reading
a row. Use `facts` when the question is which concepts exist at all, and `concept/`
when you already know the tag.

## What does not connect to the rest

`submissions.json` has an `lei` field and it is almost always null — fifteen
registrants sampled across the ticker list had it empty, every one. There is no
reliable key from here to the GLEIF store; crossing over means matching on name, and a
name match is a candidate rather than a fact.
"#;

/// The tree's entry point, served by the tree itself.
const CATALOG: &str = r#"# SEC EDGAR, as a filesystem

Every file here is one request to the SEC, answered live. Nothing is stored.

    /by-cik/<CIK>/                any filer, by its SEC number
    /search/<field>/<value>/…/    full-text search over filing documents

## Which way in

A registrant is addressed, not found by listing: `ls /by-cik` names nothing but its own
note. If you do not have a CIK, `/search/entityName/` takes a company name or a ticker
and reports the CIK behind each hit.

Identifiers are worth confirming rather than recalling: a wrong CIK answers with
somebody else's filings rather than with an error.

## Search narrows the same way as elsewhere

AND is a directory, and the value is its own segment:

    /search/q/lithium/forms/10-K/startdt/2025-01-01/enddt/2025-06-30/pages/

`ls /search` lists the parameters that were confirmed to narrow a result set. A
parameter that is not listed is not one this store rejects — it is a path that does not
exist. Descending is free; the first request happens when `pages` is opened.

Results are capped at 10,000 by the search backend, so a `total` of exactly 10000 means
"at least", not "exactly". Narrow further before believing it.

## Watch for

`filings.recent` in `submissions.json` is column-oriented: parallel arrays, one per
field, not a list of filings. Zip them by index before reading a row.

A search hit carries the CIK twice, and only one of them is meant to be parsed.
`ciks` is a list of zero-padded strings — `["0001045810"]` — and that is the one to use.
`display_names` reads `NVIDIA CORP  (NVDA)  (CIK 0001045810)`, which is for a person to
read; the spacing is not a contract.
"#;

pub struct EdgarFs {
    fetch: Fetcher,
}

#[derive(Debug, PartialEq, Eq)]
enum Node {
    Root,
    Catalog,
    /// `/by-cik`
    ByCikRoot,
    ByCikReadme,
    /// `/search/…/pages/_README.md` — generated per query, so it can say how large it is.
    PagesReadme(Vec<(String, String)>),
    /// `/by-cik/<CIK>` — the CIK padded to the ten digits the SEC's URLs use.
    Entity(String),
    /// `/by-cik/<CIK>/submissions` — the wrapper, free to stat.
    SubmissionsDir(String),
    /// `/by-cik/<CIK>/submissions/submissions.json`, or the flat path beside it.
    Submissions(String),
    /// `/by-cik/<CIK>/facts`
    FactsDir(String),
    /// `/by-cik/<CIK>/facts/facts.json`, or the flat path.
    Facts(String),
    /// `/by-cik/<CIK>/concept`
    ConceptRoot(String),
    /// `/by-cik/<CIK>/concept/<taxonomy>`
    Taxonomy(String, String),
    /// `/by-cik/<CIK>/concept/<taxonomy>/<tag>.json`
    Concept(String, String, String),
    /// `/search`
    SearchRoot,
    /// A query so far, plus a parameter awaiting a value.
    Field(Vec<(String, String)>, String),
    Query(Vec<(String, String)>),
    Pages(Vec<(String, String)>),
    Page(Vec<(String, String)>, usize),
}

impl EdgarFs {
    /// SEC answers **403 with an HTML page** unless the `User-Agent` identifies the
    /// caller with a contact address — a product name alone is refused. Measured, and
    /// the reason this is a parameter rather than a constant: the right value is
    /// whoever is running this.
    pub fn new(user_agent: &str) -> Self {
        Self {
            fetch: Fetcher::new(user_agent),
        }
    }

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
            ["by-cik"] => Some(Node::ByCikRoot),
            ["by-cik", "_README.md"] => Some(Node::ByCikReadme),
            ["by-cik", cik, rest @ ..] => {
                let cik = normalize_cik(cik)?;
                match rest {
                    [] => Some(Node::Entity(cik)),
                    ["submissions"] => Some(Node::SubmissionsDir(cik)),
                    ["facts"] => Some(Node::FactsDir(cik)),
                    // Inside the wrapper, and the flat form beside it. The flat one is
                    // never listed and costs nothing to keep: someone who guesses the
                    // obvious name should not come away empty.
                    ["submissions", "submissions.json"] | ["submissions.json"] => {
                        Some(Node::Submissions(cik))
                    }
                    ["facts", "facts.json"] | ["facts.json"] => Some(Node::Facts(cik)),
                    ["concept"] => Some(Node::ConceptRoot(cik)),
                    ["concept", tax] if TAXONOMIES.contains(tax) => {
                        Some(Node::Taxonomy(cik, tax.to_string()))
                    }
                    ["concept", tax, tag] if TAXONOMIES.contains(tax) && tag.ends_with(".json") => {
                        Some(Node::Concept(
                            cik,
                            tax.to_string(),
                            tag.trim_end_matches(".json").to_string(),
                        ))
                    }
                    _ => None,
                }
            }
            ["search"] => Some(Node::SearchRoot),
            ["search", rest @ ..] => Self::parse_query(rest),
            _ => None,
        }
    }

    /// Same grammar as the GLEIF store's, for the same reason: AND by descending, and
    /// a field and its value as separate segments so `/search` can list the fields.
    fn parse_query(rest: &[&str]) -> Option<Node> {
        let (pairs, tail, readme) = match rest {
            // The note belongs to this query, not to `pages` in general: it reports the
            // query's own size, which the listing has already paid for.
            [init @ .., "pages", "_README.md"] => (init, Some(None), true),
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
            [init @ .., "pages"] => (init, Some(None), false),
            _ => (rest, None, false),
        };

        if pairs.is_empty() {
            return None;
        }
        let (pairs, dangling) = if pairs.len() % 2 == 1 {
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
            if !FILTERS.contains(&f) || out.iter().any(|(prev, _)| prev == f) {
                return None;
            }
            out.push((f.to_string(), percent_decode(v)));
        }
        if let Some(field) = dangling {
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

    /// `page` is 1-based here; the API offsets by hits, so the two differ by a factor.
    fn query_url(pairs: &[(String, String)], page: usize) -> String {
        let mut url = format!("{EFTS}?from={}", (page - 1) * HITS_PER_PAGE);
        let mut dated = false;
        for (f, v) in pairs {
            if f == "startdt" || f == "enddt" {
                dated = true;
            }
            url.push_str(&format!("&{}={}", urlencode(f), urlencode(v)));
        }
        // The date bounds are ignored unless the range is declared custom, which is
        // the API's own coupling rather than a choice — a caller naming `startdt` in a
        // path means the dates to apply.
        if dated {
            url.push_str("&dateRange=custom");
        }
        url
    }

    /// What a request bought. Taken from the node rather than parsed back out of the
    /// key, which is only a string that looks like a path.
    fn kind(node: &Node) -> &'static str {
        match node {
            Node::Submissions(_) => "submissions",
            Node::Facts(_) => "facts",
            Node::Concept(..) => "concept",
            Node::Page(..) => "search page",
            _ => "other",
        }
    }

    fn key(node: &Node) -> String {
        match node {
            Node::Submissions(cik) => format!("/by-cik/{cik}/submissions.json"),
            Node::Facts(cik) => format!("/by-cik/{cik}/facts.json"),
            Node::Concept(cik, tax, tag) => format!("/by-cik/{cik}/concept/{tax}/{tag}.json"),
            Node::Page(pairs, n) => {
                let mut sorted = pairs.clone();
                sorted.sort();
                let joined: Vec<String> =
                    sorted.iter().map(|(f, v)| format!("{f}={v}")).collect();
                format!("/search?{}#page-{n}", joined.join("&"))
            }
            _ => String::new(),
        }
    }

    fn url(node: &Node) -> Option<String> {
        Some(match node {
            Node::Submissions(cik) => format!("{DATA}/submissions/CIK{cik}.json"),
            Node::Facts(cik) => format!("{DATA}/api/xbrl/companyfacts/CIK{cik}.json"),
            Node::Concept(cik, tax, tag) => {
                format!("{DATA}/api/xbrl/companyconcept/CIK{cik}/{tax}/{tag}.json")
            }
            Node::Page(pairs, n) => Self::query_url(pairs, *n),
            _ => return None,
        })
    }

    async fn body(&self, node: &Node) -> io::Result<Arc<Vec<u8>>> {
        match node {
            Node::Catalog => Ok(Arc::new(CATALOG.as_bytes().to_vec())),
            Node::ByCikReadme => Ok(Arc::new(BY_CIK_README.as_bytes().to_vec())),
            _ => {
                let url = Self::url(node).ok_or(io::ErrorKind::IsADirectory)?;
                self.fetch.get(Self::kind(node), &Self::key(node), &url).await
            }
        }
    }

    /// The note for one query's `pages`, with that query's own size in it.
    ///
    /// The count comes from page 1, which the listing has already fetched, so reading it
    /// here is a cache hit. Without it the note can only say where the count is, and the
    /// caller pulls a hundred hits into view to read one number.
    async fn pages_note(&self, pairs: &[(String, String)]) -> io::Result<Arc<Vec<u8>>> {
        let total = self.total_hits(pairs).await?;
        let last = total.div_ceil(HITS_PER_PAGE).max(1);
        Ok(Arc::new(
            format!("This query has {last} page(s), {total} hit(s).\n\n{PAGES_README}").into_bytes(),
        ))
    }

    /// Whether a query has a page `n`, from the hit count the first page reports.
    ///
    /// Only page 1 is ever listed, so this answers about a page named directly.
    async fn total_hits(&self, pairs: &[(String, String)]) -> io::Result<usize> {
        let bytes = self.body(&Node::Page(pairs.to_vec(), 1)).await?;
        let doc: serde_json::Value = serde_json::from_slice(&bytes).map_err(io_err)?;
        Ok(doc
            .pointer("/hits/total/value")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize)
    }
}

impl FileSystem for EdgarFs {
    fn stat<'a>(&'a self, path: &'a Path) -> BoxFuture<'a, io::Result<Stat>> {
        Box::pin(async move {
            let node = Self::parse(path).ok_or(io::ErrorKind::NotFound)?;
            Ok(match &node {
                Node::Catalog => Stat::new(DirentKind::File, CATALOG.len() as u64),
                Node::ByCikReadme => Stat::new(DirentKind::File, BY_CIK_README.len() as u64),
                Node::PagesReadme(pairs) => {
                    Stat::new(DirentKind::File, self.pages_note(pairs).await?.len() as u64)
                }
                Node::Submissions(_)
                | Node::Facts(_)
                | Node::Concept(..)
                | Node::Page(..) => {
                    let bytes = self.body(&node).await?;
                    Stat::new(DirentKind::File, bytes.len() as u64)
                }
                _ => Stat::new(DirentKind::Dir, 0),
            })
        })
    }

    fn list<'a>(&'a self, path: &'a Path) -> BoxFuture<'a, io::Result<Vec<Dirent>>> {
        Box::pin(async move {
            let node = Self::parse(path).ok_or(io::ErrorKind::NotFound)?;
            let dir = |n: &str| Dirent::new(n, DirentKind::Dir);
            let note = |n: &str, s: &str| {
                Dirent::with_stat(n, Stat::new(DirentKind::File, s.len() as u64))
            };
            Ok(match node {
                Node::Root => vec![
                    note("CATALOG.md", CATALOG),
                    dir("by-cik"),
                    dir("search"),
                ],
                // A directory here is free to
                // stat, so the kernel's getattr storm over eight thousand names costs
                // nothing beyond the one fetch that built the index.
                Node::ByCikRoot => vec![note("_README.md", BY_CIK_README)],
                // Every fetchable resource is wrapped in a directory, so this listing
                // names what is available without asking for any of it. A directory
                // read is followed by a getattr per entry; a directory's size is zero
                // and a document's is not knowable without fetching it, so a file here
                // would be fetched just to be listed. It was, once: naming
                // `submissions.json` cost 742 requests in a single run, because an
                // ordinary recursive pass over a listing turned into one fetch
                // per registrant.
                Node::Entity(_) => vec![dir("submissions"), dir("facts"), dir("concept")],
                // The cost lands here instead, where a caller has asked for the thing.
                Node::SubmissionsDir(cik) => {
                    let b = self.body(&Node::Submissions(cik.clone())).await?;
                    vec![Dirent::with_stat(
                        "submissions.json",
                        Stat::new(DirentKind::File, b.len() as u64),
                    )]
                }
                Node::FactsDir(cik) => {
                    let b = self.body(&Node::Facts(cik.clone())).await?;
                    vec![Dirent::with_stat(
                        "facts.json",
                        Stat::new(DirentKind::File, b.len() as u64),
                    )]
                }
                Node::ConceptRoot(_) => TAXONOMIES.iter().map(|t| dir(t)).collect(),
                // Which tags a registrant reports is in `facts.json`, and enumerating
                // them here would mean fetching it for a listing.
                Node::Taxonomy(..) => vec![],
                Node::SearchRoot => FILTERS.iter().map(|f| dir(*f)).collect(),
                Node::Field(..) => vec![],
                // Free: the parameters are known and the results are one level down.
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
                // One request, and it names only the page it already holds.
                Node::Pages(pairs) => {
                    let first = self.body(&Node::Page(pairs.clone(), 1)).await?;
                    vec![
                        Dirent::with_stat(
                            "page-001.json",
                            Stat::new(DirentKind::File, first.len() as u64),
                        ),
                        note("_README.md", &String::from_utf8_lossy(
                            &self.pages_note(&pairs).await?,
                        )),
                    ]
                }
                _ => return Err(io::ErrorKind::NotADirectory.into()),
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
            // A page past the end is not an error upstream — it answers with an empty
            // hit list — but as a file it should simply not be there.
            if let Node::Page(pairs, n) = &node {
                let total = self.total_hits(pairs).await?;
                if (*n - 1) * HITS_PER_PAGE >= total.max(1) {
                    return Err(io::ErrorKind::NotFound.into());
                }
            }
            let bytes = match &node {
                Node::PagesReadme(pairs) => self.pages_note(pairs).await?,
                _ => self.body(&node).await?,
            };
            let start = (offset as usize).min(bytes.len());
            let n = buf.len().min(bytes.len() - start);
            buf[..n].copy_from_slice(&bytes[start..start + n]);
            Ok(n)
        })
    }
}

/// `1045810` and `CIK0001045810` and `0001045810` all name one registrant.
///
/// `companies.json` gives the unpadded number and the SEC's own URLs want ten digits,
/// so a caller who read the list and formed a path from what it said would otherwise
/// get nothing. Padding here costs a line; the alternative is a working identifier
/// that does not work.
fn normalize_cik(s: &str) -> Option<String> {
    let digits = s.strip_prefix("CIK").unwrap_or(s);
    if digits.is_empty() || digits.len() > 10 || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    Some(format!("{:0>10}", digits))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(p: &str) -> Option<Node> {
        EdgarFs::parse(Path::new(p))
    }

    #[test]
    fn a_cik_is_the_same_registrant_however_it_was_written() {
        // The form `companies.json` hands out, the form the SEC's URLs use, and the
        // padded number on its own.
        let want = Some(Node::Entity("0001045810".into()));
        assert_eq!(parse("/by-cik/1045810"), want);
        assert_eq!(parse("/by-cik/CIK0001045810"), want);
        assert_eq!(parse("/by-cik/0001045810"), want);
        assert_eq!(parse("/by-cik/12345678901"), None);
        assert_eq!(parse("/by-cik/NVDA"), None);
    }

    #[test]
    fn facts_is_reachable_but_not_advertised() {
        // Four megabytes. Addressable, because it is the only concept index there is.
        assert_eq!(
            parse("/by-cik/1045810/facts.json"),
            Some(Node::Facts("0001045810".into()))
        );
    }

    #[test]
    fn concepts_are_taxonomy_scoped() {
        assert_eq!(
            parse("/by-cik/1045810/concept/us-gaap/Revenues.json"),
            Some(Node::Concept(
                "0001045810".into(),
                "us-gaap".into(),
                "Revenues".into()
            ))
        );
        // A taxonomy the API does not serve is a path that does not exist.
        assert_eq!(parse("/by-cik/1045810/concept/made-up/Revenues.json"), None);
    }

    #[test]
    fn date_bounds_bring_their_own_range_mode() {
        let Some(Node::Query(pairs)) = parse("/search/q/lithium/startdt/2025-01-01") else {
            panic!("query")
        };
        let url = EdgarFs::query_url(&pairs, 1);
        assert!(url.contains("dateRange=custom"), "{url}");
        // And a query without dates does not carry it.
        let Some(Node::Query(p2)) = parse("/search/q/lithium") else {
            panic!("query")
        };
        assert!(!EdgarFs::query_url(&p2, 1).contains("dateRange"));
    }

    #[test]
    fn pages_offset_by_hits_not_by_number() {
        let Some(Node::Page(pairs, n)) = parse("/search/q/lithium/pages/page-003.json") else {
            panic!("page")
        };
        assert_eq!(n, 3);
        assert!(EdgarFs::query_url(&pairs, n).contains("from=200"));
    }

    #[test]
    fn an_unconfirmed_parameter_is_not_a_path() {
        // Accepted upstream and changes nothing, which is worse than a rejection: the
        // count comes back looking like an answer.
        assert_eq!(parse("/search/locationCode/CA"), None);
        assert_eq!(parse("/search/nosuchparam/x"), None);
    }

    /// A ticker is not an address here. Nothing under `by-ticker` parses, so naming
    /// one is a missing directory rather than a request that goes nowhere.
    #[test]
    fn a_ticker_is_not_a_path() {
        for p in ["/by-ticker", "/by-ticker/NVDA", "/by-ticker/NVDA/submissions.json"] {
            assert_eq!(parse(p), None, "{p}");
        }
    }

    /// Listing a registrant must not fetch anything, and must still say what is there.
    ///
    /// Both halves matter. It fetched once: `submissions.json` sat in this listing, its
    /// size is only knowable by fetching, and a directory read asks for every entry's
    /// size — so one ordinary recursive pass over the ticker listing spent 742
    /// requests. Hiding the file fixed the cost and lost the discovery; wrapping it in
    /// a directory keeps both.
    /// The note under `pages` belongs to its query, so it can report that query's own
    /// size. Parsing has to keep the pairs for that; dropping them is how it silently
    /// becomes the same note everywhere.
    #[test]
    fn the_pages_note_keeps_its_query() {
        let Some(Node::PagesReadme(pairs)) = parse("/search/entityName/NVDA/pages/_README.md") else {
            panic!("the note did not parse as a query's own")
        };
        assert!(!pairs.is_empty(), "the note lost the query it belongs to");
        // The same query with and without the note names one query, not two.
        let Some(Node::Pages(same)) = parse("/search/entityName/NVDA/pages") else {
            panic!("pages")
        };
        assert_eq!(pairs, same);
    }

    #[tokio::test]
    async fn listing_a_registrant_names_everything_and_fetches_nothing() {
        let fs = EdgarFs::new("test");
        let entries = fs.list(Path::new("/by-cik/1045810")).await.expect("list");
        assert_eq!(fs.calls(), 0, "listing a registrant sent a request");
        let names: Vec<&str> = entries.iter().map(|e| e.name.as_str()).collect();
        assert_eq!(names, ["submissions", "facts", "concept"]);
        assert!(entries.iter().all(|e| e.kind == DirentKind::Dir));
    }

    #[test]
    fn a_resource_opens_wrapped_or_flat() {
        // The listed path, and the one someone guesses. Only the first is advertised;
        // the second costs nothing to keep and saves a caller from coming away empty.
        let want = Some(Node::Submissions("0001045810".into()));
        assert_eq!(parse("/by-cik/1045810/submissions/submissions.json"), want);
        assert_eq!(parse("/by-cik/1045810/submissions.json"), want);
        assert_eq!(
            parse("/by-cik/1045810/facts"),
            Some(Node::FactsDir("0001045810".into()))
        );
        // The wrapper holds one file, named after it — nothing else.
        assert_eq!(parse("/by-cik/1045810/submissions/facts.json"), None);
    }

    #[tokio::test]
    async fn narrowing_sends_no_requests() {
        let fs = EdgarFs::new("test");
        for p in [
            "/",
            "/by-cik",
            "/by-cik/1045810",
            "/by-cik/1045810/concept",
            "/by-cik/1045810/concept/us-gaap",
            "/search",
            "/search/q",
            "/search/q/lithium",
            "/search/q/lithium/forms/10-K",
            "/by-cik",
        ] {
            fs.list(Path::new(p))
                .await
                .unwrap_or_else(|e| panic!("list {p}: {e}"));
        }
        assert_eq!(fs.calls(), 0, "narrowing sent a request");
    }

    #[tokio::test]
    async fn every_advertised_directory_can_be_entered() {
        let fs = EdgarFs::new("test");
        for base in ["/", "/by-cik/1045810", "/search/q/lithium"] {
            for e in fs.list(Path::new(base)).await.expect("list") {
                if e.name == "pages" {
                    continue; // those would ask the API
                }
                let child = format!("{}/{}", base.trim_end_matches('/'), e.name);
                assert!(
                    EdgarFs::parse(Path::new(&child)).is_some(),
                    "listed but not enterable: {child}"
                );
            }
        }
    }
}

