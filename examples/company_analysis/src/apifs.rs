//! The part every API-backed [`FileSystem`](cortex::fs::FileSystem) needs and none of
//! them differ on: one client, one cache, one count of what was spent.
//!
//! What differs between sources is the path grammar and which URL a path means. That
//! stays in each store. Fetching does not, and four copies of a retry loop would be
//! four places for the backoff to drift.

use std::{
    collections::{BTreeMap, HashMap},
    io,
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};

/// Waits before the 1st and 2nd retry of a rate-limited or 5xx request.
const BACKOFF: [Duration; 2] = [Duration::from_millis(400), Duration::from_millis(1600)];

/// A client, the bodies it has fetched, and the number of requests that took.
pub struct Fetcher {
    http: reqwest::Client,
    /// key → bytes. Also the size source for `stat`.
    ///
    /// Kept for the life of the store, which is one run. Two reasons, and neither is
    /// speed. A `stat` and the reads after it have to agree on a size, and a report
    /// that cites a path has to still mean the same bytes when someone follows the
    /// citation — an index that moved under a 40-turn investigation would leave its
    /// own evidence disagreeing with itself. A run is minutes long, so one snapshot
    /// per run is the freshness that matters. (A long-lived host would want a bound
    /// here; these stores are built per run and dropped with it.)
    bodies: Mutex<HashMap<String, Arc<Vec<u8>>>>,
    calls: AtomicUsize,
    /// Requests per kind of resource. The kind comes from the caller, not from parsing
    /// the key: a key is only a string that looks like a path.
    kinds: Mutex<BTreeMap<&'static str, usize>>,
    /// Requests per cache key, which the kind alone cannot separate: one file fetched
    /// many times counts the same as many files fetched once.
    keys: Mutex<BTreeMap<String, usize>>,
}

impl Fetcher {
    /// `user_agent` is not decoration everywhere: SEC rejects a request without one.
    pub fn new(user_agent: &str) -> Self {
        Self {
            http: reqwest::Client::builder()
                .user_agent(user_agent.to_string())
                .build()
                .expect("reqwest client"),
            bodies: Mutex::new(HashMap::new()),
            calls: AtomicUsize::new(0),
            kinds: Mutex::new(BTreeMap::new()),
            keys: Mutex::new(BTreeMap::new()),
        }
    }

    /// How many requests this store has sent. A cache hit is not one.
    pub fn calls(&self) -> usize {
        self.calls.load(Ordering::Relaxed)
    }

    /// What the requests were spent on, heaviest first.
    pub fn breakdown(&self) -> Vec<(&'static str, usize)> {
        let mut v: Vec<_> = self
            .kinds
            .lock()
            .unwrap()
            .iter()
            .map(|(k, n)| (*k, *n))
            .collect();
        v.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(b.0)));
        v
    }

    /// The `n` costliest paths, heaviest first.
    pub fn hot_keys(&self, n: usize) -> Vec<(String, usize)> {
        let mut v: Vec<_> = self
            .keys
            .lock()
            .unwrap()
            .iter()
            .map(|(k, c)| (k.clone(), *c))
            .collect();
        v.sort_by_key(|(_, c)| std::cmp::Reverse(*c));
        v.truncate(n);
        v
    }

    pub fn distinct_keys(&self) -> usize {
        self.keys.lock().unwrap().len()
    }

    pub fn cached(&self, key: &str) -> Option<Arc<Vec<u8>>> {
        self.bodies.lock().unwrap().get(key).cloned()
    }

    /// Fetch `url` unless a copy is already held under `key`.
    ///
    /// The one place a body enters the cache, so `stat`'s size and the reads after it
    /// come from the same fetch.
    pub async fn get(&self, kind: &'static str, key: &str, url: &str) -> io::Result<Arc<Vec<u8>>> {
        if let Some(hit) = self.cached(key) {
            return Ok(hit);
        }
        *self.kinds.lock().unwrap().entry(kind).or_insert(0) += 1;
        *self.keys.lock().unwrap().entry(key.to_string()).or_insert(0) += 1;
        let bytes = Arc::new(self.fetch(url).await?);
        self.bodies
            .lock()
            .unwrap()
            .insert(key.to_string(), bytes.clone());
        Ok(bytes)
    }

    /// GET with retries, returning the body whatever the status.
    ///
    /// A non-2xx body is JSON too, and handing it back is how a caller learns *why*
    /// rather than only that something failed. It is also what these sources do among
    /// themselves: DART answers `{"status":"013"}` and EDINET a 401, both under HTTP
    /// 200. Only a transport failure or a 5xx that outlived its retries is an error.
    async fn fetch(&self, url: &str) -> io::Result<Vec<u8>> {
        let mut last = String::new();
        for attempt in 0..=BACKOFF.len() {
            self.calls.fetch_add(1, Ordering::Relaxed);
            match self.http.get(url).send().await {
                Ok(resp) => {
                    let status = resp.status();
                    let body = resp
                        .bytes()
                        .await
                        .map_err(|e| io::Error::other(e.to_string()))?
                        .to_vec();
                    if status.is_success() || status.is_client_error() {
                        return Ok(body);
                    }
                    last = format!("HTTP {status}");
                }
                Err(e) => last = e.to_string(),
            }
            if let Some(wait) = BACKOFF.get(attempt) {
                tokio::time::sleep(*wait).await;
            }
        }
        Err(io::Error::other(format!("{url}: {last}")))
    }
}

/// Percent-decoding for a path segment, so a value containing `/` can be named.
pub fn percent_decode(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'%' && i + 2 < b.len() {
            if let Ok(v) = u8::from_str_radix(&s[i + 1..i + 3], 16) {
                out.push(v);
                i += 3;
                continue;
            }
        }
        out.push(b[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// Encoding for a query parameter. Deliberately conservative: a comma is left alone
/// because these APIs read it as a list separator, and encoding it would change the
/// question rather than transmit it.
pub fn urlencode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' | b',' => {
                out.push(b as char)
            }
            b' ' => out.push_str("%20"),
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

pub fn io_err<E: std::fmt::Display>(e: E) -> io::Error {
    io::Error::other(e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_comma_survives_encoding() {
        // These APIs read `KR,JP` as OR. Percent-encoding it would ask a different
        // question, so the one reserved character we leave alone is this one.
        assert_eq!(urlencode("KR,JP"), "KR,JP");
        assert_eq!(urlencode("Samsung Electronics"), "Samsung%20Electronics");
        assert_eq!(urlencode("a/b"), "a%2Fb");
    }

    #[test]
    fn decoding_is_the_inverse_for_a_path_segment() {
        for s in ["A/B", "plain", "a b", "100%"] {
            assert_eq!(percent_decode(&urlencode(s)), s);
        }
    }
}
