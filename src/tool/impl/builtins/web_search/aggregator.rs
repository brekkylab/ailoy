use std::collections::HashMap;

use futures::future::join_all;
use reqwest::Client;

use super::{
    engine::{SearchEngine, SearchError},
    engines::WebSearchEngineKind,
};

/// A search result aggregated from multiple engines.
#[derive(Debug, Clone)]
pub struct AggregatedResult {
    pub title: String,
    pub url: String,
    pub description: String,
    /// Names of engines that returned this URL.
    pub sources: Vec<&'static str>,
    /// Number of engines that returned this URL (used for ranking).
    pub relevance: f32,
}

/// Normalizes a URL for deduplication:
/// - lowercases scheme and host
/// - removes "www." prefix
/// - strips trailing slash
/// - removes common tracking query params (utm_*, ref, etc.)
pub fn normalize_url(url: &str) -> String {
    let url = url.trim();

    // Remove fragment
    let url = url.split('#').next().unwrap_or(url);

    // Split at query string
    let (base, query) = match url.split_once('?') {
        Some((b, q)) => (b, Some(q)),
        None => (url, None),
    };

    // Normalize the base: only lowercase scheme+host, preserve path case (RFC 3986)
    let base = base.trim_end_matches('/');

    // Split into scheme+host and path
    let base = if let Some(authority_start) = base.find("://") {
        let (scheme, after_scheme) = base.split_at(authority_start + 3);
        let (host_part, path_part) = after_scheme
            .split_once('/')
            .map(|(h, p)| (h, format!("/{}", p)))
            .unwrap_or((after_scheme, String::new()));
        // Only lowercase scheme and host; path is case-sensitive
        format!(
            "{}{}{}",
            scheme.to_lowercase(),
            host_part.to_lowercase(),
            path_part
        )
    } else {
        base.to_lowercase()
    };

    // Remove www. from host (after scheme://)
    let base = if let Some(after_scheme) = base.strip_prefix("https://www.") {
        format!("https://{}", after_scheme)
    } else if let Some(after_scheme) = base.strip_prefix("http://www.") {
        format!("http://{}", after_scheme)
    } else {
        base
    };

    // Keep only non-tracking query params
    let tracking_params = [
        "utm_source",
        "utm_medium",
        "utm_campaign",
        "utm_term",
        "utm_content",
        "ref",
        "referrer",
        "source",
    ];
    if let Some(q) = query {
        let filtered: Vec<&str> = q
            .split('&')
            .filter(|param| {
                let key = param.split('=').next().unwrap_or("");
                !tracking_params.contains(&key)
            })
            .collect();
        if filtered.is_empty() {
            base
        } else {
            format!("{}?{}", base, filtered.join("&"))
        }
    } else {
        base
    }
}

pub struct MetaSearcher {
    pub engines: Vec<Box<dyn SearchEngine>>,
    pub client: Client,
}

impl MetaSearcher {
    /// Constructs a `MetaSearcher` using the given engine selection.
    ///
    /// An empty `engines` slice uses all available engines (default meta-search behaviour).
    pub fn new(engines: Vec<WebSearchEngineKind>) -> Self {
        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(10))
            .user_agent(super::engine::gen_useragent())
            .build()
            .expect("Failed to build HTTP client");

        let kinds = if engines.is_empty() {
            WebSearchEngineKind::ALL.to_vec()
        } else {
            engines
        };
        let engines: Vec<Box<dyn SearchEngine>> =
            kinds.into_iter().map(|k| k.instantiate()).collect();

        Self { engines, client }
    }

    pub async fn search(&self, query: &str, max_results: usize) -> Vec<AggregatedResult> {
        // Fan-out: search all engines concurrently
        let futures: Vec<_> = self
            .engines
            .iter()
            .map(|engine| engine.search(&self.client, query, max_results))
            .collect();

        let engine_results = join_all(futures).await;

        // RRF constant: k=60 is standard (Cormack et al., 2009).
        // Score for a URL at rank r from one engine = 1 / (k + r + 1).
        const RRF_K: f32 = 60.0;

        // Deduplicate by normalized URL, accumulating RRF scores per-engine result list.
        let mut seen: HashMap<String, usize> = HashMap::new();
        let mut deduped: Vec<AggregatedResult> = Vec::new();

        for (i, result) in engine_results.into_iter().enumerate() {
            match result {
                Ok(results) => {
                    for (rank, result) in results.into_iter().enumerate() {
                        let key = normalize_url(&result.url);
                        let rrf_score = 1.0 / (RRF_K + rank as f32 + 1.0);
                        if let Some(&idx) = seen.get(&key) {
                            deduped[idx].sources.push(result.engine);
                            deduped[idx].relevance += rrf_score;
                        } else {
                            let idx = deduped.len();
                            seen.insert(key, idx);
                            deduped.push(AggregatedResult {
                                title: result.title,
                                url: result.url,
                                description: result.description,
                                sources: vec![result.engine],
                                relevance: rrf_score,
                            });
                        }
                    }
                }
                Err(SearchError::NoResults) => {} // silent
                Err(e) => log::warn!("Search engine '{}' failed: {}", self.engines[i].name(), e),
            }
        }

        // Rank by RRF score (higher = ranked higher across more engines)
        deduped.sort_by(|a, b| {
            b.relevance
                .partial_cmp(&a.relevance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        deduped.truncate(max_results);
        deduped
    }
}

#[cfg(test)]
mod tests {
    use async_trait::async_trait;
    use reqwest::Client;

    use super::{
        super::engine::{SearchEngine, SearchError, SearchResult},
        *,
    };

    // --- normalize_url tests ---

    #[test]
    fn test_normalize_removes_www() {
        assert_eq!(
            normalize_url("https://www.example.com/page"),
            "https://example.com/page"
        );
    }

    #[test]
    fn test_normalize_removes_trailing_slash() {
        assert_eq!(normalize_url("https://example.com/"), "https://example.com");
    }

    #[test]
    fn test_normalize_removes_tracking_params() {
        assert_eq!(
            normalize_url("https://example.com/page?utm_source=google&id=42"),
            "https://example.com/page?id=42"
        );
    }

    #[test]
    fn test_normalize_removes_fragment() {
        assert_eq!(
            normalize_url("https://example.com/page#section"),
            "https://example.com/page"
        );
    }

    #[test]
    fn test_normalize_lowercases_host() {
        // Only scheme and host should be lowercased, not the path
        assert_eq!(
            normalize_url("HTTPS://EXAMPLE.COM/Path"),
            "https://example.com/Path"
        );
    }

    // --- MetaSearcher aggregation tests ---

    struct MockEngine {
        name: &'static str,
        results: Vec<SearchResult>,
    }

    #[async_trait]
    impl SearchEngine for MockEngine {
        fn name(&self) -> &'static str {
            self.name
        }

        async fn search(
            &self,
            _client: &Client,
            _query: &str,
            _max_results: usize,
        ) -> Result<Vec<SearchResult>, SearchError> {
            Ok(self.results.clone())
        }
    }

    fn make_result(title: &str, url: &str, engine: &'static str) -> SearchResult {
        SearchResult {
            title: title.to_string(),
            url: url.to_string(),
            description: "test".to_string(),
            engine,
        }
    }

    fn make_searcher(engines: Vec<Box<dyn SearchEngine>>) -> MetaSearcher {
        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(5))
            .build()
            .unwrap();
        MetaSearcher { engines, client }
    }

    #[tokio::test]
    async fn test_deduplication_by_url() {
        let searcher = make_searcher(vec![
            Box::new(MockEngine {
                name: "A",
                results: vec![make_result("Rust", "https://www.rust-lang.org", "A")],
            }),
            Box::new(MockEngine {
                name: "B",
                results: vec![make_result("Rust Lang", "https://rust-lang.org/", "B")],
            }),
        ]);

        let results = searcher.search("rust", 10).await;
        assert_eq!(
            results.len(),
            1,
            "www. and trailing slash should deduplicate"
        );
        assert_eq!(results[0].sources.len(), 2);
        // With RRF, relevance = sum of 1/(k+rank+1) per engine; always > 0
        assert!(results[0].relevance > 0.0);
    }

    #[tokio::test]
    async fn test_ranking_by_relevance() {
        let searcher = make_searcher(vec![
            Box::new(MockEngine {
                name: "A",
                results: vec![
                    make_result("Alpha", "https://alpha.com", "A"),
                    make_result("Beta", "https://beta.com", "A"),
                ],
            }),
            Box::new(MockEngine {
                name: "B",
                results: vec![make_result("Beta Again", "https://beta.com", "B")],
            }),
        ]);

        let results = searcher.search("test", 10).await;
        assert_eq!(
            results[0].url, "https://beta.com",
            "URL appearing in 2 engines should rank first"
        );
    }

    #[tokio::test]
    async fn test_graceful_degradation_on_all_failure() {
        struct FailEngine;
        #[async_trait]
        impl SearchEngine for FailEngine {
            fn name(&self) -> &'static str {
                "FailEngine"
            }
            async fn search(
                &self,
                _: &Client,
                _: &str,
                _: usize,
            ) -> Result<Vec<SearchResult>, SearchError> {
                Err(SearchError::Blocked)
            }
        }

        let searcher = make_searcher(vec![Box::new(FailEngine)]);
        let results = searcher.search("anything", 10).await;
        assert_eq!(
            results.len(),
            0,
            "All engines failing should return empty results"
        );
    }

    #[tokio::test]
    async fn test_max_results_limit() {
        let searcher = make_searcher(vec![Box::new(MockEngine {
            name: "A",
            results: (0..20)
                .map(|i| {
                    make_result(
                        &format!("Result {}", i),
                        &format!("https://r{}.com", i),
                        "A",
                    )
                })
                .collect(),
        })]);

        let results = searcher.search("test", 5).await;
        assert_eq!(results.len(), 5);
    }

    // --- WebSearchEngineKind / MetaSearcher::new engine selection tests ---

    #[test]
    fn test_empty_engines_uses_all() {
        let searcher = MetaSearcher::new(vec![]);
        assert_eq!(
            searcher.engines.len(),
            WebSearchEngineKind::ALL.len(),
            "empty engines vec should use all available engines"
        );
        let names: Vec<&str> = searcher.engines.iter().map(|e| e.name()).collect();
        for kind in WebSearchEngineKind::ALL {
            assert!(
                names.contains(&kind.name()),
                "engine {:?} should be present",
                kind
            );
        }
    }

    #[test]
    fn test_specific_engines_subset() {
        let kinds = vec![WebSearchEngineKind::Google, WebSearchEngineKind::Brave];
        let searcher = MetaSearcher::new(kinds.clone());
        assert_eq!(searcher.engines.len(), 2);
        let names: Vec<&str> = searcher.engines.iter().map(|e| e.name()).collect();
        for kind in &kinds {
            assert!(
                names.contains(&kind.name()),
                "engine {:?} should be present",
                kind
            );
        }
        assert!(
            !names.contains(&WebSearchEngineKind::Bing.name()),
            "Bing should not be present"
        );
    }
}
