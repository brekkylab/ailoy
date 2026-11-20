//! [`WebSearchProvider::Web`](super::WebSearchProvider::Web): the engines' own result pages,
//! one engine a call in turn. Engines with a search API of their own, Brave and Naver, are
//! searched through that API instead.

mod bing;
mod duckduckgo;
mod google;
mod mojeek;
mod startpage;
mod yahoo;
mod yandex;

use std::sync::atomic::{AtomicUsize, Ordering};

use async_trait::async_trait;
use reqwest::Client;

use bing::Bing;
use duckduckgo::DuckDuckGo;
use google::Google;
use mojeek::Mojeek;
use startpage::Startpage;
use yahoo::Yahoo;
use yandex::Yandex;

use super::engine::{SearchEngine, SearchError, SearchResult};

pub struct Web {
    engines: Vec<Box<dyn SearchEngine>>,
    /// The engine the next call starts with.
    next: AtomicUsize,
}

impl Web {
    pub fn new() -> Result<Self, SearchError> {
        Ok(Self {
            engines: vec![
                Box::new(Google::new()?),
                Box::new(Bing::new()?),
                Box::new(DuckDuckGo::new()?),
                Box::new(Startpage::new()?),
                Box::new(Yahoo::new()?),
                Box::new(Mojeek::new()?),
                Box::new(Yandex::new()?),
            ],
            next: AtomicUsize::new(0),
        })
    }
}

#[async_trait]
impl SearchEngine for Web {
    fn name(&self) -> &'static str {
        "web"
    }

    /// Ask one engine, starting a call later in the list than the last one did, so calls
    /// spread across engines. When an engine blocks, fails or finds nothing, the next one is
    /// asked; nothing found anywhere is an empty list, and only failures everywhere an error.
    async fn search(
        &self,
        client: &Client,
        query: &str,
        max_results: usize,
    ) -> Result<Vec<SearchResult>, SearchError> {
        let start = self.next.fetch_add(1, Ordering::Relaxed);
        let mut failures = Vec::new();
        for i in 0..self.engines.len() {
            let engine = &self.engines[(start + i) % self.engines.len()];
            match engine.search(client, query, max_results).await {
                Ok(mut results) if !results.is_empty() => {
                    results.truncate(max_results);
                    return Ok(results);
                }
                Ok(_) | Err(SearchError::NoResults) => {}
                Err(e) => {
                    log::warn!("search engine '{}' failed: {e}", engine.name());
                    failures.push(format!("{}: {e}", engine.name()));
                }
            }
        }
        if failures.len() == self.engines.len() {
            return Err(SearchError::Api(format!(
                "every engine failed ({})",
                failures.join("; ")
            )));
        }
        Ok(Vec::new())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type Answer = fn() -> Result<Vec<SearchResult>, SearchError>;

    /// An engine answering with `answer`, counting its calls.
    struct Scripted {
        answer: Answer,
        calls: AtomicUsize,
    }

    #[async_trait]
    impl SearchEngine for Scripted {
        fn name(&self) -> &'static str {
            "scripted"
        }

        async fn search(
            &self,
            _: &Client,
            _: &str,
            _: usize,
        ) -> Result<Vec<SearchResult>, SearchError> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            (self.answer)()
        }
    }

    fn found() -> Result<Vec<SearchResult>, SearchError> {
        Ok((0..3)
            .map(|i| SearchResult {
                title: format!("r{i}"),
                url: format!("https://r{i}.example"),
                description: String::new(),
                engine: "scripted",
            })
            .collect())
    }

    fn web(answers: &[Answer]) -> Web {
        let engines = answers
            .iter()
            .map(|&answer| {
                let calls = AtomicUsize::new(0);
                Box::new(Scripted { answer, calls }) as Box<dyn SearchEngine>
            })
            .collect();
        Web {
            engines,
            next: AtomicUsize::new(0),
        }
    }

    /// How many times each engine of `web` was called.
    fn calls(web: &Web) -> Vec<usize> {
        web.engines
            .iter()
            .map(|engine| {
                // Deliberate, to keep the test short: every engine `web` built is a
                // `Scripted`, so the cast is sound, and it spares threading shared counters in.
                let scripted = engine.as_ref() as *const dyn SearchEngine as *const Scripted;
                unsafe { (*scripted).calls.load(Ordering::Relaxed) }
            })
            .collect()
    }

    #[tokio::test]
    async fn test_calls_take_engines_in_turn() {
        let web = web(&[found, found, found]);
        let client = Client::new();
        for _ in 0..4 {
            assert_eq!(web.search(&client, "q", 2).await.unwrap().len(), 2);
        }
        assert_eq!(calls(&web), [2, 1, 1]);
    }

    #[tokio::test]
    async fn test_a_blocked_or_empty_engine_passes_to_the_next() {
        let web = web(&[|| Err(SearchError::Blocked), || Ok(Vec::new()), found]);
        let results = web.search(&Client::new(), "q", 10).await.unwrap();
        assert_eq!(results.len(), 3);
        assert_eq!(calls(&web), [1, 1, 1]);
    }

    #[tokio::test]
    async fn test_nothing_found_is_empty_and_failing_everywhere_an_error() {
        let client = Client::new();
        let empty = web(&[|| Err(SearchError::NoResults), || Ok(Vec::new())]);
        assert!(empty.search(&client, "q", 10).await.unwrap().is_empty());

        let blocked = web(&[|| Err(SearchError::Blocked), || Err(SearchError::Blocked)]);
        let err = blocked.search(&client, "q", 10).await.unwrap_err();
        assert!(err.to_string().contains("every engine failed"), "{err}");
    }
    #[tokio::test]
    #[ignore = "requires network"]
    async fn test_the_engines_find_results() {
        let web = Web::new().unwrap();
        let client = Client::new();
        for _ in 0..3 {
            let results = web.search(&client, "rust language", 5).await.unwrap();
            assert!(!results.is_empty());
            assert!(results.len() <= 5);
            println!("{}: {}", results[0].engine, results[0].url);
        }
    }
}
