//! The retry policy every HTTP provider sends its request through: retries with backoff
//! for the statuses a provider recovers from, and a circuit breaker per host so that an
//! outage fails fast instead of paying for every retry of every request.

use std::{collections::HashMap, sync::LazyLock, time::Duration};

use llm_resilience::CircuitBreaker;

use crate::model::api_error;

/// Retries after the first attempt.
const MAX_RETRIES: u32 = 3;
/// The longest wait between attempts, whatever `retry-after` asks for.
const MAX_WAIT: Duration = Duration::from_secs(10);
/// Consecutive failed attempts at one host that open its breaker.
const BREAKER_FAILURES: usize = 5;
/// How long an open breaker refuses requests before it lets one probe through. Each probe
/// that fails doubles it.
const BREAKER_OPEN_FOR: Duration = Duration::from_secs(30);

/// Rate limits, and the transient errors and overloads (529) servers recover from.
pub(super) const RETRY_STATUSES: &[u16] = &[429, 500, 502, 503, 504, 529];

/// How one provider retries.
pub(super) struct RetryPolicy {
    /// What the request is, for errors: "Anthropic API request".
    pub label: &'static str,
    /// Statuses worth another attempt.
    pub retry_statuses: &'static [u16],
    /// Whether the server's `retry-after` (in seconds) sets the wait.
    pub honor_retry_after: bool,
    /// Given a retryable status and its body, whether the failure will not pass with time
    /// (a quota that is used up), so it is returned at once.
    pub permanent: Option<fn(u16, &str) -> bool>,
}

impl RetryPolicy {
    pub(super) const fn new(label: &'static str) -> Self {
        Self {
            label,
            retry_statuses: RETRY_STATUSES,
            honor_retry_after: false,
            permanent: None,
        }
    }
}

/// One breaker per host and port, shared by every model that calls it.
static BREAKERS: LazyLock<parking_lot::Mutex<HashMap<String, CircuitBreaker>>> =
    LazyLock::new(Default::default);

fn breaker_for(url: &str) -> (String, CircuitBreaker) {
    let host = url::Url::parse(url)
        .ok()
        .and_then(|u| {
            u.host_str()
                .map(|h| format!("{h}:{}", u.port_or_known_default().unwrap_or(0)))
        })
        .unwrap_or_else(|| url.to_owned());
    let breaker = BREAKERS
        .lock()
        .entry(host.clone())
        .or_insert_with(|| CircuitBreaker::new(BREAKER_FAILURES, 1.0, BREAKER_OPEN_FOR))
        .clone();
    (host, breaker)
}

/// Sends the request `build` makes to `url`, retrying as `policy` says, and returns the
/// successful response unread.
pub(super) async fn send_with_retry(
    url: &str,
    policy: &RetryPolicy,
    build: impl FnMut() -> reqwest::RequestBuilder,
) -> anyhow::Result<reqwest::Response> {
    let (host, breaker) = breaker_for(url);
    send_through(&host, &breaker, policy, Duration::from_secs(1), build).await
}

/// [`send_with_retry`] with its breaker and backoff unit given, so tests run in
/// milliseconds.
async fn send_through(
    host: &str,
    breaker: &CircuitBreaker,
    policy: &RetryPolicy,
    backoff_unit: Duration,
    mut build: impl FnMut() -> reqwest::RequestBuilder,
) -> anyhow::Result<reqwest::Response> {
    let label = policy.label;
    for attempt in 0..=MAX_RETRIES {
        let Some(permit) = breaker.try_acquire() else {
            let wait = breaker.retry_after_sync().unwrap_or_default();
            return Err(api_error(
                503,
                format!(
                    "{label} not sent: {host} failed {BREAKER_FAILURES} attempts in a row, so it is skipped for another {}s",
                    wait.as_secs()
                ),
            ));
        };
        let response = match build().send().await {
            Ok(response) => response,
            Err(e) => {
                permit.failure();
                return Err(e.into());
            }
        };

        let status = response.status();
        if status.is_success() {
            permit.success();
            return Ok(response);
        }
        let code = status.as_u16();
        if !policy.retry_statuses.contains(&code) {
            // A request the server refused says nothing about the server's health.
            drop(permit);
            let text = response.text().await.unwrap_or_default();
            return Err(api_error(
                code,
                format!("{label} failed with status {status}: {text}"),
            ));
        }

        let retry_after = policy
            .honor_retry_after
            .then(|| response.headers().get("retry-after"))
            .flatten()
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.parse::<u64>().ok())
            .map(|secs| backoff_unit * secs as u32);
        // The body is read only when a policy or the last attempt needs it.
        let text = if policy.permanent.is_some() || attempt == MAX_RETRIES {
            response.text().await.unwrap_or_default()
        } else {
            String::new()
        };
        if policy
            .permanent
            .is_some_and(|permanent| permanent(code, &text))
        {
            drop(permit);
            return Err(api_error(
                code,
                format!("{label} failed with status {status}: {text}"),
            ));
        }
        permit.failure();

        if attempt == MAX_RETRIES {
            return Err(api_error(
                code,
                format!("{label} failed with status {status}: {text}"),
            ));
        }
        let wait = retry_after
            .unwrap_or(backoff_unit * (1u32 << attempt))
            .min(MAX_WAIT);
        log::warn!(
            "Request failed with status {status}. Retrying after {}s (attempt {}/{MAX_RETRIES})",
            wait.as_secs_f32(),
            attempt + 1
        );
        tokio::time::sleep(wait).await;
    }
    unreachable!("retry loop returns on every path")
}

#[cfg(test)]
mod tests {
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };

    use axum::{Router, http::StatusCode, routing::post};

    use super::*;

    /// A server that answers every POST with `status` and `body`, and counts the calls.
    async fn serve(status: StatusCode, body: &'static str) -> (String, Arc<AtomicUsize>) {
        let hits = Arc::new(AtomicUsize::new(0));
        let counter = hits.clone();
        let app = Router::new().route(
            "/",
            post(move || {
                let counter = counter.clone();
                async move {
                    counter.fetch_add(1, Ordering::SeqCst);
                    (status, body)
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}/", listener.local_addr().unwrap());
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        (url, hits)
    }

    fn breaker() -> CircuitBreaker {
        CircuitBreaker::new(BREAKER_FAILURES, 1.0, BREAKER_OPEN_FOR)
    }

    async fn send(
        url: &str,
        breaker: &CircuitBreaker,
        policy: &RetryPolicy,
    ) -> anyhow::Result<reqwest::Response> {
        let client = reqwest::Client::new();
        send_through(url, breaker, policy, Duration::from_millis(1), || {
            client.post(url)
        })
        .await
    }

    #[tokio::test]
    async fn an_outage_stops_costing_calls_once_the_breaker_opens() {
        let (url, hits) = serve(StatusCode::SERVICE_UNAVAILABLE, "overloaded").await;
        let breaker = breaker();
        let policy = RetryPolicy::new("test request");

        // The first request makes all four attempts; the second opens the breaker on its
        // first failure, since five attempts in a row have failed.
        let err = send(&url, &breaker, &policy).await.unwrap_err();
        assert!(err.to_string().contains("503"), "{err}");
        assert_eq!(hits.load(Ordering::SeqCst), 4);
        send(&url, &breaker, &policy).await.unwrap_err();
        assert_eq!(hits.load(Ordering::SeqCst), 5);

        // Every request after that fails at once without reaching the server. With a count
        // cap alone, these ten would have made forty more calls.
        for _ in 0..10 {
            let err = send(&url, &breaker, &policy).await.unwrap_err();
            assert!(err.to_string().contains("not sent"), "{err}");
        }
        assert_eq!(hits.load(Ordering::SeqCst), 5);
    }

    #[tokio::test]
    async fn a_refused_request_is_not_retried_and_does_not_open_the_breaker() {
        let (url, hits) = serve(StatusCode::BAD_REQUEST, "bad schema").await;
        let breaker = breaker();
        let policy = RetryPolicy::new("test request");
        for _ in 0..10 {
            let err = send(&url, &breaker, &policy).await.unwrap_err();
            assert!(err.to_string().contains("bad schema"), "{err}");
        }
        assert_eq!(hits.load(Ordering::SeqCst), 10);
        assert!(breaker.is_closed_sync());
    }

    #[tokio::test]
    async fn a_used_up_quota_is_returned_at_once() {
        let (url, hits) = serve(StatusCode::TOO_MANY_REQUESTS, "insufficient_quota").await;
        let breaker = breaker();
        let policy = RetryPolicy {
            permanent: Some(|_, text| text.contains("insufficient_quota")),
            ..RetryPolicy::new("test request")
        };
        let err = send(&url, &breaker, &policy).await.unwrap_err();
        assert!(err.to_string().contains("insufficient_quota"), "{err}");
        assert_eq!(hits.load(Ordering::SeqCst), 1);
        assert!(breaker.is_closed_sync());
    }

    #[tokio::test]
    async fn a_transient_failure_is_retried_until_it_passes() {
        let hits = Arc::new(AtomicUsize::new(0));
        let counter = hits.clone();
        let app = Router::new().route(
            "/",
            post(move || {
                let counter = counter.clone();
                async move {
                    if counter.fetch_add(1, Ordering::SeqCst) < 2 {
                        (StatusCode::BAD_GATEWAY, "")
                    } else {
                        (StatusCode::OK, "ok")
                    }
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}/", listener.local_addr().unwrap());
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

        let breaker = breaker();
        let response = send(&url, &breaker, &RetryPolicy::new("test request"))
            .await
            .unwrap();
        assert_eq!(response.text().await.unwrap(), "ok");
        assert_eq!(hits.load(Ordering::SeqCst), 3);
        assert!(breaker.is_closed_sync());
    }
}
