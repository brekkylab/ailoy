//! What a web address points at, looked up before it is added: a YouTube video or an
//! ordinary page, and the title it goes by.

use std::time::Duration;

use reqwest::{Client, Url};
use serde::{Deserialize, Serialize};

/// How much of a page is read looking for its title; `<head>` comes well before this.
const HEAD_BYTES: usize = 256 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Kind {
    Page,
    Youtube,
}

#[derive(Debug, Serialize)]
pub struct Peek {
    pub kind: Kind,
    /// The page's `<title>` or the video's; `None` when it could not be had.
    pub title: Option<String>,
    /// The channel, for a video.
    pub author: Option<String>,
}

/// Whether `url` is on YouTube, judged by its host alone.
fn is_youtube(url: &Url) -> bool {
    let host = url.host_str().unwrap_or_default();
    let host = host.strip_prefix("www.").unwrap_or(host);
    matches!(
        host,
        "youtube.com" | "m.youtube.com" | "music.youtube.com" | "youtu.be" | "youtube-nocookie.com"
    )
}

fn client() -> reqwest::Result<Client> {
    Client::builder()
        .timeout(Duration::from_secs(8))
        // Some sites answer a bare client with a bot wall instead of the page.
        .user_agent("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.0 Safari/605.1.15")
        .build()
}

/// A video's title and channel through YouTube's oEmbed, which needs no key.
async fn youtube(client: &Client, url: &Url) -> reqwest::Result<Peek> {
    #[derive(Deserialize)]
    struct OEmbed {
        title: Option<String>,
        author_name: Option<String>,
    }
    let endpoint = Url::parse_with_params(
        "https://www.youtube.com/oembed",
        &[("format", "json"), ("url", url.as_str())],
    )
    .expect("the oEmbed endpoint is a valid URL");
    let found: OEmbed = client
        .get(endpoint)
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    Ok(Peek {
        kind: Kind::Youtube,
        title: found.title,
        author: found.author_name,
    })
}

/// A page's title, from the first [`HEAD_BYTES`] of it.
async fn page(client: &Client, url: &Url) -> reqwest::Result<Peek> {
    let mut response = client.get(url.clone()).send().await?.error_for_status()?;
    let html = response
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .is_none_or(|v| v.contains("html"));
    let mut head = Vec::new();
    if html {
        while head.len() < HEAD_BYTES {
            match response.chunk().await? {
                Some(chunk) => head.extend_from_slice(&chunk),
                None => break,
            }
        }
    }
    Ok(Peek {
        kind: Kind::Page,
        title: title_of(&String::from_utf8_lossy(&head)),
        author: None,
    })
}

/// The text of `<title>`, else `og:title`, entities decoded and whitespace folded.
fn title_of(html: &str) -> Option<String> {
    let lower = html.to_ascii_lowercase();
    let from_tag = lower.find("<title").and_then(|start| {
        let open = start + lower[start..].find('>')? + 1;
        let close = open + lower[open..].find("</title")?;
        Some(&html[open..close])
    });
    let found = from_tag.or_else(|| meta_content(html, &lower, "og:title"))?;
    let folded = decode_entities(found).split_whitespace().collect::<Vec<_>>().join(" ");
    (!folded.is_empty()).then_some(folded)
}

/// The `content` of the `<meta>` whose `property` is `property`.
fn meta_content<'a>(html: &'a str, lower: &str, property: &str) -> Option<&'a str> {
    let mut from = 0;
    while let Some(at) = lower[from..].find("<meta") {
        let start = from + at;
        let end = start + lower[start..].find('>')?;
        let tag = &lower[start..end];
        if tag.contains(&format!("\"{property}\"")) || tag.contains(&format!("'{property}'")) {
            let key = tag.find("content=")? + "content=".len();
            let quote = tag[key..].chars().next()?;
            let value = start + key + 1;
            let len = lower[value..end].find(quote)?;
            return Some(&html[value..value + len]);
        }
        from = end;
    }
    None
}

/// The handful of entities titles actually carry, and numeric ones.
fn decode_entities(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(amp) = rest.find('&') {
        out.push_str(&rest[..amp]);
        rest = &rest[amp..];
        let decoded = rest.find(';').filter(|&semi| semi <= 10).and_then(|semi| {
            let name = &rest[1..semi];
            let c = match name {
                "amp" => '&',
                "lt" => '<',
                "gt" => '>',
                "quot" => '"',
                "apos" => '\'',
                "nbsp" => ' ',
                _ => {
                    let code = match name.strip_prefix("#x").or_else(|| name.strip_prefix("#X")) {
                        Some(hex) => u32::from_str_radix(hex, 16).ok()?,
                        None => name.strip_prefix('#')?.parse().ok()?,
                    };
                    char::from_u32(code)?
                }
            };
            Some((c, semi + 1))
        });
        match decoded {
            Some((c, len)) => {
                out.push(c);
                rest = &rest[len..];
            }
            None => {
                out.push('&');
                rest = &rest[1..];
            }
        }
    }
    out.push_str(rest);
    out
}

#[tauri::command]
pub async fn peek_url(url: String) -> Result<Peek, String> {
    let url = Url::parse(&url).map_err(|err| err.to_string())?;
    let client = client().map_err(|err| err.to_string())?;
    let found = if is_youtube(&url) {
        youtube(&client, &url).await
    } else {
        page(&client, &url).await
    };
    found.map_err(|err| err.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn youtube_is_told_by_host() {
        let yes = ["https://www.youtube.com/watch?v=x", "https://youtu.be/x", "https://m.youtube.com/shorts/x"];
        let no = ["https://example.com/youtube.com", "https://notyoutube.com/watch"];
        assert!(yes.iter().all(|u| is_youtube(&Url::parse(u).unwrap())));
        assert!(!no.iter().any(|u| is_youtube(&Url::parse(u).unwrap())));
    }

    #[test]
    fn title_comes_from_the_tag_then_og() {
        assert_eq!(
            title_of("<html><head><TITLE lang=en>\n  Tom &amp; Jerry &#8212; &#x41;  </TITLE>").as_deref(),
            Some("Tom & Jerry — A")
        );
        assert_eq!(
            title_of(r#"<meta content="x"><meta property="og:title" content='Café'>"#).as_deref(),
            Some("Café")
        );
        assert_eq!(title_of("<title></title>"), None);
        assert_eq!(decode_entities("a & b &bogus; c"), "a & b &bogus; c");
    }
}
