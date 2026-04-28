mod aggregator;
mod engine;
mod engines;

use std::sync::Arc;

use aggregator::MetaSearcher;

use crate::{
    datatype::Value,
    message::{ToolDesc, ToolDescBuilder},
    tool::{ToolFactory, ToolFunc},
};

pub async fn build_web_search_tool() -> anyhow::Result<ToolFactory> {
    Ok(ToolFactory::simple(
        make_web_search_tool_desc(),
        make_web_search_func(),
    ))
}

fn make_web_search_tool_desc() -> ToolDesc {
    ToolDescBuilder::new("web_search")
        .description(
            "Search the web using multiple search engines simultaneously. \
             Returns aggregated and deduplicated results ranked by how many \
             engines returned them. Use this tool to find current information, \
             facts, documentation, news, or any web-accessible content.",
        )
        .parameters(crate::to_value!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query string"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return. Default: 10. Max: 30.",
                    "default": 10
                }
            },
            "required": ["query"]
        }))
        .build()
}

fn make_web_search_func() -> ToolFunc {
    let searcher = Arc::new(MetaSearcher::new());

    ToolFunc::new(move |args: Value| {
        let searcher = searcher.clone();
        async move {
            let query = args
                .pointer("/query")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();

            let max_results = args
                .pointer("/max_results")
                .and_then(|v| v.as_unsigned())
                .unwrap_or(10)
                .min(30) as usize;

            if query.is_empty() {
                return crate::to_value!({ "results": [], "total": 0 });
            }

            let results = searcher.search(&query, max_results).await;
            let total = results.len();

            let result_values: Vec<Value> = results
                .into_iter()
                .map(|r| {
                    crate::to_value!({
                        "title": r.title.as_str(),
                        "url": r.url.as_str(),
                        "description": r.description.as_str(),
                        "sources": Value::array(r.sources.into_iter().map(Value::string))
                    })
                })
                .collect();

            crate::to_value!({
                "results": Value::array(result_values),
                "total": total as u64
            })
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tool_descriptor() {
        let desc = make_web_search_tool_desc();
        assert_eq!(desc.name, "web_search");
        assert!(desc.description.is_some());
    }

    #[test]
    fn test_tool_schema_requires_query() {
        let desc = make_web_search_tool_desc();
        let required = desc
            .parameters
            .pointer("/required")
            .and_then(|v| v.as_array());
        assert!(required.is_some());
        let required_fields: Vec<&str> = required
            .unwrap()
            .iter()
            .filter_map(|v| v.as_str())
            .collect();
        assert!(required_fields.contains(&"query"));
    }

    /// Verifies the full builtin tool pipeline end-to-end:
    /// - MetaSearcher fans out to all engines and aggregates results.
    /// - At least one result related to the brekkylab/ailoy project appears.
    /// - Results from multiple distinct engines are present (aggregation works).
    /// - Every result has the required fields in the correct format.
    #[tokio::test]
    #[ignore = "requires network"]
    async fn test_web_search_tool_aggregation() {
        // Call the underlying searcher directly so we can inspect AggregatedResult fields.
        let searcher = MetaSearcher::new();
        let results = searcher.search("ailoy", 20).await;

        println!("Total aggregated results: {}", results.len());
        for r in &results {
            println!(
                "[{}] \"{}\" — {} (sources: {:?})",
                r.relevance, r.title, r.url, r.sources
            );
        }

        // 1. There must be at least one result.
        assert!(
            !results.is_empty(),
            "Expected at least one aggregated result"
        );

        // 2. Every result must have a non-empty title and an http(s) URL.
        for r in &results {
            assert!(!r.title.is_empty(), "title must not be empty: {:?}", r);
            assert!(
                r.url.starts_with("http"),
                "url must start with http: {:?}",
                r
            );
            assert!(!r.sources.is_empty(), "sources must not be empty: {:?}", r);
        }

        // 3. At least one result should be clearly related to the brekkylab/ailoy project.
        let ailoy_project_urls = [
            "github.com/brekkylab/ailoy",
            "brekkylab.github.io/ailoy",
            "medium.com/ailoy",
            "webui.ailoy.co",
            "pypi.org/project/ailoy",
        ];
        let has_project_result = results.iter().any(|r| {
            let url_lower = r.url.to_lowercase();
            ailoy_project_urls
                .iter()
                .any(|needle| url_lower.contains(needle))
        });
        assert!(
            has_project_result,
            "Expected at least one result related to the brekkylab/ailoy project, got: {:#?}",
            results.iter().map(|r| &r.url).collect::<Vec<_>>()
        );

        // 4. Results that appeared in more than one engine should be ranked first.
        // (Relevance is monotonically non-increasing after sort in MetaSearcher.)
        for window in results.windows(2) {
            assert!(
                window[0].relevance >= window[1].relevance,
                "Results must be sorted by descending relevance: {} < {}",
                window[0].relevance,
                window[1].relevance
            );
        }

        // 5. Aggregation is working: collect all engine names that contributed.
        let all_sources: std::collections::HashSet<&str> = results
            .iter()
            .flat_map(|r| r.sources.iter().copied())
            .collect();
        println!("Engines that returned results: {:?}", all_sources);
        assert!(
            all_sources.len() >= 3,
            "Expected results from at least 3 distinct engines, got {:?}",
            all_sources
        );
    }
}
