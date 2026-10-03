//! Token arithmetic the UI shows: context in use, totals, cost.

use ailoy::message::TokenUsage;

use crate::{
    catalog::CatalogModel,
    types::{ModelCost, SessionUsage},
};

/// The input the *next* call will carry, approximated by the last call's whole input:
/// Anthropic's `input_tokens` excludes what was read from cache, so all three are summed.
pub fn context_used(u: &TokenUsage) -> u64 {
    u.input_tokens
        + u.cache_read_input_tokens.unwrap_or(0)
        + u.cache_creation_input_tokens.unwrap_or(0)
}

pub fn estimate_cost_usd(usages: &[TokenUsage], cost: &ModelCost) -> Option<f64> {
    let (input, output) = (cost.input?, cost.output?);
    let mut total = 0.0;
    for u in usages {
        total += u.input_tokens as f64 * input;
        total += u.output_tokens as f64 * output;
        total += u.cache_read_input_tokens.unwrap_or(0) as f64 * cost.cache_read.unwrap_or(input);
        total +=
            u.cache_creation_input_tokens.unwrap_or(0) as f64 * cost.cache_write.unwrap_or(input);
    }
    Some(total / 1_000_000.0)
}

pub fn session_usage(usages: &[TokenUsage], model: Option<&CatalogModel>) -> SessionUsage {
    SessionUsage {
        input_tokens: usages.iter().map(|u| u.input_tokens).sum(),
        output_tokens: usages.iter().map(|u| u.output_tokens).sum(),
        cache_read_tokens: usages
            .iter()
            .map(|u| u.cache_read_input_tokens.unwrap_or(0))
            .sum(),
        cache_write_tokens: usages
            .iter()
            .map(|u| u.cache_creation_input_tokens.unwrap_or(0))
            .sum(),
        estimated_cost_usd: model
            .and_then(|m| m.cost.as_ref())
            .and_then(|c| estimate_cost_usd(usages, c)),
        context_used: usages.last().map(context_used),
        context_limit: model.and_then(|m| m.context),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn u(i: u64, o: u64, cr: Option<u64>, cw: Option<u64>) -> TokenUsage {
        TokenUsage {
            input_tokens: i,
            output_tokens: o,
            cache_read_input_tokens: cr,
            cache_creation_input_tokens: cw,
        }
    }

    #[test]
    fn context_used_sums_cached_input() {
        assert_eq!(
            context_used(&u(50, 10, Some(200_000), Some(1_000))),
            201_050
        );
    }

    #[test]
    fn cost_uses_cache_prices_when_present() {
        let cost = ModelCost {
            input: Some(5.0),
            output: Some(25.0),
            cache_read: Some(0.5),
            cache_write: Some(6.25),
        };
        let usd = estimate_cost_usd(
            &[u(1_000_000, 1_000_000, Some(1_000_000), Some(1_000_000))],
            &cost,
        )
        .unwrap();
        assert!((usd - (5.0 + 25.0 + 0.5 + 6.25)).abs() < 1e-9);
        assert!(
            estimate_cost_usd(
                &[u(1, 1, None, None)],
                &ModelCost {
                    input: None,
                    output: Some(1.0),
                    cache_read: None,
                    cache_write: None
                }
            )
            .is_none()
        );
    }

    #[test]
    fn session_usage_reports_last_context_and_limit() {
        let model = CatalogModel {
            id: "m".into(),
            name: "m".into(),
            reasoning: true,
            tool_call: true,
            context: Some(1_000_000),
            output: None,
            cost: None,
        };
        let s = session_usage(
            &[u(10, 5, None, None), u(30, 5, Some(70), None)],
            Some(&model),
        );
        assert_eq!(s.input_tokens, 40);
        assert_eq!(s.cache_read_tokens, 70);
        assert_eq!(s.context_used, Some(100));
        assert_eq!(s.context_limit, Some(1_000_000));
        assert!(s.estimated_cost_usd.is_none());
    }
}
