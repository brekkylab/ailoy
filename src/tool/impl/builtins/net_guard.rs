//! Egress guard for host-side tools.
//!
//! Tools registered as *pure* run in the ailoy host process rather than inside
//! the sandbox VM, so the guest network policy never sees their requests. A
//! model — or a prompt-injected page a model reads — can otherwise aim such a
//! tool at loopback, the LAN, or the cloud-metadata address and read the answer
//! back into the conversation. This module decides whether a destination is on
//! the public internet.
//!
//! There are two entry points because a URL names its destination in two ways:
//! [`check_host`] for a host written as an IP literal, where the address is
//! known before connecting, and [`PublicOnlyResolver`] for a host written as a
//! name, where the addresses are known only once DNS answers.

use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};

use wreq::dns::{Addrs, GaiResolver, Name, Resolve, Resolving};

/// A destination this module refused.
///
/// A concrete type rather than a string because a refusal raised inside the
/// resolver has to be recognized again after the connector has wrapped it,
/// which [`blocked_reason`] does by downcast. Matching on the message text
/// would work today and break the moment someone rewords it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Blocked(String);

impl std::fmt::Display for Blocked {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "blocked: {}", self.0)
    }
}

impl std::error::Error for Blocked {}

/// Find a [`Blocked`] in an error's source chain.
///
/// `Display` on a client error stops one level in, while a refusal from
/// [`PublicOnlyResolver`] sits three deep — under the connect error and the
/// DNS error the connector wraps it in. Formatting the outer error therefore
/// reports a plain connection failure, which reads as a target that happens to
/// be down rather than one that will never be reachable. A model told the first
/// retries; told the second it stops. Walking the chain is what keeps a name
/// refused by the resolver reporting the same way as an IP literal refused
/// before the connection.
pub fn blocked_reason(err: &(dyn std::error::Error + 'static)) -> Option<String> {
    let mut source = Some(err);
    while let Some(e) = source {
        if let Some(blocked) = e.downcast_ref::<Blocked>() {
            return Some(blocked.to_string());
        }
        source = e.source();
    }
    None
}

/// Reject a host that is written as an IP literal pointing anywhere other than
/// the public internet. A host written as a name returns `Ok` here and is
/// gated later by [`PublicOnlyResolver`], which is the only place its actual
/// addresses are known.
///
/// Strips the brackets an IPv6 literal is written with. Both callers hand them
/// over — `url::Url::host_str` and `http::Uri::host` each return `[::1]` rather
/// than `::1` — and `IpAddr` parses neither spelling with them attached.
pub fn check_host(host: &str) -> Result<(), Blocked> {
    let bare = host
        .strip_prefix('[')
        .and_then(|h| h.strip_suffix(']'))
        .unwrap_or(host);
    match bare.parse::<IpAddr>() {
        Ok(ip) if !is_public(ip) => Err(Blocked(format!("{ip} is not a public address"))),
        _ => Ok(()),
    }
}

/// Reject a redirect hop by scheme and host. Separate from [`check_host`] so
/// the per-hop rule — including the scheme, which a `Location` header controls
/// just as freely as the host — is one testable function.
pub fn check_redirect_target(scheme: Option<&str>, host: Option<&str>) -> Result<(), Blocked> {
    match scheme {
        Some("http") | Some("https") => {}
        Some(other) => return Err(Blocked(format!("unsupported redirect scheme: {other}"))),
        None => return Err(Blocked("redirect target has no scheme".to_string())),
    }
    match host {
        Some(h) => check_host(h),
        None => Err(Blocked("redirect target has no host".to_string())),
    }
}

/// Keep only the globally routable addresses in `addrs`. Returns `Err` when
/// nothing survives, which the connector buries under two layers of its own
/// error; [`blocked_reason`] is what digs it back out so the caller reports a
/// blocked host rather than an opaque connection failure.
fn filter_public(
    host: &str,
    addrs: impl Iterator<Item = SocketAddr>,
) -> Result<Vec<SocketAddr>, Blocked> {
    let (kept, dropped): (Vec<_>, Vec<_>) = addrs.partition(|addr| is_public(addr.ip()));
    if !dropped.is_empty() {
        let ips: Vec<IpAddr> = dropped.iter().map(|addr| addr.ip()).collect();
        log::warn!("net_guard: dropped non-public address(es) for '{host}': {ips:?}");
    }
    if kept.is_empty() {
        return Err(Blocked(format!(
            "'{host}' resolves only to non-public addresses"
        )));
    }
    Ok(kept)
}

/// DNS resolver that drops every answer outside the public internet before the
/// connector can use it.
///
/// Filtering here rather than checking the addresses up front is what closes
/// the DNS-rebinding window: the connector reaches only the addresses this
/// filter returned, so there is no gap between a check and the connect for a
/// second, inward-pointing answer to land in.
///
/// One case stays uncovered: with an HTTP proxy configured, the connector
/// resolves the proxy's host and the proxy resolves the target, so the target's
/// addresses never pass through here. That configuration comes from the host
/// environment rather than from anything a model can set, and [`check_host`]
/// still applies to the URL itself.
#[derive(Clone, Default)]
pub struct PublicOnlyResolver {
    inner: GaiResolver,
}

impl Resolve for PublicOnlyResolver {
    fn resolve(&self, name: Name) -> Resolving {
        let host = name.as_str().to_string();
        let resolving = self.inner.resolve(name);
        Box::pin(async move {
            let addrs = resolving.await?;
            let kept = filter_public(&host, addrs)?;
            Ok(Box::new(kept.into_iter()) as Addrs)
        })
    }
}

/// Is `ip` a globally routable unicast address, i.e. somewhere on the public
/// internet rather than on this host, this network, or in a special-use range?
///
/// Written as a blocklist of the IANA special-purpose ranges because
/// `Ipv4Addr::is_global` is still unstable. Reserved ranges count as
/// non-public, which errs toward refusing a fetch rather than allowing an
/// internal one.
fn is_public(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => is_public_v4(v4),
        IpAddr::V6(v6) => {
            // An IPv4 address embedded in an IPv6 one is still that IPv4
            // address at the far end: `::ffff:127.0.0.1` reaches loopback, and
            // `64:ff9b::127.0.0.1` does too wherever a NAT64 gateway sits in
            // the path. Judge the embedded address, not the wrapper.
            if let Some(v4) = v6.to_ipv4_mapped() {
                return is_public_v4(v4);
            }
            if let Some(v4) = nat64_embedded_v4(v6) {
                return is_public_v4(v4);
            }
            is_public_v6(v6)
        }
    }
}

fn is_public_v4(ip: Ipv4Addr) -> bool {
    // The ranges std already has a predicate for. `is_link_local` is the one
    // that matters most here: 169.254.0.0/16 is where the cloud-metadata
    // address 169.254.169.254 lives.
    if ip.is_unspecified()
        || ip.is_loopback()
        || ip.is_private()
        || ip.is_link_local()
        || ip.is_multicast()
        || ip.is_broadcast()
        || ip.is_documentation()
    {
        return false;
    }
    // The remaining special-purpose ranges, spelled out because their
    // predicates are unstable. In order: 0.0.0.0/8 "this network", 100.64/10
    // carrier-grade NAT, 192.0.0/24 IETF protocol assignments, 192.88.99/24
    // 6to4 relay anycast, 198.18/15 benchmarking, and 240/4 reserved.
    let [a, b, c, _] = ip.octets();
    !(a == 0
        || (a == 100 && (64..128).contains(&b))
        || (a == 192 && b == 0 && c == 0)
        || (a == 192 && b == 88 && c == 99)
        || (a == 198 && (b == 18 || b == 19))
        || a >= 240)
}

fn is_public_v6(ip: Ipv6Addr) -> bool {
    if ip.is_unspecified() || ip.is_loopback() || ip.is_multicast() {
        return false;
    }
    // In order: the whole of ::/64, which also covers the deprecated
    // IPv4-compatible form `::a.b.c.d`; fc00::/7 unique-local; fe80::/10
    // link-local; fec0::/10 deprecated site-local; 2001:db8::/32
    // documentation; 2001::/23 IETF protocol assignments; and 2002::/16 6to4.
    // The last two are worth blocking because Teredo and 6to4 each embed an
    // IPv4 address, so an internal target can ride inside one.
    //
    // 64:ff9b:1::/48 is here for the same embedding reason but is refused
    // outright rather than unwrapped. RFC 8215 reserves it for network-specific
    // NAT64 prefixes of any length up to /96, so the embedded address does not
    // sit at a fixed offset the way it does under the well-known prefix, and
    // there is no address in the range worth reaching anyway.
    //
    // Then three ranges that carry no embedded address and reach no service:
    // 100::/64 discards whatever is sent to it, and 3fff::/20 and 5f00::/16 are
    // reserved for documentation and for SRv6 segment identifiers. None is
    // routable, so none belongs on the allowed side of a predicate that answers
    // "is this on the public internet".
    //
    // 2001:20::/28 (ORCHIDv2) and 2001:30::/28 (DRIP) need no entry of their
    // own — both sit inside 2001::/23 above.
    let s = ip.segments();
    !((s[0] == 0 && s[1] == 0 && s[2] == 0 && s[3] == 0)
        || (s[0] & 0xfe00) == 0xfc00
        || (s[0] & 0xffc0) == 0xfe80
        || (s[0] & 0xffc0) == 0xfec0
        || (s[0] == 0x2001 && s[1] == 0x0db8)
        || (s[0] == 0x2001 && s[1] < 0x0200)
        || s[0] == 0x2002
        || (s[0] == 0x0064 && s[1] == 0xff9b && s[2] == 0x0001)
        || (s[0] == 0x0100 && s[1] == 0 && s[2] == 0 && s[3] == 0)
        || (s[0] == 0x3fff && (s[1] & 0xf000) == 0)
        || s[0] == 0x5f00)
}

/// The IPv4 address embedded in a NAT64 well-known-prefix address
/// (`64:ff9b::/96`, RFC 6052), if this is one.
fn nat64_embedded_v4(ip: Ipv6Addr) -> Option<Ipv4Addr> {
    let s = ip.segments();
    let is_nat64 =
        s[0] == 0x0064 && s[1] == 0xff9b && s[2] == 0 && s[3] == 0 && s[4] == 0 && s[5] == 0;
    is_nat64.then(|| Ipv4Addr::from(((s[6] as u32) << 16) | s[7] as u32))
}

#[cfg(test)]
mod tests {
    use url::Url;

    use super::*;

    /// The addresses an SSRF attempt actually aims at. Every one of these must
    /// be refused before a connection is opened.
    #[test]
    fn check_host_rejects_non_public_literals() {
        for host in [
            "127.0.0.1",
            "0.0.0.0",
            "10.0.0.1",
            "172.16.0.1",
            "192.168.1.1",
            "169.254.169.254", // IMDSv1
            "100.64.0.1",
            "192.0.2.1",
            "198.18.0.1",
            "224.0.0.1",
            "240.0.0.1",
            "255.255.255.255",
            "[::1]",
            "::1",
            "[::ffff:127.0.0.1]",
            "[64:ff9b::7f00:1]", // NAT64-wrapped 127.0.0.1
            "[fd00::1]",
            "[fe80::1]",
            "[fec0::1]",
            "[2001:db8::1]",
            "[2001::1]",        // Teredo
            "[2002:7f00:1::1]", // 6to4-wrapped 127.0.0.1
            "[::7f00:1]",       // IPv4-compatible loopback
            // RFC 8215 network-specific NAT64. The prefix may be anywhere up to
            // /96, so the embedded address moves; the whole /48 is refused
            // rather than unwrapped, which these two spellings pin.
            "[64:ff9b:1::7f00:1]",
            "[64:ff9b:1:ffff::a9fe:a9fe]",
            "[100::1]",           // RFC 6666 discard-only
            "[3fff::1]",          // RFC 9637 documentation
            "[3fff:fff:ffff::1]", // top of that /20
            "[5f00::1]",          // RFC 9602 SRv6 SIDs
            "[2001:20::1]",       // ORCHIDv2, covered by 2001::/23
        ] {
            assert!(
                check_host(host).is_err(),
                "{host} must be refused as non-public"
            );
        }
    }

    /// The addresses just outside the narrower reserved ranges are included so
    /// a mask that is one bit too wide fails here rather than silently refusing
    /// traffic that should have gone out.
    #[test]
    fn check_host_allows_public_literals() {
        for host in [
            "1.1.1.1",
            "8.8.8.8",
            "[2606:4700:4700::1111]",
            "[100:0:0:1::1]", // outside 100::/64
            "[3fff:1000::1]", // outside 3fff::/20
            "[5f01::1]",      // outside 5f00::/16
        ] {
            assert!(check_host(host).is_ok(), "{host} must be allowed");
        }
    }

    /// A name passes this stage even when it is known to resolve inward —
    /// `PublicOnlyResolver` is what refuses it, once the answer is in hand.
    /// Freezing the split here keeps a future edit from moving the check to a
    /// place where a rebinding answer would have a window.
    #[test]
    fn check_host_defers_names_to_the_resolver() {
        for host in ["localhost", "example.com", "metadata.google.internal"] {
            assert!(check_host(host).is_ok(), "{host} must reach the resolver");
        }
    }

    /// Obfuscated spellings of loopback. `std`'s `IpAddr` parser rejects all of
    /// these outright, so the literal check only recognizes them because `url`
    /// normalizes them to a dotted quad first and `fetch_one` checks the parsed
    /// host rather than the raw input. That ordering is what makes them safe.
    #[test]
    fn obfuscated_loopback_urls_are_rejected_after_parsing() {
        for raw in [
            "http://2130706433/", // decimal
            "http://0177.0.0.1/", // octal
            "http://0x7f.0.0.1/", // hex
            "http://127.1/",      // shorthand
            "http://127.0.0.1.:8080/",
        ] {
            let bare = raw.trim_start_matches("http://").trim_end_matches('/');
            assert!(
                bare.parse::<IpAddr>().is_err(),
                "{bare} parses as an address on its own, so this case does not \
                 exercise the normalization path it exists to cover"
            );
            let url = Url::parse(raw).unwrap_or_else(|e| panic!("parse {raw}: {e}"));
            let host = url
                .host_str()
                .unwrap_or_else(|| panic!("{raw} has no host"));
            assert!(
                check_host(host).is_err(),
                "{raw} normalized to host {host}, which must be refused"
            );
        }
    }

    #[test]
    fn redirect_target_requires_http_scheme_and_public_host() {
        assert!(check_redirect_target(Some("https"), Some("example.com")).is_ok());
        assert!(check_redirect_target(Some("http"), Some("1.1.1.1")).is_ok());
        assert!(check_redirect_target(Some("http"), Some("127.0.0.1")).is_err());
        assert!(check_redirect_target(Some("file"), Some("example.com")).is_err());
        assert!(check_redirect_target(None, Some("example.com")).is_err());
        assert!(check_redirect_target(Some("https"), None).is_err());
    }

    /// A split answer — one public address, one internal — must connect only to
    /// the public one rather than being refused outright or, worse, allowed
    /// through wholesale.
    #[test]
    fn filter_public_keeps_public_and_drops_internal() {
        let addrs = vec![
            "1.1.1.1:443".parse::<SocketAddr>().unwrap(),
            "127.0.0.1:443".parse::<SocketAddr>().unwrap(),
            "169.254.169.254:80".parse::<SocketAddr>().unwrap(),
        ];
        let kept = filter_public("split.example", addrs.into_iter()).expect("public addr survives");
        assert_eq!(kept, vec!["1.1.1.1:443".parse::<SocketAddr>().unwrap()]);
    }

    #[test]
    fn filter_public_errors_when_every_answer_is_internal() {
        let addrs = vec![
            "127.0.0.1:80".parse::<SocketAddr>().unwrap(),
            "10.1.2.3:80".parse::<SocketAddr>().unwrap(),
        ];
        let err = filter_public("rebind.example", addrs.into_iter())
            .expect_err("all-internal answer must be refused");
        let text = err.to_string();
        assert!(text.contains("rebind.example"), "error text: {text}");
    }

    /// The recovery `web_fetch` depends on: a refusal wrapped by layers that do
    /// not print their source has to stay findable. Two synthetic wrappers stand
    /// in for the connector's connect-and-DNS pair, so this holds the contract
    /// even if wreq restructures its errors.
    #[test]
    fn blocked_reason_digs_a_refusal_out_of_a_source_chain() {
        #[derive(Debug)]
        struct Opaque(Box<dyn std::error::Error + Send + Sync>);
        impl std::fmt::Display for Opaque {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                // Deliberately drops the source, which is what hides the
                // refusal in the real chain.
                write!(f, "client error (Connect)")
            }
        }
        impl std::error::Error for Opaque {
            fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
                Some(self.0.as_ref())
            }
        }

        let refusal = check_host("127.0.0.1").expect_err("loopback must be refused");
        let wrapped = Opaque(Box::new(Opaque(Box::new(refusal))));
        assert_eq!(
            blocked_reason(&wrapped).as_deref(),
            Some("blocked: 127.0.0.1 is not a public address")
        );
        assert_eq!(wrapped.to_string(), "client error (Connect)");
    }

    /// An unrelated failure must not be reported as a policy refusal — a target
    /// that is genuinely down is worth retrying, and a blocked one is not.
    #[test]
    fn blocked_reason_ignores_an_ordinary_error() {
        let err = std::io::Error::other("connection reset");
        assert_eq!(blocked_reason(&err), None);
    }

    #[test]
    fn nat64_prefix_unwraps_to_its_ipv4() {
        let ip: Ipv6Addr = "64:ff9b::7f00:1".parse().unwrap();
        assert_eq!(nat64_embedded_v4(ip), Some(Ipv4Addr::new(127, 0, 0, 1)));
        let ip: Ipv6Addr = "64:ff9b::808:808".parse().unwrap();
        assert_eq!(nat64_embedded_v4(ip), Some(Ipv4Addr::new(8, 8, 8, 8)));
        // A public address that merely starts with 0064 is not the NAT64 prefix.
        let ip: Ipv6Addr = "64:ff9c::1".parse().unwrap();
        assert_eq!(nat64_embedded_v4(ip), None);
    }

    /// The NAT64 unwrap must not turn a public embedded address into a block.
    /// Only the well-known prefix unwraps, so this stays scoped to it: the
    /// neighbouring `64:ff9b:1::/48` is refused wholesale, and `64:ff9c::/32`
    /// is an ordinary public range that must not be caught by either rule.
    #[test]
    fn nat64_wrapping_a_public_address_stays_allowed() {
        assert!(check_host("[64:ff9b::808:808]").is_ok());
        assert!(check_host("[64:ff9c::1]").is_ok());
    }
}
