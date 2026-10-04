//! What the vendor models share: transport, framing, and wire helpers.

pub(crate) mod chat;
mod framing;
mod http;
mod message;
mod schema;

pub(crate) use framing::{eventstream_drain, eventstream_flush};
pub(crate) use http::*;
pub(crate) use message::*;
pub(crate) use schema::*;
