use std::marker::PhantomData;

use serde::{Deserialize, Serialize};

use crate::{datatype::Value, message::Delta};

pub trait Marshal<T: ?Sized>: Default {
    fn marshal(&mut self, item: &T) -> Value;
}

pub trait Unmarshal<T: Delta>: Default {
    /// Parse a whole (non-streaming) response into `T`.
    fn unmarshal(&mut self, val: Value) -> anyhow::Result<T>;

    /// Parse one streaming (SSE) event's `data:` payload into an incremental
    /// delta. Stream events differ in shape from the whole response, so this is
    /// a separate parser; control events with no delta (OpenAI `[DONE]`,
    /// Anthropic `ping`) return `Ok(None)`. Defaults to unsupported; override to
    /// opt into streaming. `&mut self` matches [`unmarshal`](Self::unmarshal) and
    /// leaves room to carry aggregation state across a stream without `Arc`/
    /// `Mutex`; current impls are stateless, so it's unused but kept for that.
    fn unmarshal_event(&mut self, _data: &str) -> anyhow::Result<Option<T>> {
        Err(anyhow::anyhow!(
            "streaming (SSE) is not supported for this provider"
        ))
    }
}

impl<T, M: Marshal<T>> Marshal<[T]> for M {
    fn marshal(&mut self, item: &[T]) -> Value {
        Value::Array(item.iter().map(|elem| self.marshal(elem)).collect())
    }
}

impl<T, M: Marshal<T>> Marshal<Vec<T>> for M {
    fn marshal(&mut self, item: &Vec<T>) -> Value {
        Value::Array(item.iter().map(|elem| self.marshal(elem)).collect())
    }
}

pub struct Marshaled<'d, D: ?Sized, M: Marshal<D>> {
    data: &'d D,
    m: PhantomData<M>,
}

impl<'d, D: ?Sized, M: Marshal<D>> Marshaled<'d, D, M> {
    pub fn new(data: &'d D) -> Self {
        Self {
            data,
            m: PhantomData,
        }
    }
}

impl<'d, D: ?Sized, M: Marshal<D>> Serialize for Marshaled<'d, D, M> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut m = M::default();
        let v = m.marshal(self.data);
        v.serialize(serializer)
    }
}

impl<'d, D: ?Sized, M: Marshal<D>> From<Marshaled<'d, D, M>> for Value {
    fn from(marshaled: Marshaled<'d, D, M>) -> Self {
        M::default().marshal(marshaled.data)
    }
}

pub struct Unmarshaled<D: Delta, U: Unmarshal<D>> {
    data: D,
    u: std::marker::PhantomData<U>,
}

impl<D: Delta, U: Unmarshal<D>> Unmarshaled<D, U> {
    pub fn get(self) -> D {
        self.data
    }
}

impl<'de, D: Delta, U: Unmarshal<D>> Deserialize<'de> for Unmarshaled<D, U> {
    fn deserialize<De>(deserializer: De) -> Result<Self, De::Error>
    where
        De: serde::Deserializer<'de>,
    {
        let v = Value::deserialize(deserializer)?;
        let mut u = U::default();
        let delta = u
            .unmarshal(v)
            .map_err(|_| serde::de::Error::custom("Unable to decode"))?;
        Ok(Unmarshaled {
            data: delta,
            u: std::marker::PhantomData,
        })
    }
}
