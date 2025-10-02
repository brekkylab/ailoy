use std::marker::PhantomData;

use serde::{Deserialize, Serialize};

use crate::value::{Delta, Value};

pub trait Marshal<T>: Default {
    fn marshal(&mut self, item: &T) -> Value;
}

pub trait Unmarshal<T: Delta>: Default {
    fn unmarshal(&mut self, val: Value) -> anyhow::Result<T>;
}

pub struct Marshaled<'d, D, M: Marshal<D>> {
    data: &'d D,
    m: PhantomData<M>,
}

impl<'d, D, M: Marshal<D>> Marshaled<'d, D, M> {
    pub fn new(data: &'d D) -> Self {
        Self {
            data,
            m: PhantomData::default(),
        }
    }
}

impl<'d, D, M: Marshal<D>> Serialize for Marshaled<'d, D, M> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut m = M::default();
        let v = m.marshal(self.data);
        v.serialize(serializer)
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
            u: std::marker::PhantomData::default(),
        })
    }
}
