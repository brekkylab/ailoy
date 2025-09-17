use crate::value::Value;

pub trait Marshal<T>: Default {
    fn marshal(&mut self, item: &T) -> Value;
}

pub trait Unmarshal<T>: Default {
    fn unmarshal(&mut self, val: Value) -> T;
}

// pub trait EncodeValue<T>: Default {
//     fn encode(&mut self, item: &T) -> Value;
// }

// pub trait DecodeValue<T>: Default {
//     fn decode(&mut self, prim: &Value) -> Result<T, ValueError>;
// }

// pub struct Encodable<'data, Data, Encoder: EncodeValue<Data>> {
//     data: &'data Data,
//     encoder: std::marker::PhantomData<Encoder>,
// }

// impl<'data, Data, Encoder: EncodeValue<Data>> Encodable<'data, Data, Encoder> {
//     pub fn new(data: &'data Data) -> Self {
//         Self {
//             data,
//             encoder: std::marker::PhantomData::default(),
//         }
//     }
// }

// impl<'data, Data, Encoder: EncodeValue<Data>> Serialize for Encodable<'data, Data, Encoder> {
//     fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
//     where
//         S: serde::Serializer,
//     {
//         let mut encoder = Encoder::default();
//         let prim = encoder.encode(self.data);
//         prim.serialize(serializer)
//     }
// }

// pub struct Decodable<Data, Decoder: DecodeValue<Data>> {
//     data: Data,
//     decoder: std::marker::PhantomData<Decoder>,
// }

// impl<Data, Decoder: DecodeValue<Data>> Decodable<Data, Decoder> {
//     pub fn to_data(self) -> Data {
//         self.data
//     }
// }

// impl<'de, Data, Decoder: DecodeValue<Data>> Deserialize<'de> for Decodable<Data, Decoder> {
//     fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
//     where
//         D: serde::Deserializer<'de>,
//     {
//         let prim = Value::deserialize(deserializer)?;
//         let mut decoder = Decoder::default();
//         let data = decoder
//             .decode(&prim)
//             .map_err(|_| serde::de::Error::custom("Unable to decode"))?;
//         Ok(Decodable {
//             data,
//             decoder: std::marker::PhantomData::default(),
//         })
//     }
// }
