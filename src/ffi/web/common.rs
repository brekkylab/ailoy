use core::ops::DerefMut;
use std::{cell::RefCell, rc::Rc};

use futures::stream::Stream;
use js_sys::{Boolean, Object, Promise, Reflect};
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::future_to_promise;

struct RefCellStreamFuture<S>(Rc<RefCell<S>>);

impl<S> Clone for RefCellStreamFuture<S> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<S> RefCellStreamFuture<S> {
    fn new(stream: S) -> Self {
        Self(Rc::new(RefCell::new(stream)))
    }
}

impl<S> Future for RefCellStreamFuture<S>
where
    S: Stream + Unpin,
{
    type Output = Option<S::Item>;
    fn poll(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        std::pin::Pin::new(self.0.borrow_mut().deref_mut()).poll_next(cx)
    }
}

pub fn stream_to_async_iterable<S>(stream: S) -> Object
where
    S: Stream<Item = Result<JsValue, JsValue>> + Unpin + 'static,
{
    let next = RefCellStreamFuture::new(stream);
    let closure = Closure::<dyn FnMut() -> Promise>::new(move || {
        let cloned = next.clone();
        future_to_promise(async move {
            match cloned.await.transpose() {
                Ok(value) => {
                    let result = Object::new();
                    Reflect::set(&result, &"done".into(), &Boolean::from(value.is_none())).unwrap();
                    if let Some(value) = value {
                        Reflect::set(&result, &"value".into(), &value).unwrap();
                    }
                    Ok(result.into())
                }
                Err(err) => Err(err),
            }
        })
    });
    let iterator = Object::new();
    Reflect::set(&iterator, &"next".into(), &closure.into_js_value()).unwrap();

    // Create AsyncIterable with [Symbol.asyncIterator]
    let out = Object::new();
    let symbol_iterator = js_sys::Symbol::async_iterator();
    let iterator_rc = Rc::new(iterator);
    let iterator_fn = Closure::<dyn Fn() -> JsValue>::new(move || (*iterator_rc).clone().into());
    Reflect::set(&out, &symbol_iterator, &iterator_fn.into_js_value()).unwrap();

    out
}
