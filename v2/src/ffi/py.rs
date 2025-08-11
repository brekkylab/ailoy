use std::sync::Arc;

use futures::StreamExt;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyString};
// use pyo3_asyncio::tokio::future_into_py;
use tokio::sync::Mutex;

use crate::agent::Agent;

#[pyclass(name = "Agent")]
pub struct PyAgent {
    inner: Agent,
}

#[pymethods]
impl PyAgent {
    //     /// Python에서의 생성자.
    //     ///
    //     /// Rust 쪽에서 AnyLanguageModel/AnyTool을 어떻게 넘길지에 따라
    //     /// 아래 시그니처를 바꾸셔도 됩니다.
    //     /// 여기서는 간편하게 Rust 쪽에서 이미 만든 Agent를 넘기는
    //     /// `from_raw` 스타일을 예시로 둡니다.
    //     #[new]
    //     fn new() -> PyResult<Self> {
    //         Err(pyo3::exceptions::PyTypeError::new_err(
    //             "Use AgentBuilder or `create_local_agent(...)` factory to construct.",
    //         ))
    //     }

    //     /// Rust에서 이미 구성된 Agent를 래핑하기 위해 사용되는 헬퍼 (Python에 직접 노출하지 않음)
    //     #[allow(dead_code)]
    //     pub(crate) fn from_agent(agent: Agent) -> Self {
    //         Self {
    //             inner: Arc::new(Mutex::new(agent)),
    //         }
    //     }

    //     /// Python에서 호출: 한 번의 사용자 메시지를 처리.
    //     ///
    //     /// - `user_message`: 유저 입력
    //     /// - `on_delta`: 선택적 콜백 (파이썬 callable). 각 델타마다 문자열(debug)로 호출합니다.
    //     ///
    //     /// 반환값: `await` 가능한 코루틴. 완료되면 최종 assistant 메시지를 문자열(debug)로 돌려줍니다.
    //     ///
    //     /// 참고: 스트리밍을 Python async generator로 구현할 수도 있으나,
    //     /// 콜백 방식이 간단하고 신뢰성이 높습니다.
    //     fn run_once<'py>(
    //         &self,
    //         py: Python<'py>,
    //         user_message: &str,
    //         on_delta: Option<PyObject>,
    //     ) -> PyResult<&'py PyAny> {
    //         let agent = self.inner.clone();
    //         // Python awaitable로 감싼다.
    //         future_into_py(py, async move {
    //             let mut guard = agent.lock().await;
    //             let mut strm = Box::pin(guard.run(user_message.to_owned()));

    //             // 파이썬 콜백이 있으면, 각 델타마다 호출
    //             let mut agg = MessageAggregator::new();
    //             while let Some(next) = strm.next().await {
    //                 match next {
    //                     Ok(delta) => {
    //                         // 콜백 호출 (Debug 문자열로)
    //                         if let Some(cb) = &on_delta {
    //                             let s = format!("{delta:?}");
    //                             Python::with_gil(|py| {
    //                                 // cb(s) 호출
    //                                 if let Err(e) = cb.call1(py, (s,)) {
    //                                     e.print(py); // 콜백 에러는 로그만 찍고 계속
    //                                 }
    //                             });
    //                         }
    //                         // 내부 aggregator에도 반영 (필요시)
    //                         if let Some(_msg) = agg.update(delta) {
    //                             // 필요하면 여기서 중간 메시지를 써도 됨
    //                         }
    //                     }
    //                     Err(e) => {
    //                         return Err(pyo3::exceptions::PyRuntimeError::new_err(e));
    //                     }
    //                 }
    //             }

    //             // 최종 메시지 (옵션)
    //             let final_msg_str = if let Some(msg) = agg.finalize() {
    //                 format!("{msg:?}")
    //             } else {
    //                 String::from("")
    //             };

    //             Ok(Python::with_gil(|py| {
    //                 PyString::new(py, &final_msg_str).into_py(py)
    //             }))
    //         })
    //     }

    //     /// 대화 로그를 Python dict 리스트로 받고 싶을 때 사용 가능한 헬퍼.
    //     /// (간단 버전: Debug 문자열 배열) — 필요 없으면 제거하셔도 됩니다.
    //     fn dump_messages<'py>(&self, py: Python<'py>) -> PyResult<&'py PyAny> {
    //         let inner = self.inner.clone();
    //         future_into_py(py, async move {
    //             let msgs = inner.lock().await.messages.lock().await.clone();
    //             // 간단히 debug 문자열 리스트로 변환
    //             Python::with_gil(|py| {
    //                 let list = pyo3::types::PyList::empty(py);
    //                 for m in msgs {
    //                     list.append(PyString::new(py, &format!("{m:?}"))).unwrap();
    //                 }
    //                 Ok(list.into_py(py))
    //             })
    //         })
    //     }
}

// /// 편의: Local 모델 키만으로 Agent를 만들어 Python에 돌려주는 팩토리.
// /// 사용하지 않으면 삭제해도 됩니다.
// #[pyfunction]
// fn create_local_agent(py: Python<'_>, model_key: &str) -> PyResult<&PyAny> {
//     // 필요 시 내부 Cache/LocalLanguageModel을 여기서 생성
//     // (아래는 사용자 코드의 의존을 가정)
//     use crate::model::LocalLanguageModel;
//     let model_key = model_key.to_string();

//     future_into_py(py, async move {
//         let cache = crate::cache::Cache::new();
//         let model = cache
//             .try_create::<LocalLanguageModel>(&model_key)
//             .await
//             .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
//         let agent = Agent::new(model.into(), Vec::new());
//         // PyAgent로 래핑해서 반환
//         Python::with_gil(|py| {
//             let py_agent = PyAgent::from_agent(agent);
//             Py::new(py, py_agent).map(|obj| obj.into_py(py))
//         })
//     })
// }

// /// 파이썬 모듈 초기화
// #[pymodule]
// fn your_crate(py: Python<'_>, m: &PyModule) -> PyResult<()> {
//     // pyo3-asyncio: Tokio 런타임 초기화 (multi-thread)
//     pyo3_asyncio::tokio::init_multi_thread_once();

//     m.add_class::<PyAgent>()?;
//     m.add_function(wrap_pyfunction!(create_local_agent, m)?)?;
//     // 필요 시 다른 함수/클래스도 추가

//     // 간단한 버전정보
//     let info = PyDict::new_bound(py);
//     info.set_item("backend", "tokio")?;
//     info.set_item("streaming", "callback")?;
//     m.add("info", info)?;

//     Ok(())
// }
