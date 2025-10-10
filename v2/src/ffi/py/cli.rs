use pyo3::prelude::*;
use pyo3_stub_gen_derive::*;

use crate::ffi::py::base::await_future;

#[gen_stub_pyfunction]
#[pyfunction]
pub fn ailoy_model_cli() -> PyResult<()> {
    // When running the CLI via Python, we need to shift the args.
    let mut args: Vec<String> = std::env::args().collect();
    args.remove(0);

    await_future(crate::cli::ailoy_model::ailoy_model_cli(args))
}
