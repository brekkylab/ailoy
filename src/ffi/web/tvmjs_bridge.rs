use anyhow::{Result, anyhow};
use js_sys::Reflect;
use wasm_bindgen::prelude::*;

#[derive(Clone)]
pub struct PackedFunc {
    func: js_sys::Function,
}

impl PackedFunc {
    /// Create a new JsPackedFunc from a JavaScript function
    pub fn new(func: js_sys::Function) -> Self {
        Self { func }
    }

    /// Call with no arguments, returns promise
    pub fn call0(&self) -> Result<JsValue> {
        self.func
            .call0(&JsValue::NULL)
            .map_err(|e| anyhow!("PackedFunc call0 failed: {:?}", e))
    }

    /// Call with one argument, returns promise
    pub fn call1(&self, arg: &JsValue) -> Result<JsValue> {
        self.func
            .call1(&JsValue::NULL, arg)
            .map_err(|e| anyhow!("PackedFunc call1 failed: {:?}", e))
    }

    /// Call with two arguments, returns promise
    pub fn call2(&self, arg1: &JsValue, arg2: &JsValue) -> Result<JsValue> {
        self.func
            .call2(&JsValue::NULL, arg1, arg2)
            .map_err(|e| anyhow!("PackedFunc call2 failed: {:?}", e))
    }

    /// Call with three arguments, returns promise
    pub fn call3(&self, arg1: &JsValue, arg2: &JsValue, arg3: &JsValue) -> Result<JsValue> {
        self.func
            .call3(&JsValue::NULL, arg1, arg2, arg3)
            .map_err(|e| anyhow!("PackedFunc call3 failed: {:?}", e))
    }

    /// Call with four arguments, returns promise
    pub fn call4(
        &self,
        arg1: &JsValue,
        arg2: &JsValue,
        arg3: &JsValue,
        arg4: &JsValue,
    ) -> Result<JsValue> {
        self.func
            .call4(&JsValue::NULL, arg1, arg2, arg3, arg4)
            .map_err(|e| anyhow!("PackedFunc call4 failed: {:?}", e))
    }

    /// Call with five arguments, returns promise
    pub fn call5(
        &self,
        arg1: &JsValue,
        arg2: &JsValue,
        arg3: &JsValue,
        arg4: &JsValue,
        arg5: &JsValue,
    ) -> Result<JsValue> {
        self.func
            .call5(&JsValue::NULL, arg1, arg2, arg3, arg4, arg5)
            .map_err(|e| anyhow!("PackedFunc call5 failed: {:?}", e))
    }

    /// Get the underlying JavaScript function
    pub fn as_js_function(&self) -> &js_sys::Function {
        &self.func
    }

    pub fn dispose(&self) -> Result<()> {
        let dispose: js_sys::Function = Reflect::get(&self.func, &JsValue::from_str("dispose"))
            .map_err(|e| anyhow!("PackedFunc does not have .dispose(): {:?}", e))?
            .into();
        dispose
            .call0(&JsValue::NULL)
            .map_err(|e| anyhow!("Failed to dispose PackedFunc: {:?}", e))?;
        Ok(())
    }
}

impl From<js_sys::Function> for PackedFunc {
    fn from(value: js_sys::Function) -> Self {
        Self::new(value)
    }
}

impl From<JsValue> for PackedFunc {
    fn from(value: JsValue) -> Self {
        let func: js_sys::Function = value.into();
        Self::new(func)
    }
}

#[wasm_bindgen(raw_module = "./shim_js/dist/index.js")]
extern "C" {
    //////////////
    /// Scalar ///
    //////////////
    #[wasm_bindgen(js_name = "Scalar")]
    pub type Scalar;

    #[wasm_bindgen(constructor)]
    pub fn new(value: f64, dtype: &str) -> Scalar;

    ////////////////
    /// DLDevice ///
    ////////////////
    #[wasm_bindgen(js_name = "DLDevice")]
    pub type DLDevice;

    #[wasm_bindgen(method, js_name = sync)]
    pub async fn sync(this: &DLDevice);

    #[wasm_bindgen(method, js_name = toString)]
    pub fn to_string(this: &DLDevice) -> String;

    /////////////////
    /// TVMObject ///
    /////////////////
    #[wasm_bindgen(js_name = "TVMObject")]
    pub type TVMObject;

    #[wasm_bindgen(method, js_name = dispose)]
    pub fn dispose(this: &TVMObject);

    //////////////
    /// Tensor ///
    //////////////
    #[wasm_bindgen(js_name = "Tensor", extends = TVMObject)]
    pub type Tensor;

    #[wasm_bindgen(method, js_name = view)]
    pub fn view(this: &Tensor, shape: js_sys::Array) -> Tensor;

    #[wasm_bindgen(method, js_name = copyFrom)]
    pub fn copy_from(this: &Tensor, data: &Tensor) -> Tensor;

    ////////////////
    /// TVMArray ///
    ////////////////
    #[wasm_bindgen(js_name = "TVMArray", extends = TVMObject)]
    pub type TVMArray;

    #[wasm_bindgen(method, js_name = size)]
    pub fn size(this: &TVMArray) -> i64;

    #[wasm_bindgen(method, js_name = get)]
    pub fn get(this: &TVMArray, index: u32) -> TVMObject;

    //////////////
    /// Module ///
    //////////////
    #[wasm_bindgen(js_name = "Module", extends = TVMObject)]
    pub type Module;

    #[wasm_bindgen(method, js_name = getFunction)]
    pub fn get_function(this: &Module, name: &str) -> js_sys::Function;

    ////////////////
    /// Instance ///
    ////////////////
    #[wasm_bindgen(js_name = "Instance")]
    pub type Instance;

    #[wasm_bindgen(method, js_name = beginScope)]
    pub fn begin_scope(this: &Instance);

    #[wasm_bindgen(method, js_name = endScope)]
    pub fn end_scope(this: &Instance);

    #[wasm_bindgen(method, js_name = attachToCurrentScope)]
    pub fn attach_to_current_scope(this: &Instance, obj: TVMObject) -> TVMObject;

    #[wasm_bindgen(method, js_name = detachFromCurrentScope)]
    pub fn detach_from_current_scope(this: &Instance, obj: TVMObject) -> TVMObject;

    #[wasm_bindgen(method, js_name = systemLib)]
    pub fn system_lib(this: &Instance) -> Module;

    #[wasm_bindgen(method, js_name = createVirtualMachine)]
    pub fn create_virtual_machine(this: &Instance, device: DLDevice) -> Module;

    #[wasm_bindgen(method, js_name = getGlobalFunc)]
    pub fn get_global_func(this: &Instance, name: &str) -> js_sys::Function;

    #[wasm_bindgen(method, js_name = cpu)]
    pub fn cpu(this: &Instance) -> DLDevice;

    #[wasm_bindgen(method, js_name = webgpu)]
    pub fn webgpu(this: &Instance, device_id: u32) -> DLDevice;

    #[wasm_bindgen(method, js_name = empty)]
    pub fn empty(this: &Instance, shape: js_sys::Array, dtype: &str, dev: DLDevice) -> Tensor;

    #[wasm_bindgen(method, js_name = makeTVMArray)]
    pub fn make_tvm_array(this: &Instance, inputs: js_sys::Array) -> TVMArray;

    #[wasm_bindgen(method, js_name = makeShapeTuple)]
    pub fn make_shape_tuple(this: &Instance, shape: js_sys::Array) -> TVMObject;

    #[wasm_bindgen(method, js_name = initWebGPU)]
    pub fn init_webgpu(this: &Instance, device: DLDevice);

    #[wasm_bindgen(method, js_name = getParamsFromCacheByName)]
    pub fn get_params_from_cache_by_name(this: &Instance, param_names: js_sys::Array) -> TVMObject;

    #[wasm_bindgen(method, js_name = tensorCacheUpdate)]
    pub fn tensor_cache_update(this: &Instance, name: &str, arr: Tensor);

    #[wasm_bindgen(method, js_name = tensorCacheClear)]
    pub fn tensor_cache_clear(this: &Instance);

    #[wasm_bindgen(method, js_name = sampleTopPFromLogits)]
    pub fn sample_top_p_from_logits(
        this: &Instance,
        logits: Tensor,
        temperature: f64,
        top_p: f64,
    ) -> i64;

    #[wasm_bindgen(js_name = instantiate)]
    pub async fn instantiate(buffer_source: js_sys::ArrayBuffer) -> Instance;
}

impl Instance {
    pub fn detach<T: From<JsValue>>(&self, obj: impl Into<JsValue>) -> T {
        let detached = self.detach_from_current_scope(obj.into().into());
        T::from(detached.into())
    }
}
