use std::path::Path;
use std::{fmt::Debug, collections::HashMap, sync::{Arc, Mutex}};

use ndarray::{ArrayD, Array};
use flowrs::RuntimeConnectable;
use flowrs::{
    connection::{Input, Output},
    node::{Node, UpdateError, ChangeObserver},
};
use serde::{Deserialize, Serialize};

use futures_executor::block_on;
use wonnx::utils::InputTensor;
use wonnx::{
    utils::OutputTensor,
    WonnxError,
    Session
};

#[cfg(target_arch = "wasm32")]
use base64::{Engine as _, engine::general_purpose};
//#[cfg(target_arch = "wasm32")]
use wasm_bindgen_futures::spawn_local;

#[derive(Clone, Deserialize, Serialize, Debug)]
pub struct ModelConfig {
   pub model_path: String,
   pub model_base64: String,
}

#[derive(RuntimeConnectable)]
pub struct ModelNode
{
    #[input]
    pub input_model_config: Input<ModelConfig>,
    #[input]
    pub model_input: Input<ArrayD<f32>>,
    #[output]
    pub output: Output<ArrayD<f32>>,
    pub model_config: Option<ModelConfig>,
    pub session: Arc<Mutex<Option<Session>>>,
}

impl ModelNode
{
    pub fn new(change_observer: Option<&ChangeObserver>) -> Self {
        Self {
            input_model_config: Input::new(),
            model_input: Input::new(),
            output: Output::new(change_observer),
            model_config: None,
            session: Arc::new(Mutex::new(None)),
        }
    }
    
    #[cfg(not(target_arch = "wasm32"))]
    fn load_session(&mut self) {
        let model_file_path = Path::new(env!("CARGO_MANIFEST_DIR")).join(self.model_config.clone().unwrap().model_path);
        let loaded_session = Some(block_on(Session::from_path(model_file_path)).unwrap());
        let mut guard = self.session.lock().unwrap();
        *guard = loaded_session;
    }
    
    
    #[cfg(not(target_arch = "wasm32"))]
    fn execute_model(&mut self, model_input: ArrayD<f32>) -> Result<OutputTensor, WonnxError> {
        let mut input_data: HashMap<String, InputTensor> = HashMap::new();
        let key = &"input_data".to_string();
        input_data.insert(key.clone(), model_input.as_slice().unwrap().into());
        let guard = self.session.lock().unwrap();
        let session_ref = guard.as_ref().unwrap();
        let result = block_on(session_ref.run(&input_data)).ok();
        let output_tensor = result.unwrap().get(key).unwrap().clone();
        Ok(output_tensor)
    }

    
    //#[cfg(target_arch = "wasm32")]
    fn execute_model(&mut self, model_input: ArrayD<f32>) -> Result<OutputTensor, WonnxError> {
        let mut input_data: HashMap<String, InputTensor> = HashMap::new();
        input_data.insert("input_data".to_string(), model_input.as_slice().unwrap().into());
        let mut result: Option<OutputTensor> = None;

        spawn_local(async move {
            let guard = self.session.lock().unwrap();
            let session_ref = guard.as_ref().unwrap();
            let out = session_ref.run(&input_data).await.ok();
            let output_tensor = out.unwrap().get(&"input_data".to_string()).clone();
            result = output_tensor;
        });
        Ok(result)
    }

    #[cfg(target_arch = "wasm32")]
    fn load_session(&mut self) {
        let bytes = general_purpose::STANDARD_NO_PAD.decode(self.model_config.clone().unwrap().model_base64).unwrap();
        let mut new_session: Arc<Mutex<Option<Session>>> = Arc::new(Mutex::new(None));
        spawn_local(async move {
            let session_result = Session::from_bytes(&bytes).await.ok();
            let mut guard = new_session.lock().unwrap();
            *guard = session_result;
        });
    }    
}

impl Node for ModelNode
{  
    fn on_update(&mut self) -> Result<(), UpdateError> {
        if let Ok(input_model_config) = self.input_model_config.next() {
            self.model_config = Some(input_model_config);
            self.load_session();
        }
        if let Ok(model_input) = self.model_input.next() {
            let res = self.execute_model(model_input);
            if let Ok(out) = res {
                let result = Vec::try_from(out.clone()).unwrap();
                let _ = self.output.send(Array::from_vec(result).into_dyn());
            }
        }
        Ok(())
    }
}