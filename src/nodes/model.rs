use std::path::Path;
use std::{fmt::Debug, collections::HashMap};

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
    Session
};

#[cfg(target_arch = "wasm32")]
use base64::{Engine as _, engine::general_purpose};
#[cfg(target_arch = "wasm32")]
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
    pub session: Option<Session>,
}

impl ModelNode
{
    pub fn new(change_observer: Option<&ChangeObserver>) -> Self {
        Self {
            input_model_config: Input::new(),
            model_input: Input::new(),
            output: Output::new(change_observer),
            model_config: None,
            session: None,
        }
    }
    
    #[cfg(not(target_arch = "wasm32"))]
    fn load_session(&mut self) {
        let model_file_path = Path::new(env!("CARGO_MANIFEST_DIR")).join(self.model_config.clone().expect("no model path supplied").model_path);
        let loaded_session = Some(block_on(Session::from_path(model_file_path)).expect("could not load model"));
        self.session = loaded_session;
    }
    
    #[cfg(not(target_arch = "wasm32"))]
    fn execute_model(&mut self, model_input: ArrayD<f32>) -> Result<OutputTensor, String> {
        let mut input_data: HashMap<String, InputTensor> = HashMap::new();
        let input = model_input.as_slice();
        match input {
            Some(i) => {
                input_data.insert("data".to_string(), i.into());
            }
            None => {
                return Err("failed to read model input".to_string())
            }
        }
        
        let session_ref = self.session.as_ref();
        match session_ref {
            Some(session) => {
                match block_on(session.run(&input_data)) {
                    Ok(res) => {
                        match res.into_iter().next() {
                            Some(output_tensor) => {
                                return Ok(output_tensor.1)
                            }
                            None => {
                                return Err("could not unwrap result".to_string())
                            }
                        }
                    }
                    Err(e) => {
                        return Err(e.to_string())
                    }
                }
            }
            None => {
                return Err("could not execute model".to_string())
            }
        }
    }
    
    #[cfg(target_arch = "wasm32")]
    fn execute_model(&mut self, model_input: &ArrayD<f32>) {
        let mut input_data: HashMap<String, InputTensor> = HashMap::new();
        input_data.insert("data".to_string(), model_input.as_slice().unwrap().into());
        let session_ref = self.session.as_ref().unwrap();
        spawn_local(async move {    
            let result: Option<HashMap<String, OutputTensor>> = session_ref.run(&input_data).await.ok();
            // TODO: How to get the result out of async
        });
    }

    #[cfg(target_arch = "wasm32")]
    fn load_session(&mut self) {
        let bytes = general_purpose::STANDARD_NO_PAD.decode(self.model_config.clone().unwrap().model_base64).unwrap();
        spawn_local(async move {
            let session_result = Session::from_bytes(&bytes).await.ok();
            // TODO: result also doesn't get out of async
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
            let output_tensor = self.execute_model(model_input);
            let result: Vec<f32> = output_tensor.unwrap().try_into().unwrap();
            let _ = self.output.send(Array::from_vec(result).into_dyn());
        }
        Ok(())
    }
}