use std::collections::hash_map::RandomState;
use std::{env, fmt::Debug, collections::HashMap};
use ndarray::{ArrayD, Array};
use flowrs::RuntimeConnectable;
use flowrs::{
    connection::{Input, Output},
    node::{Node, UpdateError, ChangeObserver},
};
use serde::{Deserialize, Serialize};

use wonnx::{
    Session,
    utils::OutputTensor,
    WonnxError
};

#[derive(Clone, Deserialize, Serialize, Debug)]
pub struct ModelConfig {
   pub model_path: String
}

#[derive(RuntimeConnectable, Deserialize, Serialize)]
pub struct ModelNode
{
    #[input]
    pub input_model_config: Input<ModelConfig>,
    #[input]
    pub model_input: Input<ArrayD<f32>>,
    #[output]
    pub output: Output<ArrayD<f32>>,
    pub model_config: Option<ModelConfig>,
}

impl ModelNode
{
    pub fn new(change_observer: Option<&ChangeObserver>) -> Self {
        Self {
            input_model_config: Input::new(),
            model_input: Input::new(),
            output: Output::new(change_observer),
            model_config: None,
        }
    }
}

impl Node for ModelNode
{
    
    fn on_update(&mut self) -> Result<(), UpdateError> {
        if let Ok(input_model_config) = self.input_model_config.next() {
            self.model_config = Some(input_model_config);
        }
        if let Ok(model_input) = self.model_input.next() {
            let config = self.model_config.clone().unwrap();
            let res: Result<HashMap<String, OutputTensor, RandomState>, WonnxError>;
            #[cfg(not(target_arch = "wasm32"))]
            {
                res = pollster::block_on(run(config, model_input));
            }
            #[cfg(target_arch = "wasm32")]
            {
                res = wasm_bindgen_futures::spawn_local(run(config, model_input));
            }
            if let Ok(out) = res {
                for (_, output_tensor) in out {
                    let result = Vec::try_from(output_tensor).unwrap();
                    let _ = self.output.send(Array::from_vec(result).into_dyn());
                }
            }
        }
        Ok(())
    }
    
}

async fn run(model_config: ModelConfig, model_input: ArrayD<f32>) -> Result<HashMap<String, OutputTensor>, WonnxError> {
    let mut input_data = HashMap::new();
    input_data.insert("data".to_string(), model_input.as_slice().unwrap().into());

    let model_file_path = env::current_dir()
        .expect("Failed to obtain current directory")
        .join(model_config.model_path);
    let session = Session::from_path(model_file_path).await?;
    let result = session.run(&input_data).await?;
    Ok(result)
}

