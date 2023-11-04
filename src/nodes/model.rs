use std::{env, fmt::Debug, collections::HashMap};
use ndarray::{ArrayD};
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
    pub output: Output<i32>,
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
            #[cfg(not(target_arch = "wasm32"))]
            {
                let _ = pollster::block_on(run(config, model_input)).unwrap();
            }
            #[cfg(target_arch = "wasm32")]
            {
                wasm_bindgen_futures::spawn_local(run(config));
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
    println!("Result: {:?}", result);
    Ok(result)
}

