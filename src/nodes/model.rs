use std::{env, error::Error, path::PathBuf, fmt::Debug};

use flowrs::RuntimeConnectable;
use flowrs::{
    connection::{Input, Output},
    node::{Node, UpdateError, ChangeObserver},
};
use serde::{Deserialize, Serialize};

use onnxruntime::{
    environment::Environment, 
    LoggingLevel, 
    GraphOptimizationLevel, 
    session::Session
};

use ndarray::{IxDynImpl, 
    Dim, 
    Array
};

#[derive(Clone, Deserialize, Serialize, Debug)]
pub struct ModelConfig {
   pub model_path: String
}



#[derive(RuntimeConnectable, Deserialize, Serialize)]
pub struct ModelNode
{
    #[input]
    pub input: Input<ModelConfig>,
    #[output]
    pub output: Output<i32>,
}


impl ModelNode
{
    pub fn new(change_observer: Option<&ChangeObserver>) -> Self {
        Self {
            input: Input::new(),
            output: Output::new(change_observer),
        }
    }
}

impl Node for ModelNode
{
    fn on_update(&mut self) -> Result<(), UpdateError> {
        run_model();
        Ok(())
    }
}

fn run_model() {
    let env = load_environment();

    let model_file_path = env::current_dir()
        .expect("Failed to obtain current directory")
        .join("src/models/squeezenet1.0-12.onnx");

    let mut session = load_session(&env, model_file_path);

    let input_shape: Vec<usize> = get_input_shape(&session)
        .expect("Failed to read input dimension");

    let input_size = input_shape.clone().into_iter()
        .reduce(|a, b| a * b)
        .expect("Failed to read input size");

    let array = Array::linspace(0.0_f32, 1.0, input_size as usize)
        .into_shape(input_shape)
        .expect("Failed to create input");

    let input_tensor = vec![array];

    let outputs = session.run::<f32, f32, Dim<IxDynImpl>>(input_tensor);
    print!("Output: {:?}", outputs);
}

fn load_environment() -> Environment {
    Environment::builder()
        .with_name("test_env")
        .with_log_level(LoggingLevel::Verbose)
        .build()
        .expect("Failed to create ONNX Runtime environment.")
}

fn load_session(environment: &Environment, model_file_path: PathBuf) -> Session<'_> {
    environment
        .new_session_builder()
        .expect("Failed to create session builder.")
        .with_optimization_level(GraphOptimizationLevel::Basic)
        .expect("Failed to set optimization level.")
        .with_number_threads(1)
        .expect("Failed to set the number of threads.")
        .with_model_from_file(model_file_path)
        .expect("Failed to load the model from file.")
}

fn get_input_shape(session: &Session<'_>) -> Result<Vec<usize>, Box<dyn Error>> {
    let dimensions: Result<Vec<_>, _> = session.inputs[0]
        .dimensions()
        .map(|d| d.ok_or("Failure"))
        .collect();
    Ok(dimensions?)
}


