use std::fmt::Debug;

use flowrs::RuntimeConnectable;
use flowrs::{
    connection::{Input, Output},
    node::{Node, UpdateError, ChangeObserver},
};
use serde::{Deserialize, Serialize};

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
        if let Ok(input) = self.input.next() {
            println!("{:?} Received Config: {:?}", std::thread::current().id(), input);
            #[cfg(target_arch = "wasm32")]
            crate::log(format!("{:?} Received Config: {:?}", std::thread::current().id(),input).as_str());
            // Send output
            let _ = self.output.send(0);
        }
        Ok(())
    }
}