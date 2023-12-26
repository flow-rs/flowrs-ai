use std::fmt::Debug;

use flowrs::RuntimeConnectable;
use flowrs::{
    connection::{Input, Output},
    node::{ChangeObserver, Node, UpdateError},
};
use ndarray::{ArrayD, IxDyn, ShapeError};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ArrayReshapeNodeConfig {
    pub dimension: Vec<usize>,
}

#[derive(RuntimeConnectable, Serialize, Deserialize)]
pub struct ArrayReshapeNode {
    #[input]
    pub array_input: Input<ArrayD<f32>>,

    #[input]
    pub config_input: Input<ArrayReshapeNodeConfig>,

    #[output]
    pub array_output: Output<ArrayD<f32>>,

    config_object: Option<ArrayReshapeNodeConfig>,
}

impl ArrayReshapeNode {
    pub fn new(change_observer: Option<&ChangeObserver>) -> Self {
        Self {
            array_input: Input::new(),
            config_input: Input::new(),
            array_output: Output::new(change_observer),
            config_object: Option::None,
        }
    }
    pub fn reshape(&self, input: ArrayD<f32>) -> Result<ArrayD<f32>, ShapeError> {
        let output = input.into_shape(self.config_object.clone().unwrap().dimension);
        return output;
    }
}

impl Node for ArrayReshapeNode {
    fn on_update(&mut self) -> Result<(), UpdateError> {
        if let Ok(config) = self.config_input.next() {
            self.config_object = Some(config);
        }

        if self.config_object.is_none() {
            return Err(UpdateError::Other(anyhow::Error::msg(
                "No config to reshape array.",
            )));
        }

        if let Ok(array) = self.array_input.next() {
            match array.into_shape(IxDyn(&self.config_object.clone().unwrap().dimension)) {
                Ok(reshaped_array) => match self.array_output.clone().send(reshaped_array) {
                    Ok(_) => Ok(()),
                    Err(err) => Err(UpdateError::Other(err.into())),
                },
                Err(err) => Err(UpdateError::Other(err.into())),
            }
        } else {
            Err(UpdateError::Other(anyhow::Error::msg(
                "No array given to input.",
            )))
        }
    }
}
