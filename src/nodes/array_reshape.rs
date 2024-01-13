use flowrs::RuntimeConnectable;
use flowrs::{
    connection::{Input, Output},
    node::{ChangeObserver, Node, UpdateError},
};
use ndarray::{Array, ArrayD, Dimension, IxDyn, ShapeError};
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ArrayReshapeNodeConfig {
    pub dimension: Vec<usize>,
}

#[derive(RuntimeConnectable, Serialize, Deserialize)]
pub struct ArrayReshapeNode<D> {
    #[input]
    pub array_input: Input<Array<f32, D>>,

    #[input]
    pub config_input: Input<ArrayReshapeNodeConfig>,

    #[output]
    pub array_output: Output<ArrayD<f32>>,

    config_object: Option<ArrayReshapeNodeConfig>,
}

impl<D> ArrayReshapeNode<D> {
    pub fn new(change_observer: Option<&ChangeObserver>) -> Self {
        Self {
            array_input: Input::new(),
            config_input: Input::new(),
            array_output: Output::new(change_observer),
            config_object: Option::None,
        }
    }
}

impl<D> Node for ArrayReshapeNode<D>
where
    D: Dimension,
{
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
            let it = array.iter().cloned();
            let new_array = Array::from_iter(it);
            let shape = IxDyn(&self.config_object.clone().unwrap().dimension);
            match new_array.into_shape(shape) {
                Ok(reshaped_array) => match self.array_output.clone().send(reshaped_array) {
                    Ok(_) => return Ok(()),
                    Err(err) => return Err(UpdateError::Other(err.into())),
                },
                Err(err) => return Err(UpdateError::Other(err.into())),
            }
        }
        Ok(())
    }
}
