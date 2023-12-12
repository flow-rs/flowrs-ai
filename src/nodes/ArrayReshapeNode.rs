use std::{fmt::Debug};

use flowrs::RuntimeConnectable;
use flowrs::{
    connection::{Input, Output},
    node::{Node, UpdateError, ChangeObserver},
};
use serde::{Deserialize, Serialize};
use ndarray::{ 
    Dim,
    OwnedRepr,
    ArrayBase,
    Array4,
    Array,
    ArrayD
};

use image::{GenericImageView, imageops::FilterType, DynamicImage};

#[derive(Clone, Deserialize, Serialize, Debug)]
pub struct ReshapeConfig {
   pub dimension: ArrayD<f32>
}

//#[derive(RuntimeConnectable, Deserialize, Serialize)]
pub struct ArrayReshapeNode
{
    #[input]
    pub input: ArrayD<f32>,
    pub ReshapeConfig: Input<ReshapeConfig>,
    #[output]
    pub output: ArrayD<f32>,
}

impl ArrayReshapeNode
{
    pub fn new(change_observer: Option<&ChangeObserver>) -> Self {
        Self {
            input: Input::new(),
            output: Output::new(change_observer),
        }
    }
}


impl Node for ArrayReshapeNode
{
    fn on_update(&mut self) -> Result<(), UpdateError> {
        let output = reshape_array();
        if let Ok(output) = output{
        }
        Ok(())
    }
}

fn print_type_of<T>(_: &T) {
    println!("{}", std::any::type_name::<T>())
}

fn reshape_array() -> Result<ArrayD<f32>, UpdateError>{
    let reshaped_array = self.input.into_shape(self.ReshapeConfig.dimension)
    
    Ok(reshaped_array)
}

