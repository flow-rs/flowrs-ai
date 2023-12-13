use std::{fmt::Debug};

use flowrs::RuntimeConnectable;
use flowrs::{
    connection::{Input, Output},
    node::{Node, UpdateError, ChangeObserver},
};
use ndarray::{ 
    ArrayD,
    Dim,
    IxDynImpl
};

#[derive(Clone, Debug)]
pub struct ReshapeConfig {
   pub dimension: Dim<IxDynImpl>
}

#[derive(RuntimeConnectable)]
pub struct ArrayReshapeNode
{
    #[input]
    pub input: Input<ArrayD<f32>>,
    pub reshape_config: Input<ReshapeConfig>,
    #[output]
    pub output: Output<ArrayD<f32>>,
}

impl ArrayReshapeNode
{
    pub fn new(change_observer: Option<&ChangeObserver>) -> Self {
        Self {
            input: Input::new(),
            reshape_config: Input::new(),
            output: Output::new(change_observer),
        }
    }
}


impl Node for ArrayReshapeNode
{
    fn on_update(&mut self) -> Result<(), UpdateError> {
        let reshape_config = self.reshape_config.next();
        if let Ok(reshape_config) = reshape_config{
            let output = self.input.reshape(reshape_config.dimension);
            if output.shape() == reshape_config.dimension{
                println!("Reshaped successfully with dimension: {}", output.shape());
        }   else{
                //Err(err) => println!("Error: Could not reshape the input");
        }
        }
        Ok(())
    }
}

