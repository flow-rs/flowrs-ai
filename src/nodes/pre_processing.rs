use flowrs::RuntimeConnectable;
use flowrs::{
    connection::{Input, Output},
    node::{Node, UpdateError, ChangeObserver},
};
use serde::{Deserialize, Serialize};
use ndarray::{ 
    OwnedRepr,
    ArrayBase,
    Dim,
    ArrayD,
    IxDynImpl,
};


#[derive(RuntimeConnectable, Deserialize, Serialize)]
pub struct PreproccessingNode
{
    #[input]
    pub input: Input<ArrayD<f32>>,
    #[output]
    pub output: Output<ArrayD<f32>>,
}

impl PreproccessingNode
{
    pub fn new(change_observer: Option<&ChangeObserver>) -> Self {
        Self {
            input: Input::new(),
            output: Output::new(change_observer),
        }
    }
}


impl Node for PreproccessingNode
{
    fn on_update(&mut self) -> Result<(), UpdateError> {
        if let Ok(input) = self.input.next(){
            let preprocced_input = preproccessing_input(input);
            match self.output.send(preprocced_input) {
                Ok(_) => Ok(()),
                Err(err) => Err(UpdateError::Other(err.into())),
            }
        }else{
            return Err(UpdateError::Other(anyhow::Error::msg(
                "Unable to get image to process",)));
        }
    }
}
fn preproccessing_input(mut input: ArrayD<f32>) -> ArrayBase<OwnedRepr<f32>, Dim<IxDynImpl>> {
    for element in input.iter_mut() {
        *element = *element / 255.0;
    }
    input
}