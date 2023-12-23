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

use image::{GenericImageView,DynamicImage};


#[derive(RuntimeConnectable, Deserialize, Serialize)]
pub struct PreproccessingNode
{
    #[input]
    pub input: Input<DynamicImage>,
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
            let output = preproccessing_input(input);
        }else{
            return Err(UpdateError::Other(anyhow::Error::msg(
                "Unable to get image to process",)));
        }

        Ok(())
    }
}

fn print_type_of<T>(_: &T) {
    println!("{}", std::any::type_name::<T>())
}

fn preproccessing_input(image: DynamicImage) -> Result<ArrayBase<OwnedRepr<f32>, Dim<[usize; 4]>>, UpdateError>{
    let resized_image = image;

    let (width, height) = resized_image.dimensions();

    let dim = Dim((1,3,width as usize, height as usize));
    let mut input_tensor: Array4<f32> = Array::zeros(dim);
     
    for pixel in resized_image.pixels() {
        let x = pixel.0 as usize;
        let y = pixel.1 as usize;
        let [r,g,b,_] = pixel.2.0;
        input_tensor[[0, 0, y, x]] = (r as f32) / 255.0;
        input_tensor[[0, 1, y, x]] = (g as f32) / 255.0;
        input_tensor[[0, 2, y, x]] = (b as f32) / 255.0;
    };
    //let output = input_tensor.clone();
    Ok(input_tensor)
}

