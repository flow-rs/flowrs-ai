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
    Array
};

use image::{GenericImageView, imageops::FilterType, DynamicImage};


#[derive(RuntimeConnectable, Deserialize, Serialize)]
pub struct PreProccessingNode
{
    #[input]
    pub input: Input<DynamicImage>,
    #[output]
    pub output: Output<ArrayD<f32>>,
}

impl PreProccessingNode
{
    pub fn new(change_observer: Option<&ChangeObserver>) -> Self {
        Self {
            input: Input::new(),
            output: Output::new(change_observer),
        }
    }
}


impl Node for PreProccessingNode
{
    fn on_update(&mut self) -> Result<(), UpdateError> {
        let output = preproccessing_input();
        if let Ok(output) = output{
        }
        Ok(())
    }
}

fn print_type_of<T>(_: &T) {
    println!("{}", std::any::type_name::<T>())
}

fn preproccessing_input() -> Result<ArrayBase<OwnedRepr<f32>, Dim<[usize; 4]>>, UpdateError>{
    let input_path = "C:/Users/Marcel/LRZ Sync+Share/Master/3_Semester/Hauptseminar_2/flow-rs/flowrs-ai/src/example_pic/crosswalk.jpg";
    let img = image::open(input_path).expect("Failed to open image"); // img = input
    //let img = self.input;

    let squeezenet_width: usize = 224;
    let squeezenet_height: usize = 224;

    let resized_image = img.resize_exact(squeezenet_width as u32, squeezenet_height as u32, FilterType::CatmullRom);

    let dim = Dim((1,3,squeezenet_width, squeezenet_height));
    print_type_of(&squeezenet_height);
    let mut input_tensor: Array4<f32> = Array::zeros(dim);
    print_type_of(&input_tensor);
     
    for pixel in resized_image.pixels() {
        let x = pixel.0 as usize;
        let y = pixel.1 as usize;
        let [r,g,b,_] = pixel.2.0;
        input_tensor[[0, 0, y, x]] = (r as f32) / 255.0;
        input_tensor[[0, 1, y, x]] = (g as f32) / 255.0;
        input_tensor[[0, 2, y, x]] = (b as f32) / 255.0;
    };
    print_type_of(&input_tensor);
    
    Ok(input_tensor)
}

