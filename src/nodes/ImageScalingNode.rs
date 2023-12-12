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
pub struct ScalingConfig {
   pub width: int,
   pub height: int
}

#[derive(RuntimeConnectable, Deserialize, Serialize)]
pub struct ImageScalingNode
{
    #[input]
    pub image: Input<DynamicImage>,
    pub scaling_config: Input<ScalingConfig>,
    #[output]
    pub output: Output<DynamicImage>,
}

impl ImageScalingNode
{
    pub fn new(change_observer: Option<&ChangeObserver>) -> Self {
        Self {
            image: Input::new(),
            scaling_config: Input::new(),
            output: Output::new(change_observer),
        }
    }
}


impl Node for ImageScalingNode
{
    fn on_update(&mut self) -> Result<(), UpdateError> {
        let output = image_scale();
        if let Ok(output) = output{
        }
        Ok(())
    }
}

fn print_type_of<T>(_: &T) {
    println!("{}", std::any::type_name::<T>())
}

fn image_scale() -> Result<DynamicImage, UpdateError>{

    let width: usize = scaling_config.width;
    let height: usize = scaling_config.height;

    let resized_image = self.image.resize_exact(width as u32, height as u32, FilterType::CatmullRom);

    Ok(resized_image)
}

