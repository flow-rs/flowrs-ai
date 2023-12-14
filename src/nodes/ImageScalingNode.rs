use std::{fmt::Debug};

use flowrs::RuntimeConnectable;
use flowrs::{
    connection::{Input, Output},
    node::{Node, UpdateError, ChangeObserver},
};
use serde::{Deserialize, Serialize};

use image::{imageops::FilterType, DynamicImage};

#[derive(Clone, Debug)]
pub struct ScalingConfig {
   pub width: u32,
   pub height: u32
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
fn print_type_of<T>(_: &T) {
    println!("{}", std::any::type_name::<T>())
}

impl Node for ImageScalingNode {
    fn on_update(&mut self) -> Result<(), UpdateError> {
        let scaling_config = self.scaling_config.next();

        if let Ok(scaling_config) = scaling_config {
            let image = self.image.next();
            if let Ok(image) = image{
                let output = image.resize_exact(
                    scaling_config.width as u32,
                    scaling_config.height as u32,
                    FilterType::CatmullRom,
                );

                if output.width() == scaling_config.width && output.height() == scaling_config.height{
                    println!("Resized image dimensions: {} x {}", output.width(), output.height());
                }else{
                    return Err(UpdateError::Other(anyhow::Error::msg(
                        "Resizing was unsuccessful",)));
                }
            }else{
                return Err(UpdateError::Other(anyhow::Error::msg(
                    "Unable to get image",)));
            }
        }else{
            return Err(UpdateError::Other(anyhow::Error::msg(
                "Unable to get the scaling configuration",)));
        }

        Ok(())
    }
}

