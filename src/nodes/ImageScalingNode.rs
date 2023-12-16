use std::{fmt::Debug};

use flowrs::RuntimeConnectable;
use flowrs::{
    connection::{Input, Output},
    node::{Node, UpdateError, ChangeObserver},
};

use image::{imageops::FilterType, DynamicImage};

#[derive(Clone, Debug)]
pub struct ScalingConfig {
   pub width: u32,
   pub height: u32
}

#[derive(RuntimeConnectable)]
pub struct ImageScalingNode
{
    #[input]
    pub image: Input<DynamicImage>,
    pub input_scaling_config: Input<ScalingConfig>,
    #[output]
    pub output: Output<DynamicImage>,
    pub scaling_config: Option<ScalingConfig>,
}

impl ImageScalingNode
{
    pub fn new(change_observer: Option<&ChangeObserver>) -> Self {
        Self {
            image: Input::new(),
            input_scaling_config: Input::new(),
            output: Output::new(change_observer),
            scaling_config: None,
        }
    }
}

impl Node for ImageScalingNode {
    fn on_update(&mut self) -> Result<(), UpdateError> {
        println!("start of on update");
        if self.scaling_config.is_none(){
            if let Ok(input_scaling_config) = self.input_scaling_config.next(){
                self.scaling_config = Some(input_scaling_config);
            }else{
                return Err(UpdateError::Other(anyhow::Error::msg(
                    "Unable to get scaling configuration",)));
            }
        }
        if let Ok(image) = self.image.next(){
            let result = image_scaling(self.scaling_config.clone().unwrap(), image);
            result
        }else{
            return Err(UpdateError::Other(anyhow::Error::msg(
                "Unable to get image",)));
        }
        /*if let Ok(input_scaling_config) = self.input_scaling_config.next(){
            //self.scaling_config = Some(input_scaling_config);
        
            println!("{:?}", self.scaling_config.clone().unwrap());
            if let Ok(image) = self.image.next(){
                let output = image.resize_exact(
                    self.scaling_config.clone().unwrap().width as u32,
                    self.scaling_config.clone().unwrap().height as u32,
                    FilterType::CatmullRom,
                );
                if output.width() == self.scaling_config.clone().unwrap().width && output.height() == self.scaling_config.clone().unwrap().height{
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
    */
    }
}

fn image_scaling(scaling_conf: ScalingConfig, image: DynamicImage)-> Result<(), UpdateError>{
    let output = image.resize_exact(
        scaling_conf.width as u32,
        scaling_conf.height as u32,
        FilterType::CatmullRom,
    );
    if output.width() == scaling_conf.width && output.height() == scaling_conf.height{
        println!("Resized image dimensions: {} x {}", output.width(), output.height());
    }else{
        return Err(UpdateError::Other(anyhow::Error::msg(
            "Resizing was unsuccessful",)));
    }
    Ok(())
}
