/* 
use flowrs::{node::{Node, UpdateError, ChangeObserver}, connection::{Input, Output}};
use flowrs::RuntimeConnectable;

use ndarray::{Array3, ArrayBase, OwnedRepr, Dim};
use anyhow::{anyhow};

use serde::{Deserialize, Serialize};
*/
/*
#[derive(RuntimeConnectable, Deserialize, Serialize)]
pub struct DecodeImageNode {
    #[output]
    pub output: Output<DynamicImage>,

    #[input]
    pub input: Input<Vec<u8>>,
}

impl DecodeImageNode {
    pub fn new(change_observer: Option<&ChangeObserver>) -> Self {
        Self {
            output: Output::new(change_observer),
            input: Input::new()
        }
    }
}

impl Node for DecodeImageNode {
    fn on_update(&mut self) -> Result<(), UpdateError> {

        if let Ok(data) = self.input.next() {

            let img = ImageReader::new(Cursor::new(data))
            .with_guessed_format().map_err(|e| UpdateError::Other(e.into()))?
            .decode().map_err(|e| UpdateError::Other(e.into()))?;

            self.output.send(img).map_err(|e| UpdateError::Other(e.into()))?;
        }
        Ok(())
    }
}

// TODO:    - EncodeImageNode, Array3ToImage, ConvertImageToImage 
//          - How to replace DynamicImage with something like ImageBuffer<P, Vec<<P as Pixel>::Subpixel>>
//          - Tests 
*/
