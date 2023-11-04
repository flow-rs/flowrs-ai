use std::borrow::Borrow;
use std::{env, fmt::Debug, collections::HashMap};
use image::{imageops::FilterType, ImageBuffer, Pixel, Rgb};
use ndarray::s;
use flowrs::RuntimeConnectable;
use flowrs::{
    connection::{Input, Output},
    node::{Node, UpdateError, ChangeObserver},
};
use serde::{Deserialize, Serialize};

use wonnx::{
    Session,
    utils::OutputTensor,
    WonnxError
};
/* 
use web_sys::GpuStoreOp;
use web_sys::GpuUncapturedErrorEvent;
use web_sys::GpuCanvasContext;
use web_sys::GpuRenderBundle;
use web_sys::GpuRenderBundleEncoder;
use web_sys::GpuCommandBuffer;
use web_sys::GpuRenderPassEncoder;
use web_sys::GpuRenderPipeline;
use web_sys::GpuComputePassEncoder;
use web_sys::GpuCommandEncoder;
use web_sys::GpuComputePipeline;
use web_sys::GpuPipelineLayout;
use web_sys::GpuQuerySet;
use web_sys::GpuTexture;
use web_sys::GpuBuffer;
use web_sys::GpuSampler;
use web_sys::GpuTextureView;
use web_sys::GpuBindGroup;
use web_sys::GpuBindGroupLayout;
use web_sys::GpuShaderModule;
use web_sys::GpuQueue;
use web_sys::GpuDevice;
use web_sys::GpuAdapter;
use web_sys::GpuCanvasContext;
use web_sys::GpuSupportedFeatures;
*/
use web_sys::*;
use wasm_bindgen::prelude::*;

#[derive(Clone, Deserialize, Serialize, Debug)]
pub struct ModelConfig {
   pub model_path: String
}

#[derive(RuntimeConnectable, Deserialize, Serialize)]
pub struct ModelNode
{
    #[input]
    pub input: Input<ModelConfig>,
    #[output]
    pub output: Output<i32>,
    pub model_config: Option<ModelConfig>,
}

#[wasm_bindgen]
impl ModelNode
{
    pub fn new(change_observer: Option<&ChangeObserver>) -> Self {
        Self {
            input: Input::new(),
            output: Output::new(change_observer),
            model_config: None,
        }
    }
}
[#wasm_bindgen]
impl Node for ModelNode
{
    fn on_update(&mut self) -> Result<(), UpdateError> {
        if let Ok(input) = self.input.next() {
            self.model_config = Some(input);
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            let config = self.model_config.clone().unwrap();
            let _ = pollster::block_on(run(config)).unwrap();
            Ok(())
        }
        #[cfg(target_arch = "wasm32")]
        {
            wasm_bindgen_futures::spawn_local(run());
            Ok(())
        }
    }
}
#[wasm_bindgen]
async fn run(model_config: ModelConfig) -> Result<HashMap<String, OutputTensor>, WonnxError> {
    let mut input_data = HashMap::new();
    let image = load_image();
    input_data.insert("data".to_string(), image.as_slice().unwrap().into());

    let model_file_path = env::current_dir()
        .expect("Failed to obtain current directory")
        .join(model_config.model_path);
    let session = Session::from_path(model_file_path).await.expect("Failed to load Session");
    let result = session.run(&input_data).await?;
    Ok(result)
}

#[wasm_bindgen]
// TODO: Seperate Image Loading and Preprocessing
fn load_image() -> ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::Dim<[usize; 4]>> {
    let image_path = env::current_dir()
    .expect("Failed to obtain current directory")
    .join("src/images/7.jpg");

    let image_buffer: ImageBuffer<Rgb<u8>, Vec<u8>> = image::open(image_path)
        .unwrap()
        .resize_to_fill(224, 224, FilterType::Nearest)
        .to_rgb8();

    let mut array = ndarray::Array::from_shape_fn((1, 3, 224, 224), |(_, c, j, i)| {
        let pixel = image_buffer.get_pixel(i as u32, j as u32);
        let channels = pixel.channels();

        (channels[c] as f32) / 255.0
    });

  
    let mean = [0.485, 0.456, 0.406];
    let std = [0.229, 0.224, 0.225];
    for c in 0..3 {
        let mut channel_array = array.slice_mut(s![0, c, .., ..]);
        channel_array -= mean[c];
        channel_array /= std[c];
    }

    array
}


