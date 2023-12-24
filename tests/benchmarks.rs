
mod nodes {

    use flowrs_ai::model::{ModelNode, ModelConfig};
    use flowrs::{node::{ChangeObserver, Node}, connection::{connect, Edge}};

    use ndarray::{ArrayD, s};
    use std::{env, time::Instant};
    use image::{imageops::FilterType, ImageBuffer, Pixel, Rgb};

    #[test]
    fn benchmark_model_node() {
        // given
        let model_config = ModelConfig {
            model_path: "src/models/opt-squeeze.onnx".to_string(),
            model_base64: "".to_string(),
        };
        let model_input = load_image();

        let change_observer: ChangeObserver = ChangeObserver::new();  

        let mut model_node = ModelNode::new(Some(&change_observer));
        let mock_output = Edge::new();
        connect(model_node.output.clone(), mock_output.clone());
        let num_executions = 100;
        // when
        let _ = model_node.input_model_config.send(model_config.clone());
        let _ = model_node.on_update();
        println!("Start benchmarks with {} iterations", num_executions);
        println!("...");
        let now = Instant::now();
        for _ in 1..num_executions {
            let _ = model_node.model_input.send(model_input.clone()); 
            let _ = model_node.on_update();
        }
        let elapsed = now.elapsed().as_millis();
        // then
        println!("Result: {}ms per executions \n", elapsed / num_executions);
    }

    fn load_image() -> ArrayD<f32> {
        let image_path = env::current_dir()
        .expect("Failed to obtain current directory")
        .join("src/images/7.jpg");
        println!("Image Path: {:?}", image_path);
    
        let image_buffer: ImageBuffer<Rgb<u8>, Vec<u8>> = image::open(image_path)
            .unwrap()
            .resize_to_fill(224, 224, FilterType::Nearest)
            .to_rgb8();
    
        let mut array: ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::Dim<[usize; 4]>> = ndarray::Array::from_shape_fn((1, 3, 224, 224), |(_, c, j, i)| {
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
    
        array.into_dyn()
    }
}