use flowrs_ai::model::ModelNode;
use flowrs::{node::{ChangeObserver, Node}};

mod nodes {

    use flowrs_ai::model::{ModelNode, ModelConfig};
    use flowrs::{node::{ChangeObserver, Node}, connection::connect};
    use flowrs_std::value::ValueNode;

    use ndarray::{ArrayD, s};
    use std::{env};
    use image::{imageops::FilterType, ImageBuffer, Pixel, Rgb};

    #[test]
    fn should_run_model() -> Result<(), anyhow::Error> {
        let model_config = ModelConfig {
            model_path: "src/models/opt-squeeze.onnx".to_string(),
            model_base64: "".to_string(),
        };
        let model_input = load_image();

        let change_observer: ChangeObserver = ChangeObserver::new();  

        let image_value = ValueNode::new(model_input, Some(&change_observer));
        let config_value = ValueNode::new(model_config, Some(&change_observer));
        let mut model_node = ModelNode::new(Some(&change_observer));

        connect(config_value.output.clone(), model_node.input_model_config.clone());
        connect(image_value.output.clone(), model_node.model_input.clone());

        let _ = config_value.on_ready();
        let _ = model_node.on_update();

        let _ = image_value.on_ready();

        let result = model_node.on_update();

        Ok(assert!(result.is_ok()))
    }

    #[test]
    fn should_load_model_path() -> Result<(), anyhow::Error> {
        let path = "src/models/opt-squeeze.onnx";
        let model_config = ModelConfig {
            model_path: path.to_string(),
            model_base64: "".to_string(),
        };
        let change_observer: ChangeObserver = ChangeObserver::new();  
        let value_node = ValueNode::new(model_config, Some(&change_observer));
        let mut model_node = ModelNode::new(Some(&change_observer));
        connect(value_node.output.clone(), model_node.input_model_config.clone());
        let _ = value_node.on_ready();
        let _ = model_node.on_update();
        Ok(assert!(path == model_node.model_config.unwrap().model_path))
        //Ok(assert!(model_node.session.borrow()))
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

