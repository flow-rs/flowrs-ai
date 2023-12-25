#[cfg(test)]
mod nodes {
    use flowrs::{node::{ChangeObserver, Node}, connection::connect};
    use flowrs_std::value::ValueNode;
    use flowrs_std::debug::DebugNode;
    //use flowrs_img::webcam::{WebcamNode, WebcamNodeConfig};
    use flowrs_ai::ImageScalingNode::{ImageScalingNode, ScalingConfig};
    use flowrs_ai::PreproccessingNode::PreproccessingNode;
    use flowrs_ai::model::{ModelNode, ModelConfig};
    use std::{env};
    use image::{DynamicImage};
    use flowrs::connection::Edge;
    use flowrs_ai::{max_output_value::MaxOutputNode};

    #[test]
    fn test_webcamwithonnx() -> Result<(), anyhow::Error> {
        let change_observer: ChangeObserver = ChangeObserver::new();
        // initialize camera
        //let webcam_config = WebcamNodeConfig { device_index: 0 };
        //let mut webcam = WebcamNode::<i32>::new(webcam_config, Some(&change_observer));        
        
        // creating scaling config
        let scaling_config = ScalingConfig{
            width: 224,
            height: 224,
        };
        let model_config = ModelConfig {
            model_path: "src/models/opt-squeeze.onnx".to_string(),
            model_base64: "".to_string(),
        };
        let image_path = env::current_dir()
        .expect("Failed to obtain current directory")
        .join("src/example_pic/crosswalk.jpg");
        let img = image::open(image_path).expect("Failed to open image");
        // creating flow and test functionality
        let image_value = ValueNode::new(img, Some(&change_observer));
        let scaling_config_value = ValueNode::new(scaling_config, Some(&change_observer));
        let mut image_scaling_node = ImageScalingNode::new(Some(&change_observer));
        let mut preproccessing_node = PreproccessingNode::new(Some(&change_observer));
        let mut model_node = ModelNode::new(Some(&change_observer));
        let mut post_processing = MaxOutputNode::new(Some(&change_observer));
        // get classes from model
        let input_classes = "";
        //let mut debug = DebugNode::new(Some(&change_observer));
        let mock_output = Edge::new();
        //connect(webcam.output.clone(), image_scaling_node.image.clone());
        connect(image_value.output.clone(), image_scaling_node.image.clone());
        connect(scaling_config_value.output.clone(), image_scaling_node.input_scaling_config.clone());
        connect(image_scaling_node.output.clone(), preproccessing_node.input.clone());
        connect(preproccessing_node.output.clone(), model_node.model_input.clone());
        connect(model_node.output.clone(), mock_output.clone());
        //connect(model_node.output.clone(), post_processing.output_tensor.clone());
        //connect(post_processing.output.clone(), mock_output.clone());
        //connect(debug.output.clone(), mock_output.clone());
        let _ = model_node.input_model_config.send(model_config);
        //let _ = post_processing.input_classes.send(input_classes);
        let _ = image_value.on_ready();
        let _ = scaling_config_value.on_ready();

        let _ = image_scaling_node.on_update();
        let _ = preproccessing_node.on_update();
        let _ = model_node.on_update();
        //let _ = post_processing.on_update();
        //debug.input.send(tensor);

        //let result = debug.on_update();

        let tensor = mock_output.next()?;

        print!("{:?}", tensor);
        // condition must be changed, but it is not finished yet.
        Ok(assert!(true))

    }
    
}