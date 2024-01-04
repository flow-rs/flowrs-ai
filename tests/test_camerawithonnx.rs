#[cfg(test)]
mod nodes {
    use flowrs::{node::{ChangeObserver, Node}, connection::connect};
    //use flowrs_img::webcam::{WebcamNode, WebcamNodeConfig};
    use flowrs_ai::ImageScalingNode::{ImageScalingNode, ScalingConfig};
    use flowrs_ai::PreproccessingNode::PreproccessingNode;
    use flowrs_ai::model::{ModelNode, ModelConfig};
    use std::{env, path::Path, fs::File, io::Read};
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
        let image_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/images/pelican.jpeg");
        let img = image::open(image_path).expect("Failed to open image");

        let mut labels_file: Vec<u8> = Vec::new();
        let labels_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/models/squeeze-labels.txt");
        let mut file = File::open(labels_path).expect("Failed to load file");   
        let _ = file.read_to_end(&mut labels_file).unwrap();
        
        // creating flow and test functionality
        //let image_value = ValueNode::new(img, Some(&change_observer));
        // let scaling_config_value = ValueNode::new(scaling_config, Some(&change_observer));
        let mut image_scaling_node = ImageScalingNode::new(Some(&change_observer));
        let mut preproccessing_node = PreproccessingNode::new(Some(&change_observer));
        let mut model_node = ModelNode::new(Some(&change_observer));
        let mut post_processing = MaxOutputNode::new(Some(&change_observer));
        // get classes from model
        //let mut debug = DebugNode::new(Some(&change_observer));
        let mock_output = Edge::new();
        connect(image_scaling_node.output.clone(), preproccessing_node.input.clone());
        connect(preproccessing_node.output.clone(), model_node.model_input.clone());
        connect(model_node.output.clone(), post_processing.output_tensor.clone());
        connect(post_processing.output_class.clone(), mock_output.clone());
        let _ = model_node.input_model_config.send(model_config);
        let _ = post_processing.input_classes.send(labels_file);

        let _ = image_scaling_node.input_scaling_config.send(scaling_config);
        let _ = image_scaling_node.image.send(img);
        let _ = image_scaling_node.on_update();
        let _ = preproccessing_node.on_update();
        let _ = model_node.on_update();

        let _ = post_processing.on_update();

        let actual = mock_output.next()?;

        // condition must be changed, but it is not finished yet.
        Ok(assert!(actual == "n02051845 pelican"))

    }
    
}