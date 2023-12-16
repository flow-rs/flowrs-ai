#[cfg(test)]
mod nodes {
    use flowrs::{node::{ChangeObserver, Node}, connection::connect};
    use flowrs_std::value::ValueNode;
    use flowrs_std::debug::DebugNode;
    //use flowrs_img::webcam::{WebcamNode, WebcamNodeConfig};
    use flowrs_ai::ImageScalingNode::{ImageScalingNode, ScalingConfig};
    use flowrs_ai::PreproccessingNode::PreproccessingNode;
    use std::{env};
    use image::{DynamicImage};

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
        let image_path = env::current_dir()
        .expect("Failed to obtain current directory")
        .join("src/example_pic/crosswalk.jpg");
        let img = image::open(image_path).expect("Failed to open image");
        // creating flow and test functionality
        let image_value = ValueNode::new(img, Some(&change_observer));
        let scaling_config_value = ValueNode::new(scaling_config, Some(&change_observer));
        let mut image_scaling_node = ImageScalingNode::new(Some(&change_observer));
        let mut preproccessing_node = PreproccessingNode::new(Some(&change_observer));
        let mut debug = DebugNode::new(Some(&change_observer));

        //connect(webcam.output.clone(), image_scaling_node.image.clone());
        connect(image_value.output.clone(), image_scaling_node.image.clone());
        connect(scaling_config_value.output.clone(), image_scaling_node.input_scaling_config.clone());
        connect(image_scaling_node.output.clone(), preproccessing_node.input.clone());
        connect(preproccessing_node.output.clone(), debug.input.clone());

        let _ = image_value.on_ready();
        let _ = scaling_config_value.on_ready();

        let rescaled_image = image_scaling_node.on_update();


        Ok(assert!(rescaled_image.is_ok()))

    }
    
}