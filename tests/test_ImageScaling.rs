mod nodes {
    use flowrs::{node::{ChangeObserver, Node}, connection::connect};
    use flowrs_std::value::ValueNode;
    use flowrs_ai::ImageScalingNode::{ImageScalingNode, ScalingConfig};
    use std::{env};
    use image::{DynamicImage};


    #[test]
    fn test_scaling() -> Result<(), anyhow::Error> {
        let change_observer: ChangeObserver = ChangeObserver::new();
        // getting the image to resize
        let image_path = env::current_dir()
        .expect("Failed to obtain current directory")
        .join("src/example_pic/crosswalk.jpg");
        let img = image::open(image_path).expect("Failed to open image");
        // creating scaling config
        let scaling_config = ScalingConfig{
            width: 224,
            height: 224,
        };
        // creating flow and test functionality

        let image_value = ValueNode::new(img, Some(&change_observer));
        let scaling_config_value = ValueNode::new(scaling_config, Some(&change_observer));
        let mut image_scaling_node = ImageScalingNode::new(Some(&change_observer));

        connect(scaling_config_value.output.clone(), image_scaling_node.input_scaling_config.clone());
        connect(image_value.output.clone(), image_scaling_node.image.clone());

        let _ = image_value.on_ready();
        let _ = scaling_config_value.on_ready();

        let result = image_scaling_node.on_update();

        let _ = image_value.on_ready();

        let result2 = image_scaling_node.on_update();


        Ok(assert!(result.is_ok()));
        Ok(assert!(result2.is_ok()))

    }
    
}