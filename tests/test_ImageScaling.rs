mod nodes {
    use flowrs::{node::{ChangeObserver, Node}, connection::connect};
    use flowrs_ai::image_scaling::{ImageScalingNode, ScalingConfig};
    use std::{env};
    use flowrs::connection::Edge;


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

        let mut image_scaling_node = ImageScalingNode::new(Some(&change_observer));
        let mock_output = Edge::new();

        connect(image_scaling_node.output.clone(), mock_output.clone());

        let _ = image_scaling_node.input_scaling_config.send(scaling_config);
        let _ = image_scaling_node.image.send(img);

        let result = image_scaling_node.on_update();

        Ok(assert!(result.is_ok()))

    }
    
}