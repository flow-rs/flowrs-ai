mod nodes {
    use flowrs_ai::PreproccessingNode::PreproccessingNode;
    use flowrs::{node::{ChangeObserver, Node}};
    use std::env;
    use image::{imageops::FilterType, DynamicImage};
    use flowrs_std::value::ValueNode;
    use flowrs::connection::connect;
    use flowrs::connection::Edge;

    #[test]
    fn should_run_model() -> Result<(), anyhow::Error> {
        let change_observer: ChangeObserver = ChangeObserver::new();  
        let image_path = env::current_dir()
        .expect("Failed to obtain current directory")
        .join("src/example_pic/crosswalk.jpg");
        let img = image::open(image_path).expect("Failed to open image");

        let image = img.resize_exact(
            224,
            224,
            FilterType::CatmullRom,
        );

        let image_value = ValueNode::new(image, Some(&change_observer));
        let mock_output = Edge::new();
        let mut preproccessing_node = PreproccessingNode::new(Some(&change_observer));
        connect(image_value.output.clone(), preproccessing_node.input.clone());
        connect(preproccessing_node.output.clone(), mock_output.clone());

        let _ = image_value.on_ready();
        let result = preproccessing_node.on_update();
        
       Ok(assert!(result.is_ok()))
    }
    
}