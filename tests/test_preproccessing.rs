mod nodes {
    use flowrs_ai::PreproccessingNode::PreproccessingNode;
    use flowrs::{node::{ChangeObserver, Node}};
    use std::env;
    use image::{imageops::FilterType};
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

        let mock_output = Edge::new();
        let mut preproccessing_node = PreproccessingNode::new(Some(&change_observer));
        connect(preproccessing_node.output.clone(), mock_output.clone());

        let _ = preproccessing_node.input.send(image);
        let result = preproccessing_node.on_update();
        
       Ok(assert!(result.is_ok()))
    }
    
}