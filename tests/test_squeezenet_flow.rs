#[cfg(test)]
mod nodes {
    use flowrs::{node::{ChangeObserver, Node}, connection::connect};
    use flowrs_ai::{image_scaling::{ImageScalingNode, ScalingConfig}, array_reshape::{ArrayReshapeNode, ArrayReshapeNodeConfig}};
    use flowrs_ai::normalize::NormalizeNode;
    use flowrs_ai::model::{ModelNode, ModelConfig};
    use std::{env, path::Path, fs::File, io::Read};
    use flowrs::connection::Edge;
    use flowrs_ai::{max_output_value::MaxOutputNode};
    use flowrs_img::transform::ImageToArray3Node;

    #[test]
    fn test_squeezenet_flow() -> Result<(), anyhow::Error> {
        let change_observer: ChangeObserver = ChangeObserver::new();

        // creating config
        let scaling_config = ScalingConfig{
            width: 224,
            height: 224,
        };
        let reshape_conifg = ArrayReshapeNodeConfig{
            dimension: vec![1, 3, 224, 224],
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
        let mut image_scaling_node = ImageScalingNode::new(Some(&change_observer));
        let mut image_to_array3 = ImageToArray3Node::<f32>::new(Some(&change_observer));
        let mut array_reshape = ArrayReshapeNode::new(Some(&change_observer));
        let mut preproccessing_node = NormalizeNode::new(Some(&change_observer));
        let mut model_node = ModelNode::new(Some(&change_observer));
        let mut post_processing = MaxOutputNode::new(Some(&change_observer));
        // get classes from model
        let mock_output = Edge::new();
        connect(image_scaling_node.output.clone(), image_to_array3.input.clone());
        connect(image_to_array3.output.clone(), array_reshape.array_input.clone());
        connect(array_reshape.array_output.clone(), preproccessing_node.input.clone());
        connect(preproccessing_node.output.clone(), model_node.model_input.clone());
        connect(model_node.output.clone(), post_processing.output_tensor.clone());
        connect(post_processing.output_class.clone(), mock_output.clone());
        let _ = model_node.input_model_config.send(model_config);
        let _ = post_processing.input_classes.send(labels_file);

        let _ = array_reshape.config_input.send(reshape_conifg);
        let _ = image_scaling_node.input_scaling_config.send(scaling_config);
        let _ = image_scaling_node.image.send(img);
        let _ = image_scaling_node.on_update();
        let _ = image_to_array3.on_update();
        let _ = array_reshape.on_update();
        let _ = preproccessing_node.on_update();
        let _ = model_node.on_update();

        let _ = post_processing.on_update();

        let actual = mock_output.next()?;

        Ok(assert!(actual == "n02051845 pelican"))

    }
    
}