mod nodes {

    use flowrs_ai::model::{ModelNode, ModelConfig};
    use flowrs::{node::{ChangeObserver, Node}, connection::{connect, Edge}};

    use ndarray::{ArrayD, s, IxDyn};
    use std::{env};
    use image::{imageops::FilterType, ImageBuffer, Pixel, Rgb};

    #[test]
    fn should_run_model() -> Result<(), anyhow::Error> {
        // given
        let model_config = ModelConfig {
            model_path: "src/models/opt-squeeze.onnx".to_string(),
            model_base64: "".to_string(),
        };
        
        let shape = [1, 3, 224, 224];
        let model_input = ArrayD::<f32>::zeros(IxDyn(&shape));

        let change_observer: ChangeObserver = ChangeObserver::new();  

        let mut model_node = ModelNode::new(Some(&change_observer));
        let mock_output = Edge::new();
        connect(model_node.output.clone(), mock_output.clone());
        // when
        let _ = model_node.input_model_config.send(model_config.clone());
        let _ = model_node.on_update();
        let _ = model_node.model_input.send(model_input.clone());
        let _ = model_node.on_update();
        // then
        let actual = mock_output.next();
        Ok(assert!(actual.is_ok()))
    }

    #[test]
    fn should_load_model_path() -> Result<(), anyhow::Error> {
        let path = "src/models/opt-squeeze.onnx";
        let model_config = ModelConfig {
            model_path: path.to_string(),
            model_base64: "".to_string(),
        };
        let change_observer: ChangeObserver = ChangeObserver::new();  
        let mut model_node = ModelNode::new(Some(&change_observer));
        let _ = model_node.input_model_config.send(model_config.clone());
        let _ = model_node.on_update();
        Ok(assert!(path == model_node.model_config.unwrap().model_path))
    }
}

