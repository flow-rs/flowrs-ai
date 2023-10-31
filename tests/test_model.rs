mod nodes {
    use std::borrow::Borrow;

    use flowrs_ai::model::{ModelNode, ModelConfig};
    use flowrs::{node::{ChangeObserver, Node}, connection::connect};
    use flowrs_std::value::ValueNode;

    #[test]
    fn should_run_model() -> Result<(), anyhow::Error> {
        let model_config = ModelConfig {
            model_path: "src/models/opt-squeeze.onnx".to_string(),
        };
        let change_observer: ChangeObserver = ChangeObserver::new();  
        let value_node = ValueNode::new(model_config, Some(&change_observer));
        let mut model_node = ModelNode::new(Some(&change_observer));

        connect(value_node.output.clone(), model_node.input.clone());
        let _ = value_node.on_ready();

        let result = model_node.on_update();

        Ok(assert!(result.is_ok()))
    }

    #[test]
    fn should_load_model_path() -> Result<(), anyhow::Error> {
        let path = "src/models/opt-squeeze.onnx";
        let model_config = ModelConfig {
            model_path: path.to_string(),
        };
        let change_observer: ChangeObserver = ChangeObserver::new();  
        let value_node = ValueNode::new(model_config, Some(&change_observer));
        let mut model_node = ModelNode::new(Some(&change_observer));
        connect(value_node.output.clone(), model_node.input.clone());
        let _ = value_node.on_ready();
        let _ = model_node.on_update();
        Ok(assert!(path == model_node.model_config.unwrap().model_path))
    }
    
}