


mod nodes {

    use flowrs_ai::model::{ModelNode, ModelConfig};
    use flowrs::{node::{ChangeObserver, Node}, connection::{connect, Edge}};
    use futures_executor::block_on;

    use ndarray::{ArrayD, IxDyn};
    use wonnx::{Session, utils::InputTensor};
    use std::{time::Instant, collections::HashMap};

    #[ignore]
    #[test]
    pub fn benchmark_model_node() {
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
        let num_executions = 100;
        // when
        let _ = model_node.input_model_config.send(model_config.clone());
        let _ = model_node.on_update();
        println!("benchmarking ModelNode:");
        println!("Start benchmarks with {} iterations", num_executions);
        println!("...");
        let now = Instant::now();
        for _ in 1..num_executions {
            let _ = model_node.model_input.send(model_input.clone()); 
            let _ = model_node.on_update();
        }
        let elapsed = now.elapsed().as_millis();
        // then
        println!("Result: {}ms per executions \n", elapsed / num_executions);
    }

    #[ignore]
    #[test]
    pub fn benchmark_wonnx() {
        // given
        let model_path = "src/models/opt-squeeze.onnx";
        let shape = [1, 3, 224, 224];
        let model_input = ArrayD::<f32>::zeros(IxDyn(&shape));
        let session = block_on(Session::from_path(model_path)).unwrap();
        let num_executions = 100;
        let mut input_data: HashMap<String, InputTensor> = HashMap::new();
        input_data.insert("data".to_string(), model_input.as_slice().unwrap().into());
        println!("benchmarking WONNX:");
        println!("Start benchmarks with {} iterations", num_executions);
        println!("...");
        // when
        let now = Instant::now();
        for _ in 1..num_executions {
            let _ = block_on(session.run(&input_data));
        }
        let elapsed = now.elapsed().as_millis();
        // then
        println!("Results: {}ms per execution \n", elapsed / num_executions);
    }
}
