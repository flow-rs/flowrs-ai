mod nodes {
    use flowrs_ai::{max_output_value::MaxOutputNode};
    use flowrs::{node::{ChangeObserver, Node}, connection::{connect, Edge}};
    use flowrs_std::value::ValueNode;
    use ndarray::Array1;
    #[test]
    fn should_get_max_output() -> Result<(), anyhow::Error> {
        //given
        let classes = vec!["one".to_string(), "two".to_string(), "three".to_string()];
        let tensor = Array1::from_vec(vec![0.1, 0.5, 0.2]);
        let change_observer = ChangeObserver::new();
        let value_node_classes = ValueNode::new(classes, Some(&change_observer));
        let value_node_tensor = ValueNode::new(tensor, Some(&change_observer));
        let mut max_output_node = MaxOutputNode::new(Some(&change_observer));
        let mock_output = Edge::new();
        connect(value_node_classes.output.clone(), max_output_node.input_classes.clone());
        connect(value_node_tensor.output.clone(), max_output_node.output_tensor.clone());
        connect(max_output_node.output_class.clone(), mock_output.clone());
        // when
        let _ = value_node_classes.on_ready();
        let _ = max_output_node.on_update();
        let _ = value_node_tensor.on_ready();
        let _ = max_output_node.on_update(); 
        // then
        let actual: String = mock_output.next()?.into();
        Ok(assert!(actual == "two".to_string()))
    }
}