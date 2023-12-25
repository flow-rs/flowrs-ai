mod nodes {
    use flowrs_ai::{max_output_value::MaxOutputNode};
    use flowrs::{node::{ChangeObserver, Node}, connection::{connect, Edge}};
    use ndarray::Array1;
    #[test]
    fn should_get_max_output() -> Result<(), anyhow::Error> {
        //given
        //let classes = vec!["one".to_string(), "two".to_string(), "three".to_string()];
        let classes = "one\ntwo\nthree".as_bytes().to_vec();
        let tensor = Array1::from_vec(vec![0.1, 0.5, 0.2]).into_dyn();
        let change_observer = ChangeObserver::new();
        let mut max_output_node = MaxOutputNode::new(Some(&change_observer));
        let mock_output = Edge::new();
        connect(max_output_node.output_class.clone(), mock_output.clone());
        // when
        let _ = max_output_node.input_classes.send(classes.clone());
        let _ = max_output_node.on_update();
        let _ = max_output_node.output_tensor.send(tensor.clone());
        let _ = max_output_node.on_update(); 
        // then
        let actual: String = mock_output.next()?.into();
        Ok(assert!(actual == "two".to_string()))
    }
}