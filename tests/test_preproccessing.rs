mod nodes {
    use flowrs_ai::PreproccessingNode::PreproccessingNode;
    use flowrs::{node::{ChangeObserver, Node}};

    #[test]
    fn should_run_model() -> Result<(), anyhow::Error> {
        let change_observer: ChangeObserver = ChangeObserver::new();  
        
        let mut preproccessing_node = PreproccessingNode::new(Some(&change_observer));

        let result = preproccessing_node.on_update();
        
        Ok(assert!(result.is_ok()))
    }
    
}