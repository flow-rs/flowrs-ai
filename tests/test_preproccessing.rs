mod nodes {
    use flowrs_ai::preproccessing::PreProccessingNode;
    use flowrs::{node::{ChangeObserver, Node}};

    #[test]
    fn should_run_model() -> Result<(), anyhow::Error> {
        let change_observer: ChangeObserver = ChangeObserver::new();  
        
        let mut model_node = PreProccessingNode::new(Some(&change_observer));

        let result = model_node.on_update();

        Ok(assert!(result.is_ok()))
    }
    
}