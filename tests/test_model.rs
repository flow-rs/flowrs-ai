mod nodes {
    use flowrs_ai::model::ModelNode;
    use flowrs::{node::{ChangeObserver, Node}};

    #[test]
    fn should_run_model() -> Result<(), anyhow::Error> {
        let change_observer: ChangeObserver = ChangeObserver::new();  
        
        let mut snd = ModelNode::new(Some(&change_observer));

        snd.on_update()?;

        Ok(assert!(true))
    }
    
}