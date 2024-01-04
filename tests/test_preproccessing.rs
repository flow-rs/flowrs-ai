mod nodes {
    use flowrs_ai::pre_processing::PreproccessingNode;
    use flowrs::{node::{ChangeObserver, Node}};
    use ndarray::{Array};
    use std::{cmp::Ordering};
    use flowrs::connection::connect;
    use flowrs::connection::Edge;

    #[test]
    fn should_normalize_values() -> Result<(), anyhow::Error> {
        let change_observer: ChangeObserver = ChangeObserver::new();  

         // Specify the shape of the 4D array
         let shape = (2, 2, 2, 2);
         // Initialize a 4D array with specific values
         let array_4d = Array::from_shape_fn(shape, |(_, _, _, _)| { 
            255 as f32
        });

        let mock_output = Edge::new();
        let mut preproccessing_node = PreproccessingNode::new(Some(&change_observer));
        connect(preproccessing_node.output.clone(), mock_output.clone());

        let _ = preproccessing_node.input.send(array_4d.into_dyn());
        let _ = preproccessing_node.on_update();

        let actual: ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::Dim<ndarray::IxDynImpl>> = mock_output.next()?;
        let max_value = actual.iter().max_by(|x, y| x.partial_cmp(y).unwrap_or(Ordering::Equal));
        let min_value = actual.iter().min_by(|x, y| x.partial_cmp(y).unwrap_or(Ordering::Equal));

        Ok(assert!(*max_value.unwrap() <= 1.0 && *min_value.unwrap() >= 0.0))
    }   
}

