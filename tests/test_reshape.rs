#[cfg(test)]
mod nodes {
    use flowrs::connection::{connect, Edge};
    use flowrs::node::{ChangeObserver, Node, ReceiveError};
    use flowrs_ai::ArrayReshapeNode::{ArrayReshapeNode, ArrayReshapeNodeConfig};
    use ndarray::{ArrayD, IxDyn};

    #[test]
    fn should_return_reshaped_array() -> Result<(), ReceiveError> {
        let change_observer: ChangeObserver = ChangeObserver::new();
        let array_shape_config = ArrayReshapeNodeConfig {
            dimension: vec![1, 2, 3],
        };
        let mut reshape_node = ArrayReshapeNode::new(Some(&change_observer));

        let mock_output = Edge::new();
        connect(reshape_node.array_output.clone(), mock_output.clone());

        let init_retsult = reshape_node.on_init();
        if let Err(err) = init_retsult {
            return Err(ReceiveError::Other(err.into()));
        }

        let _ = reshape_node
            .array_input
            .send(ArrayD::<f32>::zeros(IxDyn(&[3, 2, 1])));
        let _ = reshape_node.config_input.send(array_shape_config);

        let update_result = reshape_node.on_update();
        if let Err(err) = update_result {
            return Err(ReceiveError::Other(err.into()));
        }

        let result_array = mock_output.next()?;
        let result_shape = result_array.shape();

        assert_eq!(1, result_shape[0]);
        assert_eq!(2, result_shape[1]);
        assert_eq!(3, result_shape[2]);

        let shutdown_result = reshape_node.on_shutdown();
        if let Err(err) = shutdown_result {
            return Err(ReceiveError::Other(err.into()));
        };
        Ok(())
    }
}
