
#[cfg(test)]
mod nodes {
    use std::{sync::mpsc::channel, thread, time::Duration};

    use flowrs::{
        connection::connect,
        exec::{
            execution::{Executor, StandardExecutor},
            node_updater::{MultiThreadedNodeUpdater, NodeUpdater, SingleThreadedNodeUpdater},
        },
        flow_impl::Flow,
        node::ChangeObserver,
        sched::round_robin::RoundRobinScheduler,
    };

    use flowrs_ai::{cluster::ClusterNode, scale::ScaleNode, csv_to_array::CSVToArrayNNode, dimred::DimRedNode};
    use flowrs_std::{
        debug::DebugNode,
        timer::{PollTimer, TimerStrategy, WaitTimer},
        value::ValueNode,
    };
    use serde::{Deserialize, Serialize};
    use flowrs_std::vec;
    use ndarray::{Array3, ArrayBase, OwnedRepr, Dim, arr2};
    use anyhow::{anyhow};
    use linfa::prelude::*;
    use ndarray::Array2;
    use ndarray::prelude::*;
    use linfa::traits::{Fit, Predict};
    use linfa_reduction::Pca;
    use linfa_clustering::KMeans;

    fn connect_test_with<
        T: TimerStrategy<bool> + Send + 'static,
        U: NodeUpdater + Drop + Send + 'static,
    >(
        node_updater: U,
        timer: T,
    ) where
        T: Clone + Deserialize<'static> + Serialize,
    {
        let sleep_seconds = 1;
        let timer_interval_seconds = 1;

        let change_observer: ChangeObserver = ChangeObserver::new();
        let (sender, receiver) = channel::<bool>();


        // TEST DATA \/\/\/\/
        // #####################################################################
        let data: Array2<f64> = Array2::from_shape_vec((10, 6), vec![1.1, 2.5, 3.2, 4.6, 5.2, 6.7, 7.8, 8.2, 9.5, 10.3, 11.0, 12.0, 13.0, 14.0, 15.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0])
        .expect("Failed to create data array");
    
        // Define labels as a vector
        let labels: Vec<u32> = vec![0, 1, 2, 4, 5];
    
        // Define feature names as a vector of strings
        let feature_names: Vec<&str> = vec!["Feature1", "Feature2", "Feature3", "a", "b", "c"];
    
        // Create a DatasetBase with data, labels, and feature names
        let dataset = DatasetBase::new(data, labels);
        let dataset = dataset.with_feature_names(feature_names);

        // TEST DATA /\/\/\/\
        // #####################################################################

        let node_1 = ValueNode::new(
            dataset,
            Some(&change_observer),
        );

        

        //let node_2: CSVToArrayNNode<> = CSVToArrayNNode::new(Some(&change_observer));
        //let node_3: ScaleNode<> = ScaleNode::new(Some(&change_observer));
        let node_4: DimRedNode<> = DimRedNode::new(Some(&change_observer));
        let node_5: ClusterNode<> = ClusterNode::new(Some(&change_observer));

        let node_6 = DebugNode::<DatasetBase<ArrayBase<OwnedRepr<f64>, _>, Vec<u32>>>::new(Some(&change_observer));


        /*connect(node_1.output.clone(), node_2.input.clone());
        connect(node_2.output.clone(), node_3.input.clone());
        connect(node_3.output.clone(), node_4.input.clone());
        connect(node_4.output.clone(), node_5.input.clone());
        connect(node_5.output.clone(), node_6.input.clone());

        let mut flow: Flow = Flow::new_empty();
        flow.add_node(node_1);
        flow.add_node(node_2);
        flow.add_node(node_3);
        flow.add_node(node_4);
        flow.add_node(node_5);
        flow.add_node(node_6);*/

        connect(node_1.output.clone(), node_4.input.clone());
        connect(node_4.output.clone(), node_5.input.clone());
        connect(node_5.output.clone(), node_6.input.clone());

        let mut flow: Flow = Flow::new_empty();
        flow.add_node(node_1);
        flow.add_node(node_4);
        flow.add_node(node_5);
        flow.add_node(node_6);

        let (controller_sender, controller_receiver) = channel();
        let thread_handle = thread::spawn(move || {
            let mut executor = StandardExecutor::new(change_observer);

            controller_sender
                .send(executor.controller())
                .expect("Controller sender cannot send.");

            executor
                .run(flow, RoundRobinScheduler::new(), node_updater)
                .expect("Run failed.");
        });

        let controller = controller_receiver.recv().unwrap();

        thread::sleep(Duration::from_secs(sleep_seconds));

        //println!("                                      ----> {:?} CANCEL", std::thread::current().id());

        controller.lock().unwrap().cancel();

        thread_handle.join().unwrap();

        let num_iters = receiver.iter().count();

        let asserted_num_iters = sleep_seconds / timer_interval_seconds;

        //println!("{} {}", num_iters, asserted_num_iters.abs_diff(num_iters as u64));
        assert!(asserted_num_iters.abs_diff(num_iters as u64) <= 1);
    }

    #[test]
    fn test() {
        connect_test_with(MultiThreadedNodeUpdater::new(4), WaitTimer::new(true));

    }
}
