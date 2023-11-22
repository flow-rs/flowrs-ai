
#[cfg(test)]
mod nodes {
    use std::{sync::mpsc::channel, thread, time::Duration};

    use flowrs::{
        connection::connect,
        exec::{
            execution::{Executor, StandardExecutor},
            node_updater::{MultiThreadedNodeUpdater, NodeUpdater},
        },
        flow_impl::Flow,
        node::ChangeObserver,
        sched::round_robin::RoundRobinScheduler,
    };

    use flowrs_ai::{csv2arrayn::{CSVToArrayNNode, self},
                    dbscan::DbscanNode,
                    diffusionmap::{DiffusionMapNode, self},
                    dimred::{DimRedNode, self},
                    kmeans::KmeansNode,
                    maxabsscale::{MaxAbsScleNode, self},
                    minmaxscale::{MinMaxScaleNode, self},
                    l2normscaler::{NormscalerNode, self},
                    pca::PCANode,
                    standardscale::{StandardscaleNode, self},
                    tsne::{TsneNode, self}};
    use flowrs_std::{
        debug::DebugNode,
        timer::{TimerStrategy, WaitTimer},
        value::ValueNode,
    };
    use linfa_clustering::KMeans;
    use serde::{Deserialize, Serialize};
    use ndarray::{Array3, ArrayBase, OwnedRepr, arr2, Dim};
    use linfa::prelude::*;
    use ndarray::Array2;

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

        // TestData
        let test_input = String::from("Feature1,Feature2,Freature3,Feature4\n1.0,2.0,3.0,4.0\n3.0,4.0,5.0,6.0\n5.0,6.0,7.0,8.0\n7.0,4.0,1.0,9.0");

        let value_node = ValueNode::new(
            test_input,
            Some(&change_observer),
        );
        let csv2arrayn_node: CSVToArrayNNode<> = CSVToArrayNNode::new(Some(&change_observer));
        let dbscan_node: DbscanNode<> = DbscanNode::new(Some(&change_observer));
        let diffusionmap_node: DiffusionMapNode<> = DiffusionMapNode::new(Some(&change_observer));
        let dimred_node: DimRedNode<> = DimRedNode::new(Some(&change_observer));
        let kmeans_node: KmeansNode<> = KmeansNode::new(Some(&change_observer));
        let maxabsscale_node: MaxAbsScleNode<> = MaxAbsScleNode::new(Some(&change_observer));
        let minmaxscale_node: MinMaxScaleNode<> = MinMaxScaleNode::new(Some(&change_observer));
        let normscaler_node: NormscalerNode<> = NormscalerNode::new(Some(&change_observer));
        let pca_node: PCANode<> = PCANode::new(Some(&change_observer));
        let standardscale_node: StandardscaleNode<> = StandardscaleNode::new(Some(&change_observer));
        let tsne_node: TsneNode<> = TsneNode::new(Some(&change_observer));
        let debug_node = DebugNode::<DatasetBase<ArrayBase<ndarray::OwnedRepr<f64>, Dim<[usize; 2]>>, ArrayBase<ndarray::OwnedRepr<usize>, Dim<[usize; 1]>>>>::new(Some(&change_observer));

        connect(value_node.output.clone(), csv2arrayn_node.input.clone());
        connect(csv2arrayn_node.output.clone(), normscaler_node.input.clone());
        connect(normscaler_node.output.clone(), pca_node.input.clone());
        connect(pca_node.output.clone(), kmeans_node.input.clone());
        connect(kmeans_node.output.clone(), debug_node.input.clone());

        let mut flow: Flow = Flow::new_empty();
        flow.add_node(value_node);
        flow.add_node(csv2arrayn_node);
        flow.add_node(normscaler_node);
        flow.add_node(pca_node);
        flow.add_node(kmeans_node);
        flow.add_node(debug_node);

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
