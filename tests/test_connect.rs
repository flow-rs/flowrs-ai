
#[cfg(test)]
mod nodes {
    use std::{sync::mpsc::channel, thread, time::Duration, rc::Rc, any::Any};

    use flowrs::{
        connection::{connect, Output},
        exec::{
            execution::{Executor, StandardExecutor},
            node_updater::{MultiThreadedNodeUpdater, NodeUpdater},
        },
        flow_impl::Flow,
        node::ChangeObserver,
        sched::round_robin::RoundRobinScheduler,
    };

    use flowrs_ai::{csv2arrayn::{CSVToArrayNNode, self},
                    dbscan::{DbscanNode, DbscanConfig},
                    diffusionmap::{DiffusionMapNode, self},
                    kmeans::KmeansNode,
                    maxabsscale::{MaxAbsScleNode, self},
                    minmaxscale::{MinMaxScaleNode, self},
                    pca::PCANode,
                    standardscale::{StandardscaleNode, self}, convertndarray2datasetbase::ConvertNdarray2DatasetBase, l1normscaler::L1NormscalerNode,
                    };
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
        let test_data_input = String::from("Feate1,Feature2,Feature3\n1,2,3\n4,5,6\n7,8,9");
        /*let test_config_input = CSVToDatasetBaseConfig{
            separator: b',',
            has_feature_names: true
        };*/


        let value_node = ValueNode::new(
            test_input,
            Some(&change_observer),
        );
        let csv2arrayn_node: CSVToArrayNNode<> = CSVToArrayNNode::new(Some(&change_observer));
        let dbscan_node: DbscanNode<> = DbscanNode::new(Some(&change_observer));
        let diffusionmap_node: DiffusionMapNode<> = DiffusionMapNode::new(Some(&change_observer));
        let kmeans_node: KmeansNode<> = KmeansNode::new(Some(&change_observer));
        let maxabsscale_node: MaxAbsScleNode<> = MaxAbsScleNode::new(Some(&change_observer));
        let minmaxscale_node: MinMaxScaleNode<> = MinMaxScaleNode::new(Some(&change_observer));
        //let l2normscaler_node: L2NormscalerNode<> = L2NormscalerNode::new(Some(&change_observer));
        let pca_node: PCANode<> = PCANode::new(Some(&change_observer));
        //let pca_node: L2NormscalerNode<> = L2NormscalerNode::new(Some(&change_observer));
        let standardscale_node: StandardscaleNode<> = StandardscaleNode::new(Some(&change_observer));
        //let tsne_node: TsneNode<> = TsneNode::new(Some(&change_observer));
        let debug_node = DebugNode::<DatasetBase<ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>, ArrayBase<OwnedRepr<()>, Dim<[usize; 1]>>>>::new(Some(&change_observer));
        let convertndarray2datasetbase: ConvertNdarray2DatasetBase<> = ConvertNdarray2DatasetBase::new(Some(&change_observer));

/* 
        connect(value_node.output.clone(), csv2arrayn_node.input.clone());
        connect(csv2arrayn_node.output.clone(), l2normscaler_node.input.clone());
        connect(l2normscaler_node.output.clone(), pca_node.input.clone());
        connect(pca_node.output.clone(), kmeans_node.input.clone());
        connect(kmeans_node.output.clone(), debug_node.input.clone());
*/

        connect(value_node.output.clone(), csv2arrayn_node.input.clone());
        connect(csv2arrayn_node.output.clone(), convertndarray2datasetbase.input.clone());
        connect(convertndarray2datasetbase.output.clone(), debug_node.input.clone());
        //connect(convertndarray2datasetbase.output.clone(), pca_node.input.clone());
        //connect(pca_node.output.clone(), debug_node.input.clone());


        let mut flow: Flow = Flow::new_empty();
     /*   flow.add_node(value_node);
        flow.add_node(csv2arrayn_node);
        flow.add_node(l2normscaler_node);
        flow.add_node(kmeans_node);
        flow.add_node(debug_node);
 */
        flow.add_node(value_node);
        flow.add_node(convertndarray2datasetbase);
        flow.add_node(pca_node);
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

        controller.lock().unwrap().cancel();

        thread_handle.join().unwrap();

        let num_iters = receiver.iter().count();

        let asserted_num_iters = sleep_seconds / timer_interval_seconds;

        assert!(asserted_num_iters.abs_diff(num_iters as u64) <= 1);
    }

    #[test]
    fn csv_array_dataset_standard_diffmap_dbscan() {
        let change_observer: ChangeObserver = ChangeObserver::new();

        let test_input = String::from("Feature1,Feature2,Freature3,Feature4\n1.0,2.0,3.0,4.0\n3.0,4.0,5.0,6.0\n5.0,6.0,7.0,8.0\n7.0,4.0,1.0,9.0");

        // Input Node
        let value_node = ValueNode::new(
            test_input,
            Some(&change_observer),
        );

        // Config Nodes
        let test_config_input = DbscanConfig{
            min_points: 2,
            tolerance: 0.5
        };
        let dbscan_config_node = ValueNode::new(
            test_config_input,
            Some(&change_observer),
        );

        let csv2arrayn_node: CSVToArrayNNode<> = CSVToArrayNNode::new(Some(&change_observer));
        let convertndarray2datasetbase: ConvertNdarray2DatasetBase<> = ConvertNdarray2DatasetBase::new(Some(&change_observer));
        let standardscale_node: StandardscaleNode<> = StandardscaleNode::new(Some(&change_observer));
        let diffusionmap_node: DiffusionMapNode<> = DiffusionMapNode::new(Some(&change_observer));
        let dbscan_node: DbscanNode<> = DbscanNode::new(Some(&change_observer));
        let debug_node = DebugNode::<DatasetBase<ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>, ArrayBase<OwnedRepr<Option<usize>>, Dim<[usize; 1]>>>>::new(Some(&change_observer));

        // Connections
        connect(value_node.output.clone(), csv2arrayn_node.input.clone());
        connect(csv2arrayn_node.output.clone(), convertndarray2datasetbase.input.clone());
        connect(convertndarray2datasetbase.output.clone(), standardscale_node.input.clone());
        connect(standardscale_node.output.clone(), diffusionmap_node.input.clone());
        connect(diffusionmap_node.output.clone(), dbscan_node.dataset_input.clone());
        connect(dbscan_config_node.output.clone(), dbscan_node.config_input.clone());
        connect(dbscan_node.output.clone(), debug_node.input.clone());

        // Flow
        let mut flow: Flow = Flow::new_empty();
        flow.add_node(value_node);
        flow.add_node(csv2arrayn_node);
        flow.add_node(convertndarray2datasetbase);
        flow.add_node(standardscale_node);
        flow.add_node(diffusionmap_node);
        flow.add_node(dbscan_node);
        flow.add_node(dbscan_config_node);
        flow.add_node(debug_node);


        // Controller
        let (controller_sender, controller_receiver) = channel();
        let thread_handle = thread::spawn(move || {
            let mut executor = StandardExecutor::new(change_observer);

            controller_sender
                .send(executor.controller())
                .expect("Controller sender cannot send.");

            controller_receiver
                .recv()
                .expect("JW failed.");

            executor
                .run(flow, RoundRobinScheduler::new(), MultiThreadedNodeUpdater::new(4))
                .expect("Run failed.");
        });
        
        thread::sleep(Duration::from_secs(10));

        assert!(true);

    }

    #[test]
    fn csv_array_dataset_l1_pca_kmeans() {
        let change_observer: ChangeObserver = ChangeObserver::new();

        let test_input = String::from("Feature1,Feature2,Freature3,Feature4\n1.0,2.0,3.0,4.0\n3.0,4.0,5.0,6.0\n5.0,6.0,7.0,8.0\n7.0,4.0,1.0,9.0");

        // Nodes
        let value_node = ValueNode::new(
            test_input,
            Some(&change_observer),
        );

        let csv2arrayn_node: CSVToArrayNNode<> = CSVToArrayNNode::new(Some(&change_observer));
        let convertndarray2datasetbase: ConvertNdarray2DatasetBase<> = ConvertNdarray2DatasetBase::new(Some(&change_observer));
        let l1_node: L1NormscalerNode<> = L1NormscalerNode::new(Some(&change_observer));
        let pca_node: PCANode<> = PCANode::new(Some(&change_observer));
        let kmeans_node: KmeansNode<> = KmeansNode::new(Some(&change_observer));
        let debug_node = DebugNode::<DatasetBase<ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>, ArrayBase<OwnedRepr<usize>, Dim<[usize; 1]>>>>::new(Some(&change_observer));

        // Connections
        connect(value_node.output.clone(), csv2arrayn_node.input.clone());
        connect(csv2arrayn_node.output.clone(), convertndarray2datasetbase.input.clone());
        connect(convertndarray2datasetbase.output.clone(), l1_node.input.clone());
        connect(l1_node.output.clone(), pca_node.dataset_input.clone());
        connect(pca_node.output.clone(), kmeans_node.input.clone());
        connect(kmeans_node.output.clone(), debug_node.input.clone());

        // Flow
        let mut flow: Flow = Flow::new_empty();
        flow.add_node(value_node);
        flow.add_node(csv2arrayn_node);
        flow.add_node(convertndarray2datasetbase);
        flow.add_node(l1_node);
        flow.add_node(pca_node);
        flow.add_node(kmeans_node);
        flow.add_node(debug_node);


        // Controller
        let (controller_sender, controller_receiver) = channel();
        let thread_handle = thread::spawn(move || {
            let mut executor = StandardExecutor::new(change_observer);

            controller_sender
                .send(executor.controller())
                .expect("Controller sender cannot send.");

            controller_receiver
                .recv()
                .expect("JW failed.");

            executor
                .run(flow, RoundRobinScheduler::new(), MultiThreadedNodeUpdater::new(4))
                .expect("Run failed.");
        });
        
        thread::sleep(Duration::from_secs(4));

        assert!(true);

    }
}
