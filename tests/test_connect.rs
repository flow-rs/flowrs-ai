#[cfg(test)]
use flowrs::node::{ChangeObserver, Node};
use flowrs_ai::{dbscan::{DbscanConfig, DbscanNode}, diffusionmap::{DiffusionMapConfig, DiffusionMapNode}, csv2ndarray::{CSVToNdarrayNode, CSVToNdarrayConfig}, ndarray2dataset::NdarrayToDatasetNode, standardscaler::StandardScalerNode};
use flowrs_std::value::ValueNode;
use linfa::DatasetBase;
use log::info;
use ndarray::{ArrayBase, OwnedRepr, Dim, Array1, array};


#[test]
fn test_csv_ndarray_dataset_s1_diff_dbscan() {
    env_logger::Builder::from_default_env()
    .filter_level(log::LevelFilter::Info)
    .filter_level(log::LevelFilter::Debug)
    .init();

    let change_observer: ChangeObserver = ChangeObserver::new();

    // Input
    let test_input = String::from("Feature1,Feature2,Freature3,Feature4\n1.0,2.0,3.0,4.0\n3.0,4.0,5.0,6.0\n5.0,6.0,7.0,8.0\n7.0,4.0,1.0,9.0");

    // Config Nodes
    let csv_config_input = CSVToNdarrayConfig{
        separator: b',',
        has_feature_names: true
    };
    let mut csv_config_node = ValueNode::new(
        csv_config_input,
        Some(&change_observer),
    );

    let dbscan_config_input = DbscanConfig{
        min_points: 2,
        tolerance: 0.5
    };
    let mut dbscan_config_node = ValueNode::new(
        dbscan_config_input,
        Some(&change_observer),
    );
    let diffusionmap_config_input = DiffusionMapConfig{
        embedding_size: 2,
        steps: 1
    };
    let mut diffusionmap_config_node = ValueNode::new(
        diffusionmap_config_input,
        Some(&change_observer),
    );

    // Nodes
    let mut csv2arrayn_node: CSVToNdarrayNode<f64> = CSVToNdarrayNode::new(Some(&change_observer));
    let mut ndarray_to_dataset_node: NdarrayToDatasetNode<f64> = NdarrayToDatasetNode::new(Some(&change_observer));
    let mut standardscale_node: StandardScalerNode<f64> = StandardScalerNode::new(Some(&change_observer));
    let mut diffusionmap_node: DiffusionMapNode<f64> = DiffusionMapNode::new(Some(&change_observer));
    let mut dbscan_node: DbscanNode<f64> = DbscanNode::new(Some(&change_observer));
    let mock_output = flowrs::connection::Edge::new();

    flowrs::connection::connect(csv_config_node.output.clone(), csv2arrayn_node.config_input.clone());
    flowrs::connection::connect(dbscan_config_node.output.clone(), dbscan_node.config_input.clone());
    flowrs::connection::connect(diffusionmap_config_node.output.clone(), diffusionmap_node.config_input.clone());

    flowrs::connection::connect(csv2arrayn_node.output.clone(), ndarray_to_dataset_node.data_input.clone());
    flowrs::connection::connect(ndarray_to_dataset_node.output.clone(), standardscale_node.data_input.clone());
    flowrs::connection::connect(standardscale_node.output.clone(), diffusionmap_node.data_input.clone());
    flowrs::connection::connect(diffusionmap_node.output.clone(), dbscan_node.data_input.clone());
    flowrs::connection::connect(dbscan_node.output.clone(), mock_output.clone());

    let _ = csv2arrayn_node.data_input.send(test_input.clone());
    let _ = csv_config_node.on_update();
    let _ = dbscan_config_node.on_update();
    let _ = diffusionmap_config_node.on_update();

    let _ = csv2arrayn_node.on_update();
    let _ = ndarray_to_dataset_node.on_update();
    let _ = standardscale_node.on_update();
    let _ = diffusionmap_node.on_update();
    let _ = dbscan_node.on_update();

    let actual: Result<DatasetBase<ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>, ArrayBase<OwnedRepr<Option<usize>>, Dim<[usize; 1]>>>, flowrs::node::ReceiveError>= mock_output.next();

    match actual {
        Ok(data) => {
            let expected_targets: Array1<Option<usize>> = array![None, Some(0), None, Some(0)];
            let expected_records = array![[-0.6079560992874712, 0.4689266237227759],
            [-0.3756132537444994, -0.0300442766363188],
            [-0.42679034838888313, -0.6319864344228838],
            [0.06980139370290697, 0.024295850639015487]];

            info!("{:?}", data);
            assert!(expected_targets == data.targets);
            assert!(expected_records == data.records);
        }
        Err(err) => {
            println!("ReceiveError: {}", err);
        }
    }
}
 