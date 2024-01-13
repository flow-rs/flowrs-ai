#[cfg(test)]
mod tests {
    use csv::{WriterBuilder, ReaderBuilder};
    use flowrs_ai::{pca::{PCAConfig, PCANode, self}, kmeans::{KmeansConfig, KmeansNode, self}, csv2dataset::{CSVToDatasetConfig, CSVToDatasetNode}, diffusionmap::{DiffusionMapNode, self}, standardscaler::{StandardScalerNode, self}, l1normscaler::{self, L1NormScalerNode}, maxabsscaler::{self, MaxAbsSclerNode}, maxnormscaler::{self, MaxNormScalerNode}, minmaxrangescaler::{self, MinMaxRangeScalerNode}, minmaxscaler::{self, MinMaxScalerNode}, l2normscaler::L2NormScalerNode, tsne::TsneNode, dbscan::DbscanNode};
    use linfa::{dataset, DatasetBase};
    use log::info;
    use ndarray::{array, Array2};
    use std::{time::{Instant, Duration}, fs};

    use ndarray_csv::{Array2Reader, Array2Writer};
    use std::error::Error;
    use std::fs::File;

    use flowrs::node::{ChangeObserver, Node, UpdateError};
    use flowrs_std::value::ValueNode;


    #[test]
    fn test_result_evaluation() -> Result<(), UpdateError> {
        let change_observer: ChangeObserver = ChangeObserver::new();
    
        // Input
        let test_input = fs::read_to_string("tests/test_iris.csv").expect("Should have been able to read the file");
    
        // Config Nodes
        let converter_config = CSVToDatasetConfig{
            separator: b',',
            has_feature_names: true
        };

        let converter_config_node = ValueNode::new(
            converter_config,
            Some(&change_observer),
        );

        let dim_red_config = PCAConfig{
            embedding_size: 2,
        };

        let dim_red_config_node = ValueNode::new(
            dim_red_config,
            Some(&change_observer),
        );
    
        let clustering_config = KmeansConfig{
            num_of_dim: 3,
            max_n_iterations: 100,
            tolerance: 0.1
        };

        let clustering_config_node = ValueNode::new(
            clustering_config,
            Some(&change_observer),
        );

        // Nodes
        let mut converter_node: CSVToDatasetNode<f32> = CSVToDatasetNode::new(Some(&change_observer));
        let mut scaling_node: StandardScalerNode<f32> = StandardScalerNode::new(Some(&change_observer));
        let mut dim_red_node: PCANode<f32> = PCANode::new(Some(&change_observer));
        let mut clustering_node: KmeansNode<f32> = KmeansNode::new(Some(&change_observer));
        let mock_output = flowrs::connection::Edge::new();
        
        // Connections
        flowrs::connection::connect(converter_config_node.output.clone(), converter_node.config_input.clone());
        flowrs::connection::connect(dim_red_config_node.output.clone(), dim_red_node.config_input.clone());
        flowrs::connection::connect(clustering_config_node.output.clone(), clustering_node.config_input.clone());
    
        flowrs::connection::connect(converter_node.output.clone(), scaling_node.data_input.clone());
        flowrs::connection::connect(scaling_node.output.clone(), dim_red_node.data_input.clone());
        flowrs::connection::connect(dim_red_node.output.clone(), clustering_node.data_input.clone());
        flowrs::connection::connect(clustering_node.output.clone(), mock_output.clone());
        
        // Node Setup Updating
        let _ = converter_config_node.on_ready();
        let _ = dim_red_config_node.on_ready();
        let _ = clustering_config_node.on_ready();

        let _ = converter_node.on_update();
        let _ = scaling_node.on_update();
        let _ = dim_red_node.on_update();
        let _ = clustering_node.on_update();
                
        // Node Datastream Updating 
        let _ = converter_node.data_input.send(test_input.clone());

        let _ = converter_node.on_update();
        let _ = scaling_node.on_update();
        let _ = dim_red_node.on_update();
        let _ = clustering_node.on_update();
    
        let result = mock_output.next()?;
        
        // Write To File
        let records_file = File::create("tests/result_records.csv").map_err(|e| UpdateError::Other(e.into()))?;
        let targets_file = File::create("tests/result_targets.csv").map_err(|e| UpdateError::Other(e.into()))?;
        let mut records_writer = WriterBuilder::new().has_headers(false).from_writer(records_file);
        let mut targets_writer = WriterBuilder::new().has_headers(false).from_writer(targets_file);
        let targets = result.targets.clone().into_shape((result.targets.clone().len(), 1)).map_err(|e| UpdateError::Other(e.into()))?;
        records_writer.serialize_array2(&result.records.clone()).map_err(|e| UpdateError::Other(e.into()))?;
        targets_writer.serialize_array2(&targets).map_err(|e| UpdateError::Other(e.into()))?;

        Ok(())
    }


    #[test]
    fn test_runtime_evaluation() -> Result<(), UpdateError>{
        const N: usize = 10;

        let change_observer: ChangeObserver = ChangeObserver::new();
        
        // Setup
        let test_input = fs::read_to_string("tests/test_iris.csv").expect("Should have been able to read the file");

        let converter_config = CSVToDatasetConfig{
            separator: b',',
            has_feature_names: true
        };
        let converter_config_node = ValueNode::new(
            converter_config,
            Some(&change_observer),
        );

        let mut converter_node: CSVToDatasetNode<f64> = CSVToDatasetNode::new(Some(&change_observer));
        let converter_output = flowrs::connection::Edge::new();

        flowrs::connection::connect(converter_config_node.output.clone(), converter_node.config_input.clone());
        flowrs::connection::connect(converter_node.output.clone(), converter_output.clone()); 

        let _ = converter_config_node.on_ready();
        let _ = converter_node.data_input.send(test_input.clone());
        let _ = converter_node.on_update();

        let iris_dataset = converter_output.next()?;

        // Scaling evaluation
        println!("Scaling runtime evaluation");

        let mut l1normscaler_node: L1NormScalerNode<f64> = L1NormScalerNode::new(Some(&change_observer));
        let mut l2normscaler_node: L2NormScalerNode<f64> = L2NormScalerNode::new(Some(&change_observer));
        let mut maxabsscaler_node: MaxAbsSclerNode<f64> = MaxAbsSclerNode::new(Some(&change_observer));
        let mut maxnormscaler_node: MaxNormScalerNode<f64> = MaxNormScalerNode::new(Some(&change_observer));
        let mut minmaxrangescaler_node: MinMaxRangeScalerNode<f64> = MinMaxRangeScalerNode::new(Some(&change_observer));
        let mut minmaxscaler_node: MinMaxScalerNode<f64> = MinMaxScalerNode::new(Some(&change_observer));
        let mut standardscaler_node: StandardScalerNode<f64> = StandardScalerNode::new(Some(&change_observer));

        let mock_output = flowrs::connection::Edge::new();
 
        flowrs::connection::connect(l1normscaler_node.output.clone(), mock_output.clone());
        flowrs::connection::connect(l2normscaler_node.output.clone(), mock_output.clone());
        flowrs::connection::connect(maxabsscaler_node.output.clone(), mock_output.clone());
        flowrs::connection::connect(maxnormscaler_node.output.clone(), mock_output.clone());
        flowrs::connection::connect(minmaxrangescaler_node.output.clone(), mock_output.clone());
        flowrs::connection::connect(minmaxscaler_node.output.clone(), mock_output.clone());
        flowrs::connection::connect(standardscaler_node.output.clone(), mock_output.clone());
                
        for _ in 1..=N {
            let _ = l1normscaler_node.data_input.send(iris_dataset.clone());
            let _ = l2normscaler_node.data_input.send(iris_dataset.clone());
            let _ = maxabsscaler_node.data_input.send(iris_dataset.clone());
            let _ = maxnormscaler_node.data_input.send(iris_dataset.clone());
            let _ = minmaxrangescaler_node.data_input.send(iris_dataset.clone());
            let _ = minmaxscaler_node.data_input.send(iris_dataset.clone());
            let _ = standardscaler_node.data_input.send(iris_dataset.clone());

            let _ = l1normscaler_node.on_update();
            let _ = l2normscaler_node.on_update();
            let _ = maxabsscaler_node.on_update();
            let _ = maxnormscaler_node.on_update();
            let _ = minmaxrangescaler_node.on_update();
            let _ = minmaxscaler_node.on_update();
            let _ = standardscaler_node.on_update();
        }

        // Dimension_Reduction runtime evaluation
        println!("Dimension_Reduction runtime evaluation");

        let mut diffusionmap_node: DiffusionMapNode<f64> = DiffusionMapNode::new(Some(&change_observer));
        let mut pca_node: PCANode<f64> = PCANode::new(Some(&change_observer));
        let mut tsne_node: TsneNode<f64> = TsneNode::new(Some(&change_observer));

        let mock_output = flowrs::connection::Edge::new();
 
        flowrs::connection::connect(diffusionmap_node.output.clone(), mock_output.clone());
        flowrs::connection::connect(pca_node.output.clone(), mock_output.clone());
        flowrs::connection::connect(tsne_node.output.clone(), mock_output.clone());
                
        for _ in 1..=N {
            let _ = diffusionmap_node.data_input.send(iris_dataset.clone());
            let _ = pca_node.data_input.send(iris_dataset.clone());
            let _ = tsne_node.data_input.send(iris_dataset.clone());

            let _ = diffusionmap_node.on_update();
            let _ = pca_node.on_update();
            let _ = tsne_node.on_update();
        }

        // Clustering runtime evaluation
        println!("Dimesnion_Reduction runtime evaluation");

        let mut dbsan_node: DbscanNode<f64> = DbscanNode::new(Some(&change_observer));
        let mut kmeans_node: KmeansNode<f64> = KmeansNode::new(Some(&change_observer));

        let dbscan_output = flowrs::connection::Edge::new();
        let kmeans_output = flowrs::connection::Edge::new();
 
        flowrs::connection::connect(dbsan_node.output.clone(), dbscan_output.clone());
        flowrs::connection::connect(kmeans_node.output.clone(), kmeans_output.clone());
                
        for _ in 1..=N {
            let _ = dbsan_node.data_input.send(iris_dataset.clone());
            let _ = kmeans_node.data_input.send(iris_dataset.clone());

            let _ = dbsan_node.on_update();
            let _ = kmeans_node.on_update();
        }

        Ok(())
    }
}