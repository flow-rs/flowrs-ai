use std::mem::swap;

use flowrs::{node::{Node, UpdateError, ChangeObserver}, connection::{Input, Output}};
use flowrs::RuntimeConnectable;

use ndarray::{Array2, Array1};
use linfa::dataset::{DatasetBase, self};
use linfa::traits::Transformer;
use ndarray::{ArrayBase, OwnedRepr, Dim, array};
use linfa_clustering::Dbscan;
use linfa::dataset::Labels;
use serde::{Deserialize, Serialize};

#[derive(Clone, Deserialize, Serialize)]
pub struct DbscanConfig {
    pub min_points: usize,
    pub tolerance: f64
}

#[derive(RuntimeConnectable)]
pub struct DbscanNode {
    #[input]
    pub config_input: Input<DbscanConfig>,
    
    #[output]
    pub output: Output<DatasetBase<ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>, ArrayBase<OwnedRepr<Option<usize>>, Dim<[usize; 1]>>>>,

    #[input]
    pub dataset_input: Input<DatasetBase<ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>, ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>>>, 

    input_dataset: Option<DatasetBase<ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>, ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>>>,
}

impl DbscanNode {
    pub fn new(change_observer: Option<&ChangeObserver>) -> Self {
        Self {
            config_input: Input::new(),
            dataset_input: Input::new(),
            output: Output::new(change_observer),
            input_dataset : Option::None
        }
    }
}

impl Node for DbscanNode {
    fn on_update(&mut self) -> Result<(), UpdateError> {
        println!("JW-Debug: DbscanNode got an update!");

        if let Ok(dataset) = self.dataset_input.next() {
            println!("JW-Debug: DbscanNode has received: \n Records: {} \n Targets: {}.", dataset.records, dataset.targets);
            self.input_dataset = Some(dataset);
        }

        if let Ok(config) = self.config_input.next() {
            println!("JW-Debug DbscanNode has received config: {}, {}", config.min_points, config.tolerance);
        }

        if let Some(data) = self.input_dataset.clone() {

            if let Ok(config) = self.config_input.next() {
                println!("JW-Debug DbscanNode has received config: {}, {}", config.min_points, config.tolerance);
                
                // dbscan
                let clusters = Dbscan::params(config.min_points)
                .tolerance(config.tolerance)
                .transform(data)
                .unwrap();
            
                // debug
                //println!("Clusters:");
                /*let label_count = clusters.label_count().remove(0);
                for (label, count) in label_count {
                    match label {
                        None => println!(" - {} noise points", count),
                        Some(i) => println!(" - {} points in cluster {}", count, i),
                    }
                }*/

                self.output.send(clusters).map_err(|e| UpdateError::Other(e.into()))?;
            }
        }

        Ok(())
    }
}


#[test]
fn input_output_test() -> Result<(), UpdateError> {
    let change_observer = ChangeObserver::new();
    let record_input: Array2<f64> = array![[1.1, 2.5, 3.2, 4.6, 5.2, 6.7], 
    [7.8, 8.2, 9.5, 10.3, 11.0, 12.0], 
    [13.0, 14.0, 15.0, 1.0, 2.0, 3.0], 
    [4.0, 5.0, 6.0, 7.0, 8.0, 9.0], 
    [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 
    [7.0, 8.0, 9.0, 10.0, 11.0, 12.0], 
    [13.0, 14.0, 15.0, 1.0, 2.0, 3.0], 
    [4.0, 5.0, 6.0, 7.0, 8.0, 9.0], 
    [10.0, 11.0, 12.0, 13.0, 14.0, 15.0]];
    let target_input: Array2<f64> = array![[-3.076047733203457, -10.562293260063301],
    [-3.561730416569943, 3.951032231750752],
    [14.63575200500477, 1.1072539713398344], 
    [-3.347031741680441, -4.147375003300382],
    [-4.622799446757189, 10.4931265494172],
    [-2.709147889142067, -11.467625779659173],
    [-3.984915594218815, 3.1728757730584096],
    [14.63575200500477, 1.1072539713398344],
    [-3.347031741680441, -4.147375003300382],
    [-4.622799446757189, 10.4931265494172]];
    let input_data: DatasetBase<ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>, ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>> = DatasetBase::new(record_input.clone(), target_input.clone());
    let test_config_input = DbscanConfig{
        min_points: 2,
        tolerance: 0.5
    };

    let mut and: DbscanNode<> = DbscanNode::new(Some(&change_observer));
    let mock_output = flowrs::connection::Edge::new();
    flowrs::connection::connect(and.output.clone(), mock_output.clone());
    and.dataset_input.send(input_data)?;
    and.config_input.send(test_config_input);
    and.on_update()?;

    let expected: Array1<Option<usize>> = array![None, None, Some(0), Some(1), Some(2), None, None, Some(0), Some(1), Some(2)];
    let actual: DatasetBase<Array2<f64>, Array1<Option<usize>>> = mock_output.next()?;

    Ok(assert!(expected == actual.targets))
}