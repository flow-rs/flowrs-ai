use flowrs::{node::{Node, UpdateError, ChangeObserver}, connection::{Input, Output}};
use flowrs::RuntimeConnectable;

use ndarray::prelude::*;
use ndarray::{Array2, Array1, array};
use linfa::dataset::DatasetBase;
use linfa::traits::Transformer;
use linfa_clustering::Dbscan;
use linfa::dataset::Labels;
use serde::{Deserialize, Serialize};


#[derive(RuntimeConnectable, Deserialize, Serialize)]
pub struct DbscanNode {
    #[output]
    pub output: Output<DatasetBase<Array2<f64>, Array1<Option<usize>>>>,

    #[input]
    pub input: Input<DatasetBase<ArrayBase<ndarray::OwnedRepr<f64>, Dim<[usize; 2]>>, ArrayBase<ndarray::OwnedRepr<()>, Dim<[usize; 1]>>>>,
}

impl DbscanNode {
    pub fn new(change_observer: Option<&ChangeObserver>) -> Self {
        Self {
            output: Output::new(change_observer),
            input: Input::new()
        }
    }
}

impl Node for DbscanNode {
    fn on_update(&mut self) -> Result<(), UpdateError> {

        if let Ok(data) = self.input.next() {
            println!("JW-Debug PCANode has received: {}.", data.records);

            // transform to DatasetBase
            //let dataset = DatasetBase::from(data);

            // parameter
            let min_points = 2;
            let tolerance = 0.5;

            // dbscan
            let clusters = Dbscan::params(min_points)
                .tolerance(tolerance)
                .transform(data)
                .unwrap();
            
            // debug
            println!("Clusters:");
            let label_count = clusters.label_count().remove(0);
            for (label, count) in label_count {
                match label {
                    None => println!(" - {} noise points", count),
                    Some(i) => println!(" - {} points in cluster {}", count, i),
                }
            }

            self.output.send(clusters).map_err(|e| UpdateError::Other(e.into()))?;
        }
        Ok(())
    }
}


#[test]
fn input_output_test() -> Result<(), UpdateError> {
    let change_observer = ChangeObserver::new();
    let data: Array2<f64> = array![[1.1, 2.5, 3.2, 4.6, 5.2, 6.7],
                                        [7.8, 8.2, 9.5, 10.3, 11.0, 12.0],
                                        [13.0, 14.0, 15.0, 1.0, 2.0, 3.0],
                                        [4.0, 5.0, 6.0, 7.0, 8.0, 9.0], 
                                        [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
                                        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 
                                        [7.0, 8.0, 9.0, 10.0, 11.0, 12.0], 
                                        [13.0, 14.0, 15.0, 1.0, 2.0, 3.0], 
                                        [4.0, 5.0, 6.0, 7.0, 8.0, 9.0], 
                                        [10.0, 11.0, 12.0, 13.0, 14.0, 15.0]];

    let test_input = DatasetBase::from(data);

    let mut and: DbscanNode<> = DbscanNode::new(Some(&change_observer));
    let mock_output = flowrs::connection::Edge::new();
    flowrs::connection::connect(and.output.clone(), mock_output.clone());
    and.input.send(test_input)?;
    and.on_update()?;

    let expected: Array1<Option<usize>> = array![None, None, Some(0), Some(1), Some(2), None, None, Some(0), Some(1), Some(2)];
    let actual: DatasetBase<Array2<f64>, Array1<Option<usize>>> = mock_output.next()?;

    Ok(assert!(expected == actual.targets))
}