use flowrs::{node::{Node, UpdateError, ChangeObserver}, connection::{Input, Output}};
use flowrs::RuntimeConnectable;

use ndarray::{Array3, ArrayBase, OwnedRepr, Dim, arr2};
use linfa::prelude::*;
use ndarray::Array2;
use ndarray::prelude::*;
use linfa::traits::{Fit, Predict};
use linfa_reduction::Pca;
use linfa_clustering::KMeans;
use serde::{Deserialize, Serialize};


#[derive(RuntimeConnectable, Deserialize, Serialize)]
pub struct ClusterNode {
    #[output]
    pub output: Output<DatasetBase<ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>, Vec<u32>>>,

    #[input]
    pub input: Input<DatasetBase<ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>, Vec<u32>>>,
}

impl ClusterNode {
    pub fn new(change_observer: Option<&ChangeObserver>) -> Self {
        Self {
            output: Output::new(change_observer),
            input: Input::new()
        }
    }
}

impl Node for ClusterNode {
    fn on_update(&mut self) -> Result<(), UpdateError> {

        if let Ok(dataset) = self.input.next() {
            println!("JW-Debug: ClusterNode has received: {}.", dataset.records);

            // Perform KMeans
            let model = KMeans::params(3)
                .max_n_iterations(200)
                .tolerance(1e-5)
                .fit(&dataset)
                .expect("Error while fitting KMeans to the dataset");

            // Predict cluster assignments
            let result = model.predict(dataset);

            // Output the cluster assignments
            for (point, cluster) in result.records.outer_iter().zip(result.targets.iter()) {
                println!("Point {:?} is in Cluster {}", point, cluster);
            } 


            let data = result.records;
            
            let myVec: Vec<u32> = vec![0, 1, 2, 4, 5];
            let dataset = DatasetBase::new(data, myVec);

            self.output.send(dataset).map_err(|e| UpdateError::Other(e.into()))?;
        }
        Ok(())
    }
}

/*
#[test]
fn input_output_test() -> Result<(), UpdateError> {
    let change_observer = ChangeObserver::new();
    let test_input: Array2<u8> = arr2(&[[1, 2, 3], [4, 5, 6], [7, 8, 9]]);

    let mut and: ClusterNode<> = ClusterNode::new(Some(&change_observer));
    let mock_output = flowrs::connection::Edge::new();
    flowrs::connection::connect(and.output.clone(), mock_output.clone());
    and.input.send(test_input)?;
    and.on_update()?;

    let expected = arr2(&[[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
    let actual = mock_output.next()?;
    Ok(assert!(expected == actual))
} */