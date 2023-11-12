use flowrs::{node::{Node, UpdateError, ChangeObserver}, connection::{Input, Output}};
use flowrs::RuntimeConnectable;

use ndarray::Array2;
use linfa::dataset::DatasetBase;
use linfa::traits::Transformer;
use ndarray::{ArrayBase, OwnedRepr, Dim, arr2};
use linfa_clustering::Dbscan;
use linfa::dataset::Labels;
use serde::{Deserialize, Serialize};


#[derive(RuntimeConnectable, Deserialize, Serialize)]
pub struct DbscanNode {
    #[output]
    pub output: Output<DatasetBase<ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>, ArrayBase<OwnedRepr<Option<usize>>, Dim<[usize; 1]>>>>,

    #[input]
    pub input: Input<Array2<f64>>,
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
            println!("JW-Debug PCANode has received: {}.", data);

            let dataset = DatasetBase::from(data);

            let clusters = Dbscan::params(2)
                .tolerance(0.5)
                .transform(dataset)
                .unwrap();

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
    let test_input: Array2<f64> = arr2(&[[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]);

    let mut and: DbscanNode<> = DbscanNode::new(Some(&change_observer));
    let mock_output = flowrs::connection::Edge::new();
    flowrs::connection::connect(and.output.clone(), mock_output.clone());
    and.input.send(test_input)?;
    and.on_update()?;

    let expected: Array2<f64> = arr2(&[[-5.19615242270663],[0.],[5.19615242270663]]);
    let actual: DatasetBase<ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>, ArrayBase<OwnedRepr<Option<usize>>, Dim<[usize; 1]>>> = mock_output.next()?;

    Ok(())
}