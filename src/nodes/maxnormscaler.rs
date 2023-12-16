use flowrs::{node::{Node, UpdateError, ChangeObserver}, connection::{Input, Output}};
use flowrs::RuntimeConnectable;

use ndarray::{Array2, Array1, array, ArrayBase, OwnedRepr, Dim};
use linfa::traits::Transformer;
use linfa_preprocessing::norm_scaling::NormScaler;
use serde::{Deserialize, Serialize};
use linfa::prelude::*;

#[derive(RuntimeConnectable, Deserialize, Serialize)]
pub struct MaxNormScalerNode<T>
where
    T: Clone,
{
    #[output]
    pub output: Output<DatasetBase<Array2<T>, Array1<()>>>,

    #[input]
    pub input: Input<DatasetBase<Array2<T>, Array1<()>>>,
}

impl<T> MaxNormScalerNode<T> 
where
    T: Clone,
{
    pub fn new(change_observer: Option<&ChangeObserver>) -> Self {
        Self {
            output: Output::new(change_observer),
            input: Input::new()
        }
    }
}

impl<T> Node for MaxNormScalerNode<T> 
where
    T: Clone + Send + linfa::Float,
{
    fn on_update(&mut self) -> Result<(), UpdateError> {

        if let Ok(node_data) = self.input.next() {
            println!("JW-Debug: MaxNormScalerNode has received data!"); //println!("JW-Debug: MaxNormScalerNode has received: {}.", node_data.records);

            let scaler = NormScaler::max();
            let normalized_data = scaler.transform(node_data.clone());
    
            self.output.send(normalized_data).map_err(|e| UpdateError::Other(e.into()))?;
            println!("JW-Debug: MaxNormScalerNode has sent an output!");

        }
        Ok(())
    }
}


#[test]
fn input_output_test() -> Result<(), UpdateError> {
    let change_observer = ChangeObserver::new();
    let test_input: Array2<f64> = array![[1.0, 2.0, 3.0, 4.0],
    [3.0, 4.0, 5.0, 6.0],
    [5.0, 6.0, 7.0, 8.0],
    [7.0, 4.0, 1.0, 9.0]];

    let dataset = Dataset::from(test_input.clone());

    let mut test_node: MaxNormScalerNode<f64> = MaxNormScalerNode::new(Some(&change_observer));
    let mock_output = flowrs::connection::Edge::new();
    flowrs::connection::connect(test_node.output.clone(), mock_output.clone());
    test_node.input.send(dataset)?;
    test_node.on_update()?;

    let expected_data = array![[0.25, 0.5, 0.75, 1.0],
    [0.5, 0.6666666666666666, 0.8333333333333334, 1.0],
    [0.625, 0.75, 0.875, 1.0],
    [0.7777777777777778, 0.4444444444444444, 0.1111111111111111, 1.0]];

    let actual: DatasetBase<ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>, ArrayBase<OwnedRepr<()>, Dim<[usize; 1]>>> = mock_output.next()?;
    let expected: DatasetBase<ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>, ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>> = DatasetBase::new(expected_data.clone(), expected_data.clone());

    Ok(assert!(expected.records == actual.records))
}