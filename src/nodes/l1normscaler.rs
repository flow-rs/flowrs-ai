use flowrs::{node::{Node, UpdateError, ChangeObserver}, connection::{Input, Output}};
use flowrs::RuntimeConnectable;

use ndarray::{Array2, array, ArrayBase, OwnedRepr, Dim, Array1};
use linfa::traits::Transformer;
use linfa_preprocessing::norm_scaling::NormScaler;
use serde::{Deserialize, Serialize};
use linfa::prelude::*;

#[derive(RuntimeConnectable, Deserialize, Serialize)]
pub struct L1NormscalerNode<T>
where
    T: Clone,
{
    #[output]
    pub output: Output<DatasetBase<Array2<T>, Array1<()>>>,

    #[input]
    pub input: Input<DatasetBase<Array2<T>, Array1<()>>>,
}

impl<T> L1NormscalerNode<T> 
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

impl<T> Node for L1NormscalerNode<T>
where
    T: Clone + Send + linfa::Float,
{
    fn on_update(&mut self) -> Result<(), UpdateError> {

        if let Ok(node_data) = self.input.next() {
            println!("JW-Debug: L1NormscalerNode has received an update!");//println!("JW-Debug: L1NormscalerNode has received: {}.", node_data.records);

            let scaler = NormScaler::l1();
            let normalized_data = scaler.transform(node_data.clone());
    
            self.output.send(normalized_data).map_err(|e| UpdateError::Other(e.into()))?;
            println!("JW-Debug: L1NormscalerNode has sent an output!");

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

    let mut and: L1NormscalerNode<f64> = L1NormscalerNode::new(Some(&change_observer));
    let mock_output = flowrs::connection::Edge::new();
    flowrs::connection::connect(and.output.clone(), mock_output.clone());
    and.input.send(dataset)?;
    and.on_update()?;

    let expected_data = array![[0.1, 0.2, 0.3, 0.4],
    [0.16666666666666666, 0.2222222222222222, 0.2777777777777778, 0.3333333333333333],
    [0.19230769230769232, 0.23076923076923078, 0.2692307692307692, 0.3076923076923077],
    [0.3333333333333333, 0.19047619047619047, 0.047619047619047616, 0.42857142857142855]];

    let actual: DatasetBase<ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>, ArrayBase<OwnedRepr<()>, Dim<[usize; 1]>>> = mock_output.next()?;
    let expected: DatasetBase<ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>, ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>> = DatasetBase::new(expected_data.clone(), expected_data.clone());

    Ok(assert!(expected.records == actual.records))
}