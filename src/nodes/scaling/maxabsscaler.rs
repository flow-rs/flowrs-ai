use flowrs::{node::{Node, UpdateError, ChangeObserver}, connection::{Input, Output}};
use flowrs::RuntimeConnectable;

use ndarray::{Array2, Array1, array, ArrayBase, Dim, OwnedRepr};
use linfa::{dataset::DatasetBase, Dataset, Float};
use linfa::traits::{Fit, Transformer};
use linfa_preprocessing::linear_scaling::LinearScaler;
use serde::{Deserialize, Serialize};
use log::debug;


#[derive(RuntimeConnectable, Deserialize, Serialize)]
pub struct MaxAbsSclerNode<T> 
where
    T: Clone,
{
    #[output]
    pub output: Output<DatasetBase<Array2<T>, Array1<()>>>, 

    #[input]
    pub data_input: Input<DatasetBase<Array2<T>, Array1<()>>>,

}


impl<T> MaxAbsSclerNode<T> 
where
    T: Clone
{
    pub fn new(change_observer: Option<&ChangeObserver>) -> Self {
        Self {
            output: Output::new(change_observer),
            data_input: Input::new()
        }
    }
}


impl<T> Node for MaxAbsSclerNode<T> 
where
    T: Clone + Send + Float
{
    fn on_update(&mut self) -> Result<(), UpdateError> {

        if let Ok(data) = self.data_input.next() {
            debug!("MaxAbsScalerNode has received an update!");//debug!("MaxAbsScalerNode has received: {}.", dataset.records);

            let scaler = LinearScaler::max_abs().fit(&data).unwrap();
            let dataset = scaler.transform(data);

            self.output.send(dataset).map_err(|e| UpdateError::Other(e.into()))?;
            debug!("MaxAbsScalerNode has sent an output!");

        }
        Ok(())
    }
}


#[test]
fn input_output_test() -> Result<(), UpdateError> {
    let change_observer = ChangeObserver::new();
    let test_input: Array2<f64> = array![[1.1, 2.5, 3.2, 4.6, 5.2, 6.7], 
                                         [7.8, 8.2, 9.5, 10.3, 11.0, 12.0], 
                                         [13.0, 14.0, 15.0, 1.0, 2.0, 3.0], 
                                         [4.0, 5.0, 6.0, 7.0, 8.0, 9.0], 
                                         [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
                                         [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 
                                         [7.0, 8.0, 9.0, 10.0, 11.0, 12.0], 
                                         [13.0, 14.0, 15.0, 1.0, 2.0, 3.0], 
                                         [4.0, 5.0, 6.0, 7.0, 8.0, 9.0], 
                                         [10.0, 11.0, 12.0, 13.0, 14.0, 15.0]];
    let dataset = Dataset::from(test_input.clone());

    let mut test_node: MaxAbsSclerNode<f64> = MaxAbsSclerNode::new(Some(&change_observer));
    let mock_output = flowrs::connection::Edge::new();
    flowrs::connection::connect(test_node.output.clone(), mock_output.clone());
    test_node.data_input.send(dataset)?;
    test_node.on_update()?;

    let expected_data: Array2<f64> = array![[0.08461538461538462, 0.17857142857142855, 0.21333333333333335, 0.35384615384615387, 0.37142857142857144, 0.44666666666666666],
                                       [0.6, 0.5857142857142856, 0.6333333333333333, 0.7923076923076924, 0.7857142857142857, 0.8],
                                       [1., 1., 1., 0.07692307692307693, 0.14285714285714285, 0.2],
                                       [0.3076923076923077, 0.3571428571428571, 0.4, 0.5384615384615385, 0.5714285714285714, 0.6],
                                       [0.7692307692307693, 0.7857142857142857, 0.8, 1., 1., 1.],
                                       [0.07692307692307693, 0.14285714285714285, 0.2, 0.3076923076923077, 0.3571428571428571, 0.4],
                                       [0.5384615384615385, 0.5714285714285714, 0.6, 0.7692307692307693, 0.7857142857142857, 0.8],
                                       [1., 1., 1., 0.07692307692307693, 0.14285714285714285, 0.2],
                                       [0.3076923076923077, 0.3571428571428571, 0.4, 0.5384615384615385, 0.5714285714285714, 0.6],
                                       [0.7692307692307693, 0.7857142857142857, 0.8, 1., 1., 1.]];
    let actual: DatasetBase<ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>, ArrayBase<OwnedRepr<()>, Dim<[usize; 1]>>> = mock_output.next()?;
    let expected: DatasetBase<ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>, ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>> = DatasetBase::new(expected_data.clone(), expected_data.clone());

    Ok(assert!(expected.records == actual.records))
}


#[test]
fn test_f32() -> Result<(), UpdateError> {
    let change_observer = ChangeObserver::new();
    let mut node: MaxAbsSclerNode<f32> = MaxAbsSclerNode::new(Some(&change_observer));
    let mock_output = flowrs::connection::Edge::new();
    flowrs::connection::connect(node.output.clone(), mock_output.clone());

    let test_data = array![[1.0, 2.0, 3.0, 4.0],
    [3.0, 4.0, 5.0, 6.0],
    [5.0, 6.0, 7.0, 8.0],
    [7.0, 4.0, 1.0, 9.0]];
    let test_data_input = DatasetBase::from(test_data);

    node.data_input.send(test_data_input)?;
    node.on_update()?;

    let expected = array![[0.14285715, 0.33333334, 0.42857146, 0.44444445],
    [0.42857146, 0.6666667, 0.71428573, 0.6666667],
    [0.71428573, 1., 1., 0.8888889],
    [1., 0.6666667, 0.14285715, 1.]];

    let actual = mock_output.next()?.records;

    Ok(assert!(expected == actual))
}


#[test]
fn test_f64() -> Result<(), UpdateError> {
    let change_observer = ChangeObserver::new();
    let mut node: MaxAbsSclerNode<f64> = MaxAbsSclerNode::new(Some(&change_observer));
    let mock_output = flowrs::connection::Edge::new();
    flowrs::connection::connect(node.output.clone(), mock_output.clone());

    let test_data = array![[1.0, 2.0, 3.0, 4.0],
    [3.0, 4.0, 5.0, 6.0],
    [5.0, 6.0, 7.0, 8.0],
    [7.0, 4.0, 1.0, 9.0]];
    let test_data_input = DatasetBase::from(test_data);

    node.data_input.send(test_data_input)?;
    node.on_update()?;

    let expected = array![[0.14285714285714285, 0.3333333333333333, 0.42857142857142855, 0.4444444444444444],
    [0.42857142857142855, 0.6666666666666666, 0.7142857142857142, 0.6666666666666666],
    [0.7142857142857142, 1., 1., 0.8888888888888888],
    [1., 0.6666666666666666, 0.14285714285714285, 1.]];

    let actual = mock_output.next()?.records;
    
    Ok(assert!(expected == actual))
}