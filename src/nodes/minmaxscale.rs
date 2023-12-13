use flowrs::{node::{Node, UpdateError, ChangeObserver}, connection::{Input, Output}};
use flowrs::RuntimeConnectable;

use ndarray::{Array2, Array1, array, ArrayBase, OwnedRepr, Dim};
use linfa::{dataset::DatasetBase, Dataset};
use linfa::traits::{Fit, Transformer};
use linfa_preprocessing::linear_scaling::LinearScaler;
use serde::{Deserialize, Serialize};


#[derive(RuntimeConnectable, Deserialize, Serialize)]
pub struct MinMaxScaleNode {
    #[output]
    pub output: Output<DatasetBase<Array2<f64>, Array1<()>>>, // <--- Wir haben in diesem Fall eine Output-Variable vom Typ Array2<u8>

    #[input]
    pub input: Input<DatasetBase<Array2<f64>, Array1<()>>>, // <--- Wir haben in diesem Fall eine Input-Variable vom Typ Array2<u8>

}

impl MinMaxScaleNode {
    pub fn new(change_observer: Option<&ChangeObserver>) -> Self {
        Self {
            output: Output::new(change_observer),
            input: Input::new()
        }
    }
}

impl Node for MinMaxScaleNode {
    fn on_update(&mut self) -> Result<(), UpdateError> {

        if let Ok(node_data) = self.input.next() {
            println!("JW-Debug: MinMaxScaleNode has received: {}.", node_data.records);

            // max abs scaling
            let scaler = LinearScaler::min_max().fit(&node_data).unwrap();
            let dataset = scaler.transform(node_data);

            // debug
            println!("Scaled data: {}", dataset.records);

            self.output.send(dataset).map_err(|e| UpdateError::Other(e.into()))?;
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

    let mut test_node: MinMaxScaleNode<> = MinMaxScaleNode::new(Some(&change_observer));
    let mock_output = flowrs::connection::Edge::new();
    flowrs::connection::connect(test_node.output.clone(), mock_output.clone());
    test_node.input.send(dataset)?;
    test_node.on_update()?;

    let expected_data: Array2<f64> = array![[0.00833333333333334, 0.041666666666666664, 0.01666666666666668, 0.29999999999999993, 0.26666666666666666, 0.30833333333333335],
                                       [0.5666666666666667, 0.5166666666666666, 0.5416666666666666, 0.775, 0.75, 0.75],
                                       [1., 1., 1., 0., 0., 0.],
                                       [0.25, 0.25, 0.25, 0.5, 0.5, 0.5],
                                       [0.75, 0.75, 0.75, 1., 1., 1.],
                                       [0., 0., 0., 0.25, 0.25, 0.25],
                                       [0.5, 0.5, 0.5, 0.75, 0.75, 0.75],
                                       [1., 1., 1., 0., 0., 0.],
                                       [0.25, 0.25, 0.25, 0.5, 0.5, 0.5],
                                       [0.75, 0.75, 0.75, 1., 1., 1.]];
    let actual: DatasetBase<ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>, ArrayBase<OwnedRepr<()>, Dim<[usize; 1]>>> = mock_output.next()?;
    let expected: DatasetBase<ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>, ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>> = DatasetBase::new(expected_data.clone(), expected_data.clone());

    Ok(assert!(expected.records == actual.records))
}