use flowrs::{node::{Node, UpdateError, ChangeObserver}, connection::{Input, Output}};
use flowrs::RuntimeConnectable;

use ndarray::{Array2, array, OwnedRepr};
use ndarray::prelude::*;
use serde::{Deserialize, Serialize};
use linfa::prelude::*;

#[derive(RuntimeConnectable, Deserialize, Serialize)]
pub struct ConvertNdarray2DatasetBase { 
    #[input]
    pub input: Input<Array2<f64>>,

    #[output]
    pub output: Output<DatasetBase<Array2<f64>, Array1<()>>>
}

impl ConvertNdarray2DatasetBase {
    pub fn new(change_observer: Option<&ChangeObserver>) -> Self {
        Self {
            output: Output::new(change_observer),
            input: Input::new()
        }
    }
}

impl Node for ConvertNdarray2DatasetBase {
    fn on_update(&mut self) -> Result<(), UpdateError> {

        if let Ok(node_data) = self.input.next() {
            println!("JW-Debug: ConvertNdarray2DatasetBase has received an update!");

            let dataset = Dataset::from(node_data.clone());

            self.output.send(dataset).map_err(|e| UpdateError::Other(e.into()))?;
        }
        Ok(())
    }
}

#[test]
fn input_output_test() -> Result<(), UpdateError> {
    let change_observer = ChangeObserver::new();
    let test_input: Array2<f64> = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];

    let mut test_node: ConvertNdarray2DatasetBase<> = ConvertNdarray2DatasetBase::new(Some(&change_observer));
    let mock_output = flowrs::connection::Edge::new();
    flowrs::connection::connect(test_node.output.clone(), mock_output.clone());
    test_node.input.send(test_input.clone())?;
    test_node.on_update()?;

    let actual: DatasetBase<ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>, ArrayBase<OwnedRepr<()>, Dim<[usize; 1]>>> = mock_output.next()?;
    let expected: Array2<f64> = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];

    Ok(assert!(expected == actual.records))
}