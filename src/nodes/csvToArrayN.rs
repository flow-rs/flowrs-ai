
use flowrs::{node::{Node, UpdateError, ChangeObserver}, connection::{Input, Output}};
use flowrs::RuntimeConnectable;

use ndarray::arr2;
use ndarray::Array2;
use linfa::dataset::DatasetBase;
use csv::ReaderBuilder;
use ndarray_csv::Array2Reader;
use serde::{Deserialize, Serialize};



#[derive(RuntimeConnectable, Deserialize, Serialize)]
pub struct CSVToArrayNNode {
    #[output]
    pub output: Output<Array2<f64>>,

    #[input]
    pub input: Input<String>,
}

impl CSVToArrayNNode {
    pub fn new(change_observer: Option<&ChangeObserver>) -> Self {
        Self {
            output: Output::new(change_observer),
            input: Input::new()
        }
    }
}

impl Node for CSVToArrayNNode {
    fn on_update(&mut self) -> Result<(), UpdateError> {

        if let Ok(data) = self.input.next() {
            println!("JW-Debug CSVToArrayNNode has received: {}.", data);

            // input has feature_names
            let has_fature_names = false;

            let mut reader = ReaderBuilder::new().has_headers(has_fature_names).from_reader(data.as_bytes());

            let data_ndarray: Array2<f64> = reader.deserialize_array2_dynamic().map_err(|e| UpdateError::Other(e.into()))?;
            println!("{}", data_ndarray);

            self.output.send(data_ndarray).map_err(|e| UpdateError::Other(e.into()))?;
        }
        Ok(())
    }
}

#[test]
fn input_output_test() -> Result<(), UpdateError> {
    let change_observer = ChangeObserver::new();
    let test_input = String::from("1,2,3\n4,5,6\n7,8,9");

    let mut and: CSVToArrayNNode<> = CSVToArrayNNode::new(Some(&change_observer));
    let mock_output = flowrs::connection::Edge::new();
    flowrs::connection::connect(and.output.clone(), mock_output.clone());
    and.input.send(test_input)?;
    and.on_update()?;

    let expected: Array2<f64> = arr2(&[[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]);
    let actual: Array2<f64> = mock_output.next()?;

    Ok(assert!(expected == actual))
}