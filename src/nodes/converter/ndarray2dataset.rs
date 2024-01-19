use flowrs::RuntimeConnectable;
use flowrs::{
    connection::{Input, Output},
    node::{ChangeObserver, Node, UpdateError},
};

use linfa::prelude::*;
use log::debug;
use ndarray::prelude::*;
use ndarray::{array, Array2};
use serde::{Deserialize, Serialize};

/// A runtime connectable node that converts an ndarray into a dataset.
///
/// The `NdarrayToDatasetNode` struct is designed to be used as a connectable node within a data flow graph.
/// It receives input data in the form of an ndarray and sends the corresponding dataset to its output port.
///

#[derive(RuntimeConnectable, Deserialize, Serialize)]
pub struct NdarrayToDatasetNode<T>
where
    T: Clone,
{
    /// The input port for receiving data as an ndarray.
    #[input]
    pub data_input: Input<Array2<T>>,

    /// The output port for sending the resulting dataset.
    #[output]
    pub output: Output<DatasetBase<Array2<T>, Array1<()>>>,
}

impl<T> NdarrayToDatasetNode<T>
where
    T: Clone,
{
    /// Create a new instance of `NdarrayToDatasetNode`.
    ///
    /// # Parameters
    ///
    /// - `change_observer`: An optional reference to a `ChangeObserver` for tracking changes.
    ///
    /// # Returns
    ///
    /// A new instance of `NdarrayToDatasetNode`.
    pub fn new(change_observer: Option<&ChangeObserver>) -> Self {
        Self {
            output: Output::new(change_observer),
            data_input: Input::new(),
        }
    }
}

impl<T> Node for NdarrayToDatasetNode<T>
where
    T: Clone + Send,
{
    /// Process input ndarray data and send the resulting dataset to the output port.
    ///
    /// This method is called when the node is updated. It receives the input ndarray data,
    /// creates a dataset from it, and sends the resulting dataset to the output port.
    ///
    /// # Returns
    ///
    /// - `Ok(())`: If the update is successful.
    /// - `Err(UpdateError)`: If an error occurs during the update.
    fn on_update(&mut self) -> Result<(), UpdateError> {
        if let Ok(data) = self.data_input.next() {
            debug!("NdarrayToDatasetNode has received an update!");

            let dataset = Dataset::from(data.clone());

            self.output
                .send(dataset)
                .map_err(|e| UpdateError::Other(e.into()))?;
        }
        Ok(())
    }
}

#[test]
fn input_output_test() -> Result<(), UpdateError> {
    let change_observer = ChangeObserver::new();
    let test_input: Array2<f64> = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];

    let mut test_node: NdarrayToDatasetNode<f64> =
        NdarrayToDatasetNode::new(Some(&change_observer));
    let mock_output = flowrs::connection::Edge::new();
    flowrs::connection::connect(test_node.output.clone(), mock_output.clone());
    test_node.data_input.send(test_input.clone())?;
    test_node.on_update()?;

    let actual = mock_output.next()?;
    let expected: Array2<f64> = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];

    Ok(assert!(expected == actual.records))
}

#[test]
fn test_f32() -> Result<(), UpdateError> {
    let change_observer = ChangeObserver::new();
    let mut node: NdarrayToDatasetNode<f32> = NdarrayToDatasetNode::new(Some(&change_observer));
    let mock_output = flowrs::connection::Edge::new();
    flowrs::connection::connect(node.output.clone(), mock_output.clone());

    let test_data_input = array![
        [1.0, 2.0, 3.0, 4.0],
        [3.0, 4.0, 5.0, 6.0],
        [5.0, 6.0, 7.0, 8.0],
        [7.0, 4.0, 1.0, 9.0]
    ];

    node.data_input.send(test_data_input.clone())?;
    node.on_update()?;

    let actual = mock_output.next()?.records;

    Ok(assert!(test_data_input == actual))
}

#[test]
fn test_f64() -> Result<(), UpdateError> {
    let change_observer = ChangeObserver::new();
    let mut node: NdarrayToDatasetNode<f64> = NdarrayToDatasetNode::new(Some(&change_observer));
    let mock_output = flowrs::connection::Edge::new();
    flowrs::connection::connect(node.output.clone(), mock_output.clone());

    let test_data_input = array![
        [1.0, 2.0, 3.0, 4.0],
        [3.0, 4.0, 5.0, 6.0],
        [5.0, 6.0, 7.0, 8.0],
        [7.0, 4.0, 1.0, 9.0]
    ];

    node.data_input.send(test_data_input.clone())?;
    node.on_update()?;

    let actual = mock_output.next()?.records;

    Ok(assert!(test_data_input == actual))
}
