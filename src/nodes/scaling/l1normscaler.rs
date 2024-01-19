use flowrs::RuntimeConnectable;
use flowrs::{
    connection::{Input, Output},
    node::{ChangeObserver, Node, UpdateError},
};

use linfa::{traits::Transformer, DatasetBase, Float};
use linfa_preprocessing::norm_scaling::NormScaler;
use ndarray::{array, Array1, Array2};
use serde::{Deserialize, Serialize};

use linfa::prelude::*;
use log::debug;

/// A runtime connectable node that applies L1 normalization scaling to input data.
///
/// The `L1NormScalerNode` struct is designed to be used as a connectable node within a flowrs-flow.
/// It receives input data and applies L1 normalization scaling to it, then sends the
/// normalized data to its output port.
///

#[derive(RuntimeConnectable, Deserialize, Serialize)]
pub struct L1NormScalerNode<T>
where
    T: Clone,
{
    /// The output port for sending the normalized data.
    #[output]
    pub output: Output<DatasetBase<Array2<T>, Array1<()>>>,

    /// The input port for receiving data to be normalized.
    #[input]
    pub data_input: Input<DatasetBase<Array2<T>, Array1<()>>>,
}

impl<T> L1NormScalerNode<T>
where
    T: Clone + Send + Float,
{
    /// Create a new instance of `L1NormScalerNode`.
    ///
    /// # Parameters
    ///
    /// - `change_observer`: An optional reference to a `ChangeObserver` for tracking changes.
    ///
    /// # Returns
    ///
    /// A new instance of `L1NormScalerNode`.
    pub fn new(change_observer: Option<&ChangeObserver>) -> Self {
        Self {
            output: Output::new(change_observer),
            data_input: Input::new(),
        }
    }
}

impl<T> Node for L1NormScalerNode<T>
where
    T: Clone + Send + Float,
{
    /// Process and normalize input data.
    ///
    /// This method is called when the node is updated. It receives input data, applies L1
    /// normalization scaling, and sends the normalized data to the output port.
    ///
    /// # Returns
    ///
    /// - `Ok(())`: If the update is successful.
    /// - `Err(UpdateError)`: If an error occurs during the update.
    fn on_update(&mut self) -> Result<(), UpdateError> {
        // Receiving data
        if let Ok(data) = self.data_input.next() {
            debug!("L1NormScalerNode has received an update!");

            // Apply L1 normalization scaling to input data
            let scaler = NormScaler::l1();
            let normalized_data = scaler.transform(data);

            // Send the normalized data to the output port
            self.output
                .send(normalized_data)
                .map_err(|e| UpdateError::Other(e.into()))?;
            debug!("L1NormScalerNode has sent an output!");
        }
        Ok(())
    }
}

#[test]
fn input_output_test() -> Result<(), UpdateError> {
    let change_observer = ChangeObserver::new();
    let test_input: Array2<f64> = array![
        [1.0, 2.0, 3.0, 4.0],
        [3.0, 4.0, 5.0, 6.0],
        [5.0, 6.0, 7.0, 8.0],
        [7.0, 4.0, 1.0, 9.0]
    ];

    let dataset = DatasetBase::from(test_input.clone());

    let mut and: L1NormScalerNode<f64> = L1NormScalerNode::new(Some(&change_observer));
    let mock_output = flowrs::connection::Edge::new();
    flowrs::connection::connect(and.output.clone(), mock_output.clone());
    and.data_input.send(dataset)?;
    and.on_update()?;

    let expected_data = array![
        [0.1, 0.2, 0.3, 0.4],
        [
            0.16666666666666666,
            0.2222222222222222,
            0.2777777777777778,
            0.3333333333333333
        ],
        [
            0.19230769230769232,
            0.23076923076923078,
            0.2692307692307692,
            0.3076923076923077
        ],
        [
            0.3333333333333333,
            0.19047619047619047,
            0.047619047619047616,
            0.42857142857142855
        ]
    ];

    let actual = mock_output.next()?;
    let expected = DatasetBase::new(expected_data.clone(), expected_data.clone());

    Ok(assert!(expected.records == actual.records))
}

#[test]
fn test_f32() -> Result<(), UpdateError> {
    let change_observer = ChangeObserver::new();
    let mut node: L1NormScalerNode<f32> = L1NormScalerNode::new(Some(&change_observer));
    let mock_output = flowrs::connection::Edge::new();
    flowrs::connection::connect(node.output.clone(), mock_output.clone());

    let test_data = array![
        [1.0, 2.0, 3.0, 4.0],
        [3.0, 4.0, 5.0, 6.0],
        [5.0, 6.0, 7.0, 8.0],
        [7.0, 4.0, 1.0, 9.0]
    ];
    let test_data_input = DatasetBase::from(test_data);

    node.data_input.send(test_data_input)?;
    node.on_update()?;

    let expected = array![
        [0.1, 0.2, 0.3, 0.4],
        [
            0.16666666666666666,
            0.2222222222222222,
            0.2777777777777778,
            0.3333333333333333
        ],
        [
            0.19230769230769232,
            0.23076923076923078,
            0.2692307692307692,
            0.3076923076923077
        ],
        [
            0.3333333333333333,
            0.19047619047619047,
            0.047619047619047616,
            0.42857142857142855
        ]
    ];

    let actual = mock_output.next()?.records;

    Ok(assert!(expected == actual))
}

#[test]
fn test_f64() -> Result<(), UpdateError> {
    let change_observer = ChangeObserver::new();
    let mut node: L1NormScalerNode<f64> = L1NormScalerNode::new(Some(&change_observer));
    let mock_output = flowrs::connection::Edge::new();
    flowrs::connection::connect(node.output.clone(), mock_output.clone());

    let test_data = array![
        [1.0, 2.0, 3.0, 4.0],
        [3.0, 4.0, 5.0, 6.0],
        [5.0, 6.0, 7.0, 8.0],
        [7.0, 4.0, 1.0, 9.0]
    ];
    let test_data_input = DatasetBase::from(test_data);

    node.data_input.send(test_data_input)?;
    node.on_update()?;

    let expected = array![
        [0.1, 0.2, 0.3, 0.4],
        [
            0.16666666666666666,
            0.2222222222222222,
            0.2777777777777778,
            0.3333333333333333
        ],
        [
            0.19230769230769232,
            0.23076923076923078,
            0.2692307692307692,
            0.3076923076923077
        ],
        [
            0.3333333333333333,
            0.19047619047619047,
            0.047619047619047616,
            0.42857142857142855
        ]
    ];

    let actual = mock_output.next()?.records;

    Ok(assert!(expected == actual))
}
