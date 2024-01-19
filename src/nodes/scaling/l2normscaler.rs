use flowrs::RuntimeConnectable;
use flowrs::{
    connection::{Input, Output},
    node::{ChangeObserver, Node, UpdateError},
};

use linfa::prelude::*;
use linfa::traits::Transformer;
use linfa_preprocessing::norm_scaling::NormScaler;
use log::debug;
use ndarray::{array, Array1, Array2};
use serde::{Deserialize, Serialize};

/// A runtime connectable node that applies L2 normalization scaling to input data.
///
/// The `L2NormScalerNode` struct is designed to be used as a connectable node within a flowrs-flow.
/// It receives input data and applies L2 normalization scaling to it, then sends the
/// normalized data to its output port.
///

#[derive(RuntimeConnectable, Deserialize, Serialize)]
pub struct L2NormScalerNode<T>
where
    T: Clone + Float,
{
    /// The output port for sending the normalized data.
    #[output]
    pub output: Output<DatasetBase<Array2<T>, Array1<()>>>,

    /// The input port for receiving data to be normalized.
    #[input]
    pub data_input: Input<DatasetBase<Array2<T>, Array1<()>>>,
}

impl<T> L2NormScalerNode<T>
where
    T: Clone + Float,
{
    /// Create a new instance of `L2NormScalerNode`.
    ///
    /// # Parameters
    ///
    /// - `change_observer`: An optional reference to a `ChangeObserver` for tracking changes.
    ///
    /// # Returns
    ///
    /// A new instance of `L2NormScalerNode`.
    pub fn new(change_observer: Option<&ChangeObserver>) -> Self {
        Self {
            output: Output::new(change_observer),
            data_input: Input::new(),
        }
    }
}

impl<T> Node for L2NormScalerNode<T>
where
    T: Clone + Send + Float,
{
    /// Process and normalize input data.
    ///
    /// This method is called when the node is updated. It receives input data, applies L2
    /// normalization scaling, and sends the normalized data to the output port.
    ///
    /// # Returns
    ///
    /// - `Ok(())`: If the update is successful.
    /// - `Err(UpdateError)`: If an error occurs during the update.
    fn on_update(&mut self) -> Result<(), UpdateError> {
        // Receiving data
        if let Ok(data) = self.data_input.next() {
            debug!("L2NormScalerNode has received an update!");

            // Apply L2 normalization scaling to input data
            let scaler = NormScaler::l2();
            let normalized_data = scaler.transform(data);

            // Send the normalized data to the output port
            self.output
                .send(normalized_data)
                .map_err(|e| UpdateError::Other(e.into()))?;
            debug!("L2NormScalerNode has sent an output!");
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

    let dataset = Dataset::from(test_input.clone());

    let mut test_node: L2NormScalerNode<f64> = L2NormScalerNode::new(Some(&change_observer));
    let mock_output = flowrs::connection::Edge::new();
    flowrs::connection::connect(test_node.output.clone(), mock_output.clone());
    test_node.data_input.send(dataset)?;
    test_node.on_update()?;

    let expected_data = array![
        [
            0.18257418583505536,
            0.3651483716701107,
            0.5477225575051661,
            0.7302967433402214
        ],
        [
            0.3234983196103152,
            0.43133109281375365,
            0.539163866017192,
            0.6469966392206304
        ],
        [
            0.3790490217894517,
            0.454858826147342,
            0.5306686305052324,
            0.6064784348631227
        ],
        [
            0.5773502691896257,
            0.329914439536929,
            0.08247860988423225,
            0.7423074889580903
        ]
    ];

    let actual = mock_output.next()?;
    let expected = DatasetBase::new(expected_data.clone(), expected_data.clone());

    Ok(assert!(expected.records == actual.records))
}

#[test]
fn test_f32() -> Result<(), UpdateError> {
    let change_observer = ChangeObserver::new();
    let mut node: L2NormScalerNode<f32> = L2NormScalerNode::new(Some(&change_observer));
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
        [0.18257418, 0.36514837, 0.5477225, 0.73029673],
        [0.3234983, 0.4313311, 0.5391638, 0.6469966],
        [0.37904903, 0.45485884, 0.5306686, 0.60647845],
        [0.57735026, 0.32991445, 0.08247861, 0.7423075]
    ];

    let actual = mock_output.next()?;
    Ok(assert!(expected == actual.records))
}

#[test]
fn test_f64() -> Result<(), UpdateError> {
    let change_observer = ChangeObserver::new();
    let mut node: L2NormScalerNode<f64> = L2NormScalerNode::new(Some(&change_observer));
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
        [
            0.18257418583505536,
            0.3651483716701107,
            0.5477225575051661,
            0.7302967433402214
        ],
        [
            0.3234983196103152,
            0.43133109281375365,
            0.539163866017192,
            0.6469966392206304
        ],
        [
            0.3790490217894517,
            0.454858826147342,
            0.5306686305052324,
            0.6064784348631227
        ],
        [
            0.5773502691896257,
            0.329914439536929,
            0.08247860988423225,
            0.7423074889580903
        ]
    ];

    let actual = mock_output.next()?.records;

    Ok(assert!(expected == actual))
}
