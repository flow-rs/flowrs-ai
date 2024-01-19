use flowrs::RuntimeConnectable;
use flowrs::{
    connection::{Input, Output},
    node::{ChangeObserver, Node, UpdateError},
};

use linfa::traits::{Fit, Transformer};
use linfa::{dataset::DatasetBase, Float};
use linfa_preprocessing::linear_scaling::LinearScaler;
use log::debug;
use ndarray::{array, Array1, Array2};
use serde::{Deserialize, Serialize};

/// Configuration for the Min-Max Range Scaler.
///
/// The `MinMaxRangeScalerConfig` struct represents the configuration for the Min-Max Range Scaler.
/// It specifies the minimum and maximum values for scaling input data.
#[derive(Clone, Deserialize, Serialize)]
pub struct MinMaxRangeScalerConfig {
    /// The minimum value for scaling.
    pub min: f64,
    /// The maximum value for scaling.
    pub max: f64,
}

impl MinMaxRangeScalerConfig {
    /// Create a new instance of `MinMaxRangeScalerConfig` with specified minimum and maximum values.
    ///
    /// # Parameters
    ///
    /// - `min`: The minimum value for scaling.
    /// - `max`: The maximum value for scaling.
    ///
    /// # Returns
    ///
    /// A new instance of `MinMaxRangeScalerConfig`.
    pub fn new(min: f64, max: f64) -> Self {
        MinMaxRangeScalerConfig { min, max }
    }
}

/// A runtime connectable node that applies Min-Max scaling with a configurable range to input data.
///
/// The `MinMaxRangeScalerNode` struct is designed to be used as a connectable node within a flowrs-flow.
/// It receives input data and a configuration specifying the minimum and maximum values
/// for scaling. It then applies Min-Max scaling with the specified range to the input data and sends
/// the scaled data to its output port.
///
#[derive(RuntimeConnectable, Deserialize, Serialize)]
pub struct MinMaxRangeScalerNode<T>
where
    T: Clone + Float,
{
    /// The output port for sending the scaled data.
    #[output]
    pub output: Output<DatasetBase<Array2<T>, Array1<()>>>,

    /// The input port for receiving data to be scaled.
    #[input]
    pub data_input: Input<DatasetBase<Array2<T>, Array1<()>>>,

    /// The input port for receiving the configuration.
    #[input]
    pub config_input: Input<MinMaxRangeScalerConfig>,

    /// The configuration for Min-Max scaling.
    config: MinMaxRangeScalerConfig,
}

impl<T> MinMaxRangeScalerNode<T>
where
    T: Clone + Float,
{
    /// Create a new instance of `MinMaxRangeScalerNode`.
    ///
    /// # Parameters
    ///
    /// - `change_observer`: An optional reference to a `ChangeObserver` for tracking changes.
    ///
    /// # Returns
    ///
    /// A new instance of `MinMaxRangeScalerNode`.
    pub fn new(change_observer: Option<&ChangeObserver>) -> Self {
        Self {
            output: Output::new(change_observer),
            data_input: Input::new(),
            config_input: Input::new(),
            config: MinMaxRangeScalerConfig::new(0.0, 1.0),
        }
    }
}

impl<T> Node for MinMaxRangeScalerNode<T>
where
    T: Clone + Send + Float,
{
    /// Process and scale input data using Min-Max scaling with a configurable range.
    ///
    /// This method is called when the node is updated. It receives both the configuration and input
    /// data. It then applies Min-Max scaling with the specified range to the input data and sends the
    /// scaled data to the output port.
    ///
    /// # Returns
    ///
    /// - `Ok(())`: If the update is successful.
    /// - `Err(UpdateError)`: If an error occurs during the update.
    fn on_update(&mut self) -> Result<(), UpdateError> {
        debug!("MinMaxRangeScalerNode has received an update!");

        // Receiving config
        if let Ok(config) = self.config_input.next() {
            debug!(
                "MinMaxRangeScalerNode has received config: {}, {}",
                config.min, config.max
            );

            self.config = config;
        }

        // Receiving data
        if let Ok(data) = self.data_input.next() {
            debug!("MinMaxRangeScalerNode has received data!");

            // Apply Min-Max scaling to input data with the specified range
            let scaler = LinearScaler::min_max_range(
                T::from(self.config.min).unwrap(),
                T::from(self.config.max).unwrap(),
            )
            .fit(&data)
            .unwrap();
            let scaled_data = scaler.transform(data);

            // Send the scaled data to the output port
            self.output
                .send(scaled_data)
                .map_err(|e| UpdateError::Other(e.into()))?;
        }
        Ok(())
    }
}

#[test]
fn new_config_test() -> Result<(), UpdateError> {
    let change_observer = ChangeObserver::new();
    let test_config_input = MinMaxRangeScalerConfig { min: 0.0, max: 1.0 };
    let test_input: Array2<f64> = array![
        [1.1, 2.5, 3.2, 4.6, 5.2, 6.7],
        [7.8, 8.2, 9.5, 10.3, 11.0, 12.0],
        [13.0, 14.0, 15.0, 1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        [13.0, 14.0, 15.0, 1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        [10.0, 11.0, 12.0, 13.0, 14.0, 15.0]
    ];
    let dataset = DatasetBase::from(test_input.clone());

    let mut test_node: MinMaxRangeScalerNode<f64> =
        MinMaxRangeScalerNode::new(Some(&change_observer));
    let mock_output = flowrs::connection::Edge::new();
    flowrs::connection::connect(test_node.output.clone(), mock_output.clone());
    test_node.data_input.send(dataset)?;
    test_node.config_input.send(test_config_input)?;
    test_node.on_update()?;

    let expected_data: Array2<f64> = array![
        [
            0.00833333333333334,
            0.041666666666666664,
            0.01666666666666668,
            0.29999999999999993,
            0.26666666666666666,
            0.30833333333333335
        ],
        [
            0.5666666666666667,
            0.5166666666666666,
            0.5416666666666666,
            0.775,
            0.75,
            0.75
        ],
        [1., 1., 1., 0., 0., 0.],
        [0.25, 0.25, 0.25, 0.5, 0.5, 0.5],
        [0.75, 0.75, 0.75, 1., 1., 1.],
        [0., 0., 0., 0.25, 0.25, 0.25],
        [0.5, 0.5, 0.5, 0.75, 0.75, 0.75],
        [1., 1., 1., 0., 0., 0.],
        [0.25, 0.25, 0.25, 0.5, 0.5, 0.5],
        [0.75, 0.75, 0.75, 1., 1., 1.]
    ];
    let actual = mock_output.next()?;
    let expected = DatasetBase::new(expected_data.clone(), expected_data.clone());

    Ok(assert!(expected.records == actual.records))
}

#[test]
fn default_config_test() -> Result<(), UpdateError> {
    let change_observer = ChangeObserver::new();

    let test_input: Array2<f64> = array![
        [1.1, 2.5, 3.2, 4.6, 5.2, 6.7],
        [7.8, 8.2, 9.5, 10.3, 11.0, 12.0],
        [13.0, 14.0, 15.0, 1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        [13.0, 14.0, 15.0, 1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        [10.0, 11.0, 12.0, 13.0, 14.0, 15.0]
    ];
    let dataset = DatasetBase::from(test_input.clone());

    let mut test_node: MinMaxRangeScalerNode<f64> =
        MinMaxRangeScalerNode::new(Some(&change_observer));
    let mock_output = flowrs::connection::Edge::new();
    flowrs::connection::connect(test_node.output.clone(), mock_output.clone());
    test_node.data_input.send(dataset)?;
    test_node.on_update()?;

    let expected_data: Array2<f64> = array![
        [
            0.00833333333333334,
            0.041666666666666664,
            0.01666666666666668,
            0.29999999999999993,
            0.26666666666666666,
            0.30833333333333335
        ],
        [
            0.5666666666666667,
            0.5166666666666666,
            0.5416666666666666,
            0.775,
            0.75,
            0.75
        ],
        [1., 1., 1., 0., 0., 0.],
        [0.25, 0.25, 0.25, 0.5, 0.5, 0.5],
        [0.75, 0.75, 0.75, 1., 1., 1.],
        [0., 0., 0., 0.25, 0.25, 0.25],
        [0.5, 0.5, 0.5, 0.75, 0.75, 0.75],
        [1., 1., 1., 0., 0., 0.],
        [0.25, 0.25, 0.25, 0.5, 0.5, 0.5],
        [0.75, 0.75, 0.75, 1., 1., 1.]
    ];
    let actual = mock_output.next()?;
    let expected = DatasetBase::new(expected_data.clone(), expected_data.clone());

    Ok(assert!(expected.records == actual.records))
}

#[test]
fn test_f32() -> Result<(), UpdateError> {
    let change_observer = ChangeObserver::new();
    let mut node: MinMaxRangeScalerNode<f32> = MinMaxRangeScalerNode::new(Some(&change_observer));
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
        [0., 0., 0.33333334, 0.],
        [0.33333334, 0.5, 0.6666667, 0.4],
        [0.6666667, 1., 1., 0.8],
        [1., 0.5, 0., 1.]
    ];

    let actual = mock_output.next()?.records;

    Ok(assert!(expected == actual))
}

#[test]
fn test_f64() -> Result<(), UpdateError> {
    let change_observer = ChangeObserver::new();
    let mut node: MinMaxRangeScalerNode<f64> = MinMaxRangeScalerNode::new(Some(&change_observer));
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
        [0., 0., 0.3333333333333333, 0.],
        [0.3333333333333333, 0.5, 0.6666666666666666, 0.4],
        [0.6666666666666666, 1., 1., 0.8],
        [1., 0.5, 0., 1.]
    ];

    let actual = mock_output.next()?.records;

    Ok(assert!(expected == actual))
}
