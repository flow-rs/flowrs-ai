use flowrs::RuntimeConnectable;
use flowrs::{
    connection::{Input, Output},
    node::{ChangeObserver, Node, UpdateError},
};

use linfa::traits::Transformer;
use linfa::{dataset::DatasetBase, Float};
use linfa_clustering::Dbscan;
use log::debug;
use ndarray::{array, Array1, Array2};
use serde::{Deserialize, Serialize};

/// Configuration for the DBSCAN clustering node.
#[derive(Clone, Deserialize, Serialize)]
pub struct DbscanConfig {
    /// Minimum number of points required to form a cluster.
    pub min_points: usize,
    /// Tolerance (maximum distance) for clustering.
    pub tolerance: f64,
}

impl DbscanConfig {
    /// Create a new instance of `DbscanConfig` with the specified parameters.
    pub fn new(min_points: usize, tolerance: f64) -> Self {
        DbscanConfig {
            min_points,
            tolerance,
        }
    }
}

/// A runtime connectable node that performs DBSCAN clustering on input data.
///
/// The `DbscanNode` struct is designed to be used as a connectable node within a flowrs-flow.
/// It receives input data and a configuration specifying the clustering parameters.
/// It then applies DBSCAN to the input data and sends the resulting
/// data to its output port.
///

#[derive(RuntimeConnectable, Deserialize, Serialize)]
pub struct DbscanNode<T>
where
    T: Clone + Float,
{
    /// The input port for receiving the configuration for DBSCAN clustering.
    #[input]
    pub config_input: Input<DbscanConfig>,

    /// The output port for sending the resulting clustered dataset.
    #[output]
    pub output: Output<DatasetBase<Array2<T>, Array1<Option<usize>>>>,

    /// The input port for receiving the dataset to be clustered.
    #[input]
    pub data_input: Input<DatasetBase<Array2<T>, Array1<()>>>,

    /// The configuration for DBSCAN clustering.
    config: DbscanConfig,
}

impl<T> DbscanNode<T>
where
    T: Clone + Float,
{
    /// Create a new instance of `DbscanNode`.
    ///
    /// # Parameters
    ///
    /// - `change_observer`: An optional reference to a `ChangeObserver` for tracking changes.
    ///
    /// # Returns
    ///
    /// A new instance of `DbscanNode`.
    pub fn new(change_observer: Option<&ChangeObserver>) -> Self {
        Self {
            config_input: Input::new(),
            data_input: Input::new(),
            output: Output::new(change_observer),
            config: DbscanConfig::new(2, 0.5),
        }
    }
}

impl<T> Node for DbscanNode<T>
where
    T: Clone + Send + Float,
{
    /// Run DBSCAN clustering on input data and send the clustered dataset to the output port.
    ///
    /// This method is called when the node is updated. It receives the configuration for DBSCAN clustering
    /// as well as the input dataset to be clustered. The DBSCAN clustering algorithm is applied to the data,
    /// and the resulting clustered dataset is sent to the output port.
    ///
    /// # Returns
    ///
    /// - `Ok(())`: If the update is successful.
    /// - `Err(UpdateError)`: If an error occurs during the update.
    fn on_update(&mut self) -> Result<(), UpdateError> {
        debug!("DbscanNode has received an update!");

        // receiving config
        if let Ok(config) = self.config_input.next() {
            debug!(
                "[DEBUG::DbscanNode] New Config:\n min_points: {},\n tolerance: {}",
                config.min_points, config.tolerance
            );
            self.config = config;
        }

        // receiving data
        if let Ok(data) = self.data_input.next() {
            debug!(
                "[DEBUG::DbscanNode] Received Data:\n {}",
                data.records.clone()
            );

            let clusters = Dbscan::params(self.config.min_points)
                .tolerance(T::from(self.config.tolerance).unwrap())
                .transform(data)
                .unwrap();

            debug!(
                "[DEBUG::DbscanNode] Sent Data:\n Records: {},\n Targets: {:?}",
                clusters.records.clone(),
                clusters.targets.clone()
            );
            self.output
                .send(clusters)
                .map_err(|e| UpdateError::Other(e.into()))?;

            debug!("DbscanNode has sent an output!");
        }

        Ok(())
    }
}

#[test]
fn new_config_test() -> Result<(), UpdateError> {
    let change_observer = ChangeObserver::new();
    let record_input: Array2<f64> = array![
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
    let input_data = DatasetBase::from(record_input);
    let test_config_input: DbscanConfig = DbscanConfig {
        min_points: 2,
        tolerance: 0.5,
    };

    let mut and: DbscanNode<f64> = DbscanNode::new(Some(&change_observer));
    let mock_output = flowrs::connection::Edge::new();
    flowrs::connection::connect(and.output.clone(), mock_output.clone());
    and.data_input.send(input_data)?;
    and.config_input.send(test_config_input)?;
    and.on_update()?;

    let expected: Array1<Option<usize>> = array![
        None,
        None,
        Some(0),
        Some(1),
        Some(2),
        None,
        None,
        Some(0),
        Some(1),
        Some(2)
    ];
    let actual: DatasetBase<Array2<f64>, Array1<Option<usize>>> = mock_output.next()?;

    Ok(assert!(expected == actual.targets))
}

#[test]
fn default_config_test() -> Result<(), UpdateError> {
    let change_observer = ChangeObserver::new();
    let record_input: Array2<f64> = array![
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
    let input_data = DatasetBase::from(record_input);

    let mut and: DbscanNode<f64> = DbscanNode::new(Some(&change_observer));
    let mock_output = flowrs::connection::Edge::new();
    flowrs::connection::connect(and.output.clone(), mock_output.clone());
    and.data_input.send(input_data)?;
    and.on_update()?;

    let expected: Array1<Option<usize>> = array![
        None,
        None,
        Some(0),
        Some(1),
        Some(2),
        None,
        None,
        Some(0),
        Some(1),
        Some(2)
    ];
    let actual: DatasetBase<Array2<f64>, Array1<Option<usize>>> = mock_output.next()?;

    Ok(assert!(expected == actual.targets))
}

#[test]
fn test_f32() -> Result<(), UpdateError> {
    let change_observer = ChangeObserver::new();
    let mut node: DbscanNode<f32> = DbscanNode::new(Some(&change_observer));
    let mock_output = flowrs::connection::Edge::new();
    flowrs::connection::connect(node.output.clone(), mock_output.clone());

    let test_data = array![
        [1.0, 2.0, 3.0, 4.0],
        [3.0, 4.0, 5.0, 6.0],
        [5.0, 6.0, 7.0, 8.0],
        [7.0, 4.0, 1.0, 9.0]
    ];
    let test_data_input = DatasetBase::from(test_data.clone());

    node.data_input.send(test_data_input)?;
    node.on_update()?;

    let expected_targets = array![None, None, None, None];

    let actual = mock_output.next()?;
    let actual_records = actual.records;
    let actual_targets = actual.targets;

    Ok(assert!(
        (test_data == actual_records) && (expected_targets == actual_targets)
    ))
}

#[test]
fn test_f64() -> Result<(), UpdateError> {
    let change_observer = ChangeObserver::new();
    let mut node: DbscanNode<f64> = DbscanNode::new(Some(&change_observer));
    let mock_output = flowrs::connection::Edge::new();
    flowrs::connection::connect(node.output.clone(), mock_output.clone());

    let test_data = array![
        [1.0, 2.0, 3.0, 4.0],
        [3.0, 4.0, 5.0, 6.0],
        [5.0, 6.0, 7.0, 8.0],
        [7.0, 4.0, 1.0, 9.0]
    ];
    let test_data_input = DatasetBase::from(test_data.clone());

    node.data_input.send(test_data_input)?;
    node.on_update()?;

    let expected_targets = array![None, None, None, None];

    let actual = mock_output.next()?;
    let actual_records = actual.records;
    let actual_targets = actual.targets;

    Ok(assert!(
        (test_data == actual_records) && (expected_targets == actual_targets)
    ))
}
