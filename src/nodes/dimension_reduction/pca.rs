use flowrs::RuntimeConnectable;
use flowrs::{
    connection::{Input, Output},
    node::{ChangeObserver, Node, UpdateError},
};

use linfa::traits::{Fit, Predict};
use linfa::{DatasetBase, Float};
use linfa_reduction::Pca;
use log::debug;
use ndarray::{array, Array1, Array2};
use serde::{Deserialize, Serialize};

/// Configuration for Principal Component Analysis (PCA).
///
/// The `PCAConfig` struct represents the configuration for Principal Component Analysis (PCA).
/// It specifies the desired dimensionality of the embedding.
#[derive(Clone, Deserialize, Serialize)]
pub struct PCAConfig {
    /// The desired dimensionality of the embedding.
    pub embedding_size: usize,
}

impl PCAConfig {
    /// Create a new instance of `PCAConfig` with the specified embedding size.
    ///
    /// # Parameters
    ///
    /// - `embedding_size`: The desired dimensionality of the embedding.
    ///
    /// # Returns
    ///
    /// A new instance of `PCAConfig`.
    pub fn new(embedding_size: usize) -> Self {
        PCAConfig { embedding_size }
    }
}

/// A runtime connectable node that applies Principal Component Analysis (PCA) to input data.
///
/// The `PCANode` struct is designed to be used as a connectable node within a flowrs-flow.
/// It receives input data and a configuration specifying the desired dimensionality of the embedding.
/// It then applies Principal Component Analysis (PCA) to the input data and sends the resulting
/// embedded data to its output port.
///

#[derive(RuntimeConnectable, Deserialize, Serialize)]
pub struct PCANode<T>
where
    T: Clone + Float,
{
    /// The input port for receiving the PCA configuration.
    #[input]
    pub config_input: Input<PCAConfig>,

    /// The output port for sending the embedded data.
    #[output]
    pub output: Output<DatasetBase<Array2<T>, Array1<()>>>,

    /// The input port for receiving the data to be embedded.
    #[input]
    pub data_input: Input<DatasetBase<Array2<T>, Array1<()>>>,

    /// The configuration for PCA.
    config: PCAConfig,
}

impl<T> PCANode<T>
where
    T: Clone + Float,
{
    /// Create a new instance of `PCANode`.
    ///
    /// # Parameters
    ///
    /// - `change_observer`: An optional reference to a `ChangeObserver` for tracking changes.
    ///
    /// # Returns
    ///
    /// A new instance of `PCANode`.
    pub fn new(change_observer: Option<&ChangeObserver>) -> Self {
        Self {
            config_input: Input::new(),
            data_input: Input::new(),
            output: Output::new(change_observer),
            config: PCAConfig::new(2),
        }
    }
}

impl Node for PCANode<f64> {
    /// Process input data (f64) using Principal Component Analysis (PCA) and send the embedded data to the output port.
    ///
    /// This method is called when the node is updated. It receives both the configuration and the input data,
    /// applies Principal Component Analysis (PCA) to the input data, and sends the resulting embedded data to the output port.
    ///
    /// # Returns
    ///
    /// - `Ok(())`: If the update is successful.
    /// - `Err(UpdateError)`: If an error occurs during the update.
    fn on_update(&mut self) -> Result<(), UpdateError> {
        debug!("PCANode has received an update!");

        // Receiving config
        if let Ok(config) = self.config_input.next() {
            debug!("PCANode has received config: {}", config.embedding_size);
            self.config = config;
        }

        // Receiving data
        if let Ok(data) = self.data_input.next() {
            debug!("PCANode has received data!");

            // Apply Principal Component Analysis (PCA) to the input data
            let embedding = Pca::params(self.config.embedding_size).fit(&data).unwrap();
            let embedded_data = embedding.predict(data);

            // Create a dataset from the embedded data
            let embedded_dataset = DatasetBase::from(embedded_data.targets.clone());

            // Send the embedded data to the output port
            self.output
                .send(embedded_dataset)
                .map_err(|e| UpdateError::Other(e.into()))?;
        }
        Ok(())
    }
}

impl Node for PCANode<f32> {
    /// Process input data (f32) using Principal Component Analysis (PCA) and send the embedded data to the output port.
    ///
    /// This method is called when the node is updated. It receives both the configuration and the input data,
    /// applies Principal Component Analysis (PCA) to the input data, and sends the resulting embedded data to the output port.
    ///
    /// # Returns
    ///
    /// - `Ok(())`: If the update is successful.
    /// - `Err(UpdateError)`: If an error occurs during the update.
    fn on_update(&mut self) -> Result<(), UpdateError> {
        debug!("PCANode has received an update!");

        // Receiving config
        if let Ok(config) = self.config_input.next() {
            debug!("PCANode has received config: {}", config.embedding_size);
            self.config = config;
        }

        // Daten kommen an
        if let Ok(data) = self.data_input.next() {
            debug!("PCANode has received data!");

            // Convert f32 to f64
            let data_f64 = DatasetBase::from(data.records.mapv(|x| x as f64));

            // Apply Principal Component Analysis (PCA) to the input data
            let embedding = Pca::params(self.config.embedding_size)
                .fit(&data_f64)
                .unwrap();
            let red_dataset_target = embedding.predict(data_f64);

            // Create a dataset from the embedded data
            let red_dataset = DatasetBase::from(red_dataset_target.targets.mapv(|x| x as f32));

            // Send the embedded data to the output port
            self.output
                .send(red_dataset)
                .map_err(|e| UpdateError::Other(e.into()))?;
        }
        Ok(())
    }
}

#[test]
fn input_output_test() -> Result<(), UpdateError> {
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
    let test_config_input = PCAConfig { embedding_size: 2 };
    let mut and: PCANode<f64> = PCANode::new(Some(&change_observer));
    let mock_output = flowrs::connection::Edge::new();
    flowrs::connection::connect(and.output.clone(), mock_output.clone());
    and.data_input.send(dataset)?;
    and.config_input.send(test_config_input)?;
    and.on_update()?;

    let expected: Array2<f64> = array![
        [-3.076047733203457, -10.562293260063301],
        [-3.561730416569943, 3.951032231750752],
        [14.63575200500477, 1.1072539713398344],
        [-3.347031741680441, -4.147375003300382],
        [-4.622799446757189, 10.4931265494172],
        [-2.709147889142067, -11.467625779659173],
        [-3.984915594218815, 3.1728757730584096],
        [14.63575200500477, 1.1072539713398344],
        [-3.347031741680441, -4.147375003300382],
        [-4.622799446757189, 10.4931265494172]
    ];

    let actual = mock_output.next()?;

    Ok(assert!(expected == actual.records))
}

#[test]
fn test_f32() -> Result<(), UpdateError> {
    let change_observer: ChangeObserver = ChangeObserver::new();
    let mut node: PCANode<f32> = PCANode::new(Some(&change_observer));
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
        [-4.457307, -1.3018891],
        [-1.2156291, 1.0415113],
        [2.0260487, 3.3849118],
        [3.6468875, -3.124534]
    ];

    let actual = mock_output.next()?.records;
    println!("{}", actual);
    Ok(assert!(expected == actual))
}

#[test]
fn test_f64() -> Result<(), UpdateError> {
    let change_observer: ChangeObserver = ChangeObserver::new();
    let mut node: PCANode<f64> = PCANode::new(Some(&change_observer));
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
        [-4.457306893827564, -1.3018891098082346],
        [-1.215629152862062, 1.0415112878465922],
        [2.02604858810344, 3.384911685501419],
        [3.6468874585861863, -3.1245338635397766]
    ];

    let actual = mock_output.next()?.records;

    Ok(assert!(expected == actual))
}
