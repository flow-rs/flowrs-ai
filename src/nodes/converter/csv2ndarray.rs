use flowrs::RuntimeConnectable;
use flowrs::{
    connection::{Input, Output},
    node::{ChangeObserver, Node, UpdateError},
};

use csv::ReaderBuilder;
use log::debug;
use ndarray::{array, Array2};
use ndarray_csv::Array2Reader;
use serde::{de::DeserializeOwned, Deserialize, Serialize};

/// Configuration for CSV to ndarray conversion.
///
/// The `CSVToNdarrayConfig` struct represents the configuration for converting CSV data into an ndarray.
/// It specifies the separator used in the CSV file and whether the CSV data includes feature names.
#[derive(Clone, Deserialize, Serialize)]
pub struct CSVToNdarrayConfig {
    /// The separator used in the CSV file.
    pub separator: u8,
    /// Indicates whether the CSV data includes feature names in the header row.
    pub has_feature_names: bool,
}

impl CSVToNdarrayConfig {
    /// Create a new instance of `CSVToNdarrayConfig` with the specified configuration parameters.
    ///
    /// # Parameters
    ///
    /// - `separator`: The separator used in the CSV file.
    /// - `has_feature_names`: Indicates whether the CSV data includes feature names in the header row.
    ///
    /// # Returns
    ///
    /// A new instance of `CSVToNdarrayConfig`.
    pub fn new(separator: u8, has_feature_names: bool) -> Self {
        CSVToNdarrayConfig {
            separator: separator,
            has_feature_names: has_feature_names,
        }
    }
}

/// A runtime connectable node that converts CSV data into an ndarray.
///
/// The `CSVToNdarrayNode` struct is designed to be used as a connectable node within a flowrs-flow.
/// It receives input CSV data, along with a configuration specifying the data conversion settings, and sends the resulting ndarray to its output port.
///

#[derive(RuntimeConnectable, Deserialize, Serialize)]
pub struct CSVToNdarrayNode<T>
where
    T: Clone,
{
    /// The output port for sending the resulting ndarray.
    #[output]
    pub output: Output<Array2<T>>,

    /// The input port for receiving CSV data as a string.
    #[input]
    pub data_input: Input<String>,

    /// The input port for receiving the configuration for CSV to ndarray conversion.
    #[input]
    pub config_input: Input<CSVToNdarrayConfig>,

    /// The configuration for CSV to ndarray conversion.
    config: CSVToNdarrayConfig,
}

impl<T> CSVToNdarrayNode<T>
where
    T: Clone,
{
    /// Create a new instance of `CSVToNdarrayNode`.
    ///
    /// # Parameters
    ///
    /// - `change_observer`: An optional reference to a `ChangeObserver` for tracking changes.
    ///
    /// # Returns
    ///
    /// A new instance of `CSVToNdarrayNode`.
    pub fn new(change_observer: Option<&ChangeObserver>) -> Self {
        Self {
            output: Output::new(change_observer),
            data_input: Input::new(),
            config_input: Input::new(),
            config: CSVToNdarrayConfig::new(b',', true),
        }
    }
}

impl<T> Node for CSVToNdarrayNode<T>
where
    T: Clone + Send + DeserializeOwned,
{
    /// Process input CSV data and send the resulting ndarray to the output port.
    ///
    /// This method is called when the node is updated. It receives both the configuration and the input CSV data,
    /// performs data conversion based on the configuration, and sends the resulting ndarray to the output port.
    ///
    /// # Returns
    ///
    /// - `Ok(())`: If the update is successful.
    /// - `Err(UpdateError)`: If an error occurs during the update.
    fn on_update(&mut self) -> Result<(), UpdateError> {
        debug!("CSVToNdarrayNode has received an update!");

        if let Ok(config) = self.config_input.next() {
            debug!(
                "CSVToNdarrayNode has received config: {}, {}",
                config.separator, config.has_feature_names
            );
            self.config = config;
        }

        if let Ok(data) = self.data_input.next() {
            debug!("CSVToNdarrayNode has received data!");

            let mut reader = ReaderBuilder::new()
                .delimiter(self.config.separator)
                .has_headers(self.config.has_feature_names)
                .from_reader(data.as_bytes());
            let data_ndarray = reader
                .deserialize_array2_dynamic()
                .map_err(|e| UpdateError::Other(e.into()))?;

            self.output
                .send(data_ndarray)
                .map_err(|e| UpdateError::Other(e.into()))?;
        }
        Ok(())
    }
}

#[test]
fn input_output_test() -> Result<(), UpdateError> {
    let change_observer = ChangeObserver::new();
    let test_input = String::from("L1,L2,L3\n1,2,3\n4,5,6\n7,8,9");

    let mut and: CSVToNdarrayNode<f64> = CSVToNdarrayNode::new(Some(&change_observer));
    let mock_output: flowrs::connection::Edge<
        ndarray::prelude::ArrayBase<ndarray::OwnedRepr<f64>, ndarray::prelude::Dim<[usize; 2]>>,
    > = flowrs::connection::Edge::new();
    flowrs::connection::connect(and.output.clone(), mock_output.clone());
    and.data_input.send(test_input)?;
    and.on_update()?;

    let expected: Array2<f64> = array![[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]];
    let actual: Array2<f64> = mock_output.next()?;

    Ok(assert!(expected == actual))
}

#[test]
fn test_f32() -> Result<(), UpdateError> {
    let change_observer = ChangeObserver::new();
    let mut node: CSVToNdarrayNode<f32> = CSVToNdarrayNode::new(Some(&change_observer));
    let mock_output = flowrs::connection::Edge::new();
    flowrs::connection::connect(node.output.clone(), mock_output.clone());

    let test_data_input = String::from("L1,L2,L3,L4\n1,2,3,4\n3,4,5,6\n5,6,7,8\n7,4,1,9");

    node.data_input.send(test_data_input.clone())?;
    node.on_update()?;

    let expected = array![
        [1.0, 2.0, 3.0, 4.0],
        [3.0, 4.0, 5.0, 6.0],
        [5.0, 6.0, 7.0, 8.0],
        [7.0, 4.0, 1.0, 9.0]
    ];
    let actual = mock_output.next()?;

    Ok(assert!(expected == actual))
}

#[test]
fn test_f64() -> Result<(), UpdateError> {
    let change_observer = ChangeObserver::new();
    let mut node: CSVToNdarrayNode<f64> = CSVToNdarrayNode::new(Some(&change_observer));
    let mock_output = flowrs::connection::Edge::new();
    flowrs::connection::connect(node.output.clone(), mock_output.clone());

    let test_data_input = String::from("L1,L2,L3,L4\n1,2,3,4\n3,4,5,6\n5,6,7,8\n7,4,1,9");

    node.data_input.send(test_data_input.clone())?;
    node.on_update()?;

    let expected = array![
        [1.0, 2.0, 3.0, 4.0],
        [3.0, 4.0, 5.0, 6.0],
        [5.0, 6.0, 7.0, 8.0],
        [7.0, 4.0, 1.0, 9.0]
    ];
    let actual = mock_output.next()?;

    Ok(assert!(expected == actual))
}
