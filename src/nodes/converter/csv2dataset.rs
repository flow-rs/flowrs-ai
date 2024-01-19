use flowrs::RuntimeConnectable;
use flowrs::{
    connection::{Input, Output},
    node::{ChangeObserver, Node, UpdateError},
};

use csv::ReaderBuilder;
use linfa::DatasetBase;
use log::debug;
use ndarray::Array2;
use ndarray::{array, Array1};
use ndarray_csv::Array2Reader;
use serde::{de::DeserializeOwned, Deserialize, Serialize};

/// Configuration for CSV to dataset conversion.
///
/// The `CSVToDatasetConfig` struct represents the configuration for converting CSV data into a dataset.
/// It specifies the separator used in the CSV file and whether the CSV data includes feature names.
#[derive(Clone, Deserialize, Serialize)]
pub struct CSVToDatasetConfig {
    /// The separator used in the CSV file.
    pub separator: u8,
    /// Indicates whether the CSV data includes feature names in the header row.
    pub has_feature_names: bool,
}

impl CSVToDatasetConfig {
    /// Create a new instance of `CSVToDatasetConfig` with the specified configuration parameters.
    ///
    /// # Parameters
    ///
    /// - `separator`: The separator used in the CSV file.
    /// - `has_feature_names`: Indicates whether the CSV data includes feature names in the header row.
    ///
    /// # Returns
    ///
    /// A new instance of `CSVToDatasetConfig`.
    pub fn new(separator: u8, has_feature_names: bool) -> Self {
        CSVToDatasetConfig {
            separator: separator,
            has_feature_names: has_feature_names,
        }
    }
}

/// A runtime connectable node that converts CSV data into a dataset.
///
/// The `CSVToDatasetNode` struct is designed to be used as a connectable node within a flowrs-flow.
/// It receives input CSV data, along with a configuration specifying the data conversion settings, and sends the resulting dataset to its output port.
///

#[derive(RuntimeConnectable, Deserialize, Serialize)]
pub struct CSVToDatasetNode<T>
where
    T: Clone,
{
    /// The input port for receiving the configuration for CSV to dataset conversion.
    #[input]
    pub config_input: Input<CSVToDatasetConfig>,

    /// The input port for receiving CSV data as a string.
    #[input]
    pub data_input: Input<String>,

    /// The output port for sending the resulting dataset.
    #[output]
    pub output: Output<DatasetBase<Array2<T>, Array1<()>>>,

    /// The configuration for CSV to dataset conversion.
    config: CSVToDatasetConfig,
}

impl<T> CSVToDatasetNode<T>
where
    T: Clone,
{
    /// Create a new instance of `CSVToDatasetNode`.
    ///
    /// # Parameters
    ///
    /// - `change_observer`: An optional reference to a `ChangeObserver` for tracking changes.
    ///
    /// # Returns
    ///
    /// A new instance of `CSVToDatasetNode`.
    pub fn new(change_observer: Option<&ChangeObserver>) -> Self {
        Self {
            data_input: Input::new(),
            config_input: Input::new(),
            output: Output::new(change_observer),
            config: CSVToDatasetConfig::new(b',', false),
        }
    }
}

impl<T> Node for CSVToDatasetNode<T>
where
    T: Clone + Send + DeserializeOwned + std::fmt::Display, // for debugging
{
    /// Process input CSV data and send the resulting dataset to the output port.
    ///
    /// This method is called when the node is updated. It receives both the configuration and the input CSV data,
    /// performs data conversion based on the configuration, and sends the resulting dataset to the output port.
    ///
    /// # Returns
    ///
    /// - `Ok(())`: If the update is successful.
    /// - `Err(UpdateError)`: If an error occurs during the update.
    fn on_update(&mut self) -> Result<(), UpdateError> {
        debug!("CSVToDatasetNode has received an update!");

        if let Ok(config) = self.config_input.next() {
            debug!(
                "CSVToDatasetNode has received config: {}, {}",
                config.separator, config.has_feature_names
            );
            self.config = config;
        }

        // Receiving data
        if let Ok(data) = self.data_input.next() {
            debug!("CSVToDatasetNode has received data!");

            // Convert String to DatasetBase
            let mut reader = ReaderBuilder::new()
                .delimiter(self.config.separator)
                .has_headers(self.config.has_feature_names)
                .from_reader(data.as_bytes());

            let data_ndarray: Array2<T> = reader
                .deserialize_array2_dynamic()
                .map_err(|e| UpdateError::Other(e.into()))?;

            let dataset = DatasetBase::from(data_ndarray);

            // Get feature names if available
            if self.config.has_feature_names {
                let mut feature_names: Vec<String> = Vec::new();
                for element in reader
                    .headers()
                    .map_err(|e| UpdateError::Other(e.into()))?
                    .into_iter()
                {
                    feature_names.push(String::from(element));
                }
                let dataset_with_features = dataset.with_feature_names(feature_names);

                self.output
                    .send(dataset_with_features)
                    .map_err(|e| UpdateError::Other(e.into()))?;
                return Ok(());
            }

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
    let test_data_input = String::from("Feate1,Feature2,Feature3\n1,2,3\n4,5,6\n7,8,9");
    let test_config_input = CSVToDatasetConfig {
        separator: b',',
        has_feature_names: true,
    };

    let mut and: CSVToDatasetNode<u32> = CSVToDatasetNode::new(Some(&change_observer));
    let mock_output = flowrs::connection::Edge::new();
    flowrs::connection::connect(and.output.clone(), mock_output.clone());
    and.data_input.send(test_data_input)?;
    and.config_input.send(test_config_input)?;
    and.on_update()?;

    let expected: Array2<u32> = array![[1, 2, 3], [4, 5, 6], [7, 8, 9]];
    let actual: Array2<u32> = mock_output.next()?.records;

    Ok(assert!(expected == actual))
}

#[test]
fn test_f32() -> Result<(), UpdateError> {
    let change_observer = ChangeObserver::new();
    let mut node: CSVToDatasetNode<f32> = CSVToDatasetNode::new(Some(&change_observer));
    let mock_output = flowrs::connection::Edge::new();
    flowrs::connection::connect(node.output.clone(), mock_output.clone());

    let test_data_input = String::from("1,2,3,4\n3,4,5,6\n5,6,7,8\n7,4,1,9");

    node.data_input.send(test_data_input.clone())?;
    node.on_update()?;

    let expected = array![
        [1.0, 2.0, 3.0, 4.0],
        [3.0, 4.0, 5.0, 6.0],
        [5.0, 6.0, 7.0, 8.0],
        [7.0, 4.0, 1.0, 9.0]
    ];
    let actual = mock_output.next()?.records;

    Ok(assert!(expected == actual))
}

#[test]
fn test_f64() -> Result<(), UpdateError> {
    let change_observer = ChangeObserver::new();
    let mut node: CSVToDatasetNode<f64> = CSVToDatasetNode::new(Some(&change_observer));
    let mock_output = flowrs::connection::Edge::new();
    flowrs::connection::connect(node.output.clone(), mock_output.clone());

    let test_data_input = String::from("1,2,3,4\n3,4,5,6\n5,6,7,8\n7,4,1,9");

    node.data_input.send(test_data_input.clone())?;
    node.on_update()?;

    let expected = array![
        [1.0, 2.0, 3.0, 4.0],
        [3.0, 4.0, 5.0, 6.0],
        [5.0, 6.0, 7.0, 8.0],
        [7.0, 4.0, 1.0, 9.0]
    ];
    let actual = mock_output.next()?.records;

    Ok(assert!(expected == actual))
}
