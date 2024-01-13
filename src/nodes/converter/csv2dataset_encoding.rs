use flowrs::RuntimeConnectable;
use flowrs::{
    connection::{Input, Output},
    node::{ChangeObserver, Node, UpdateError},
};

use csv::ReaderBuilder;
use linfa::dataset::{DatasetBase, Float};
use log::debug;
use ndarray::{array, concatenate, s, Array1, Array2, ArrayBase, Axis, Dim, OwnedRepr};
use ndarray_csv::Array2Reader;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::{
    collections::{HashMap, HashSet},
    fmt,
    str::FromStr,
};

/// Configuration for data encoding.
///
/// The `EncodingConfig` struct represents the configuration for encoding data read from a CSV file.
/// It specifies the separator used in the CSV file, the list of nominal, ordinal, and other columns.
#[derive(Clone, Deserialize, Serialize)]
pub struct EncodingConfig {
    /// The separator used in the CSV file.
    pub separator: u8,
    /// The list of nominal columns.
    pub nominals: Vec<String>,
    /// The list of ordinal columns.
    pub ordinals: Vec<String>,
    /// The list of other columns.
    pub others: Vec<String>,
}

impl EncodingConfig {
    /// Create a new instance of `EncodingConfig` with the specified configuration parameters.
    ///
    /// # Parameters
    ///
    /// - `separator`: The separator used in the CSV file.
    /// - `nominals`: The list of nominal columns.
    /// - `ordinals`: The list of ordinal columns.
    /// - `others`: The list of other columns.
    ///
    /// # Returns
    ///
    /// A new instance of `EncodingConfig`.
    pub fn new(
        separator: u8,
        nominals: Vec<String>,
        ordinals: Vec<String>,
        others: Vec<String>,
    ) -> Self {
        EncodingConfig {
            separator,
            nominals,
            ordinals,
            others,
        }
    }
}

/// A runtime connectable node that converts CSV data into an encoded dataset.
///
/// The `CSVToEncodedDatasetNode` struct is designed to be used as a connectable node within a flowrs-flow.
/// It receives input CSV data, along with a configuration specifying the data encoding, and sends the resulting
/// encoded dataset to its output port.
///

#[derive(RuntimeConnectable, Deserialize, Serialize)]
pub struct CSVToEncodedDatasetNode<T>
where
    T: Clone + FromStr,
{
    /// The input port for receiving CSV data as a string.
    #[input]
    pub data_input: Input<String>,

    /// The output port for sending the encoded dataset.
    #[output]
    pub output: Output<DatasetBase<Array2<T>, Array1<()>>>,

    /// The input port for receiving the configuration for data encoding.
    #[input]
    pub config_input: Input<EncodingConfig>,

    /// The configuration for data encoding.
    config: EncodingConfig,
}

impl<T> CSVToEncodedDatasetNode<T>
where
    T: Clone + FromStr,
{
    /// Create a new instance of `CSVToEncodedDatasetNode`.
    ///
    /// # Parameters
    ///
    /// - `change_observer`: An optional reference to a `ChangeObserver` for tracking changes.
    ///
    /// # Returns
    ///
    /// A new instance of `CSVToEncodedDatasetNode`.
    pub fn new(change_observer: Option<&ChangeObserver>) -> Self {
        Self {
            data_input: Input::new(),
            config_input: Input::new(),
            output: Output::new(change_observer),
            config: EncodingConfig::new(b',', Vec::new(), Vec::new(), Vec::new()),
        }
    }
}

impl<T> Node for CSVToEncodedDatasetNode<T>
where
    <T as FromStr>::Err: fmt::Debug,
    T: Clone + Send + DeserializeOwned + FromStr + Float,
    f64: Into<T>,
{
    /// Process input CSV data and send the encoded dataset to the output port.
    ///
    /// This method is called when the node is updated. It receives both the configuration and the input CSV data,
    /// performs data encoding based on the configuration, and sends the resulting encoded dataset to the output port.
    ///
    /// # Returns
    ///
    /// - `Ok(())`: If the update is successful.
    /// - `Err(UpdateError)`: If an error occurs during the update.
    fn on_update(&mut self) -> Result<(), UpdateError> {
        debug!("CSVToEncodedDatasetNode has received an update!");

        if let Ok(config) = self.config_input.next() {
            debug!(
                "CSVToEncodedDatasetNode has received config: {}, {:?}, {:?}, {:?}",
                config.separator, config.ordinals, config.nominals, config.others
            );
            self.config = config;
        }

        if let Ok(data) = self.data_input.next() {
            debug!("CSVToEncodedDatasetNode has received data!");

            // Convert the received CSV data into a DatasetBase
            let mut reader = ReaderBuilder::new()
                .delimiter(self.config.separator)
                .has_headers(true)
                .from_reader(data.as_bytes());

            // Skip the header row
            reader.headers().unwrap(); // Use unwrap() for simplicity, handle errors as needed

            let headers: Vec<String> = reader
                .headers()
                .unwrap()
                .into_iter()
                .map(|header| header.to_string())
                .collect();

            let nominal_indices: HashSet<usize> = self
                .config
                .nominals
                .iter()
                .filter_map(|nominal| headers.iter().position(|header| header == nominal))
                .collect();

            let ordinal_indices: HashSet<usize> = self
                .config
                .ordinals
                .iter()
                .filter_map(|ordinal| headers.iter().position(|header| header == ordinal))
                .collect();

            let other_indices: HashSet<usize> = self
                .config
                .others
                .iter()
                .filter_map(|other| headers.iter().position(|header| header == other))
                .collect();

            let mut data_ndarray: Array2<String> = reader.deserialize_array2_dynamic().unwrap(); // Use unwrap() for simplicity, handle errors as needed

            // Separate nominal, ordinal, and other data:
            let mut nominal_data: Vec<Vec<String>> = Vec::new();
            let mut ordinal_data: Vec<Vec<String>> = Vec::new();
            let mut other_data: Vec<Vec<T>> = Vec::new();

            for (col_idx, col) in data_ndarray.axis_iter_mut(Axis(1)).enumerate() {
                if nominal_indices.contains(&col_idx) {
                    nominal_data.push(col.iter().map(|x| x.to_string()).collect());
                } else if ordinal_indices.contains(&col_idx) {
                    ordinal_data.push(col.iter().map(|x| x.to_string()).collect());
                } else if other_indices.contains(&col_idx) {
                    other_data.push(col.iter().map(|x| x.parse().unwrap()).collect());
                }
            }

            // Label encoding on ordinal data
            let mut label_encoded_ordinals: Vec<Vec<T>> = Vec::new();
            for ordinal in ordinal_data.iter() {
                label_encoded_ordinals.push(label_encode(&ordinal));
            }

            // One-hot encoding on nominal data
            let mut one_hot_encoded_nominals: Vec<Array2<T>> = Vec::new();
            let mut one_hot_feature_names: Vec<String> = Vec::new();

            for (nominal, feature_name) in nominal_data.iter().zip(self.config.nominals.iter()) {
                let (one_hot_encoding, feature_names) =
                    one_hot_encode(nominal, feature_name.to_string());
                one_hot_encoded_nominals.push(one_hot_encoding);
                one_hot_feature_names.extend(feature_names);
            }

            // Converting data to ndarrays:
            let nominal_records: Array2<T> = concatenate_arrays(one_hot_encoded_nominals);
            let ordinal_records: Array2<T> = Array2::from_shape_vec(
                (label_encoded_ordinals.len(), headers.len()),
                label_encoded_ordinals.into_iter().flatten().collect(),
            )
            .unwrap();
            let other_records: Array2<T> = Array2::from_shape_vec(
                (other_data.len(), headers.len()),
                other_data.into_iter().flatten().collect(),
            )
            .unwrap();

            // Combine records based on whether nominal_records is empty
            let combined_records: Array2<T> = if nominal_records.is_empty() {
                concatenate![Axis(0), ordinal_records, other_records]
            } else {
                concatenate![Axis(0), nominal_records, ordinal_records, other_records]
            };

            // Combine feature names
            let mut combined_feature_names: Vec<String> = Vec::new();
            combined_feature_names.extend(one_hot_feature_names);
            combined_feature_names.extend(self.config.ordinals.clone());
            combined_feature_names.extend(self.config.others.clone());

            // Convert to DatasetBase with feature names
            let encoded_dataset: DatasetBase<
                ArrayBase<OwnedRepr<T>, Dim<[usize; 2]>>,
                ArrayBase<OwnedRepr<()>, Dim<[usize; 1]>>,
            > = DatasetBase::from(combined_records.t().to_owned())
                .with_feature_names(combined_feature_names);

            self.output
                .send(encoded_dataset)
                .map_err(|e| UpdateError::Other(e.into()))?;
            Ok(())
        } else {
            Err(UpdateError::Other(anyhow::Error::msg(
                "No config received!",
            )))
        }
    }
}

#[test]
fn input_output_test() -> Result<(), UpdateError> {
    let change_observer = ChangeObserver::new();

    let test_data_input = String::from("Age,Food,Rating,Height,Place,Level\n33,Chicken,bad,1.75,Munich,low\n35,Biryani,ok,1.66,London,middle\n74,Kebab,good,1.84,Berlin,high\n62,Chicken,ok,1.63,Munich,middle\n55,Humus,bad,1.94,Berlin,middle\n19,Chicken,good,1.75,Munich,low");
    let test_config_input = EncodingConfig {
        separator: b',',
        nominals: vec!["Food".to_string(), "Place".to_string()],
        ordinals: vec!["Rating".to_string(), "Level".to_string()],
        others: vec!["Age".to_string(), "Height".to_string()],
    };

    let mut and: CSVToEncodedDatasetNode<f64> =
        CSVToEncodedDatasetNode::new(Some(&change_observer));
    let mock_output = flowrs::connection::Edge::new();
    flowrs::connection::connect(and.output.clone(), mock_output.clone());
    and.data_input.send(test_data_input)?;
    and.config_input.send(test_config_input)?;
    and.on_update()?;

    let expected: Array2<f64> = array![
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 33.0, 1.75],
        [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 35.0, 1.66],
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 2.0, 74.0, 1.84],
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 62.0, 1.63],
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 55.0, 1.94],
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 19.0, 1.75]
    ];
    let actual: Array2<f64> = mock_output.next()?.records;

    Ok(assert!(actual == expected))
}

#[test]
fn no_ordinals_test() -> Result<(), UpdateError> {
    let change_observer = ChangeObserver::new();

    let test_data_input = String::from("Age,Food,Rating,Height,Place,Level\n33,Chicken,bad,1.75,Munich,low\n35,Biryani,ok,1.66,London,middle\n74,Kebab,good,1.84,Berlin,high\n62,Chicken,ok,1.63,Munich,middle\n55,Humus,bad,1.94,Berlin,middle\n19,Chicken,good,1.75,Munich,low");
    let test_config_input: EncodingConfig = EncodingConfig {
        separator: b',',
        nominals: vec!["Food".to_string(), "Place".to_string()],
        ordinals: vec![],
        others: vec!["Age".to_string(), "Height".to_string()],
    };

    let mut and: CSVToEncodedDatasetNode<f64> =
        CSVToEncodedDatasetNode::new(Some(&change_observer));
    let mock_output = flowrs::connection::Edge::new();
    flowrs::connection::connect(and.output.clone(), mock_output.clone());
    and.data_input.send(test_data_input)?;
    and.config_input.send(test_config_input)?;
    and.on_update()?;

    let expected: Array2<f64> = array![
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 33.0, 1.75],
        [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 35.0, 1.66],
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 74.0, 1.84],
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 62.0, 1.63],
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 55.0, 1.94],
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 19.0, 1.75]
    ];
    let actual: Array2<f64> = mock_output.next()?.records;

    Ok(assert!(actual == expected))
}

#[test]
fn no_nominals_test() -> Result<(), UpdateError> {
    let change_observer = ChangeObserver::new();

    let test_data_input = String::from("Age,Food,Rating,Height,Place,Level\n33,Chicken,bad,1.75,Munich,low\n35,Biryani,ok,1.66,London,middle\n74,Kebab,good,1.84,Berlin,high\n62,Chicken,ok,1.63,Munich,middle\n55,Humus,bad,1.94,Berlin,middle\n19,Chicken,good,1.75,Munich,low");
    let test_config_input = EncodingConfig {
        separator: b',',
        nominals: vec![],
        ordinals: vec!["Rating".to_string(), "Level".to_string()],
        others: vec!["Age".to_string(), "Height".to_string()],
    };

    let mut and: CSVToEncodedDatasetNode<f64> =
        CSVToEncodedDatasetNode::new(Some(&change_observer));
    let mock_output = flowrs::connection::Edge::new();
    flowrs::connection::connect(and.output.clone(), mock_output.clone());
    and.data_input.send(test_data_input)?;
    and.config_input.send(test_config_input)?;
    and.on_update()?;

    let expected: Array2<f64> = array![
        [0.0, 0.0, 33.0, 1.75],
        [1.0, 1.0, 35.0, 1.66],
        [2.0, 2.0, 74.0, 1.84],
        [1.0, 1.0, 62.0, 1.63],
        [0.0, 1.0, 55.0, 1.94],
        [2.0, 0.0, 19.0, 1.75]
    ];
    let actual: Array2<f64> = mock_output.next()?.records;

    Ok(assert!(actual == expected))
}

fn label_encode<T>(input: &Vec<String>) -> Vec<T>
where
    T: Clone + Copy,
    f64: Into<T>,
{
    // Create a HashMap to store the mapping of unique strings to labels
    let mut label_mapping: HashMap<&String, T> = HashMap::new();

    // Assign labels to unique strings
    let mut label_counter = 0.0;
    let labels: Vec<T> = input
        .iter()
        .map(|s| {
            *label_mapping.entry(s).or_insert_with(|| {
                let label = label_counter;
                label_counter += 1.0;
                label.into()
            })
        })
        .collect();

    labels
}

fn one_hot_encode<T>(input: &Vec<String>, feature_name: String) -> (Array2<T>, Vec<String>)
where
    T: Clone + Float,
    f64: Into<T>,
{
    // Create a HashMap to store the mapping of unique strings to column indices
    let mut column_mapping: HashMap<&String, usize> = HashMap::new();

    // Assign column indices to unique strings
    let mut column_counter = 0;
    let mut feature_names = Vec::new();
    for s in input.iter() {
        if !column_mapping.contains_key(s) {
            column_mapping.insert(s, column_counter);
            let full_feature_name = format!("{}_{}", feature_name, s);
            feature_names.push(full_feature_name);
            column_counter += 1;
        }
    }

    // Create a sparse one-hot encoding matrix
    let rows = input.len();
    let cols = column_counter;
    let mut encoding: ArrayBase<OwnedRepr<T>, Dim<[usize; 2]>> = Array2::zeros((rows, cols));

    for (row_idx, s) in input.iter().enumerate() {
        if let Some(&col_idx) = column_mapping.get(s) {
            encoding[[row_idx, col_idx]] = 1.0.into();
        }
    }

    (encoding, feature_names)
}

fn concatenate_arrays<T>(arrays: Vec<Array2<T>>) -> Array2<T>
where
    T: Clone + Float,
{
    // Check if the vector is not empty
    if arrays.is_empty() {
        return Array2::zeros((0, 0));
    }

    // Get the shape of the first array
    let (rows, cols) = (
        arrays[0].shape()[0],
        arrays.iter().map(|arr| arr.shape()[1]).sum(),
    );

    // Create a new Array2 to hold the concatenated data
    let mut result = Array2::zeros((rows, cols));

    // Iterate over the arrays and fill the result array
    for (col_index, array) in arrays
        .iter()
        .flat_map(|arr| arr.axis_iter(Axis(1)))
        .enumerate()
    {
        // Check if the array has the same number of rows as the first one
        if array.shape()[0] != rows {
            panic!("Inconsistent number of rows in arrays");
        }

        // Copy the data from the current array into the result
        result.slice_mut(s![.., col_index]).assign(&array);
    }

    result.t().to_owned()
}
