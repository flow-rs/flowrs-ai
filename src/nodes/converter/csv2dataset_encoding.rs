
use flowrs::{node::{Node, UpdateError, ChangeObserver}, connection::{Input, Output}};
use flowrs::RuntimeConnectable;

use linfa::dataset::{DatasetBase, Float};
use ndarray::{array, ArrayBase, OwnedRepr, Dim, Axis, Array1, Array2, s, concatenate};
use csv::ReaderBuilder;
use ndarray_csv::Array2Reader;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::{collections::{HashMap, HashSet}, str::FromStr, fmt};


#[derive(Clone, Deserialize, Serialize)]
pub struct CSVToEncodedDatasetConfig {
   pub separator: u8,
   pub has_feature_names: bool,
   nominals: Vec<String>,
   ordinals: Vec<String>,
   others: Vec<String>
}

#[derive(RuntimeConnectable, Deserialize, Serialize)]
pub struct CSVToEncodedDatasetNode<T>
where
    T: Clone
{
    #[input]
    pub config_input: Input<CSVToEncodedDatasetConfig>,

    #[input]
    pub data_input: Input<String>,

    #[output]
    pub output: Output<DatasetBase<Array2<T>, Array1<()>>>,

    data_object: Option<String>
}

impl<T> CSVToEncodedDatasetNode<T>
where
    T: Clone
{
    pub fn new(change_observer: Option<&ChangeObserver>) -> Self {
        Self {
            data_input: Input::new(),
            config_input: Input::new(),
            output: Output::new(change_observer),
            data_object : Option::None
        }
    }
}

impl<T> Node for CSVToEncodedDatasetNode<T>
where
    <T as FromStr>::Err: fmt::Debug,
    T: Clone + Send + DeserializeOwned + FromStr + Float,
    f64: Into<T>
{
    fn on_update(&mut self) -> Result<(), UpdateError> {
     
        if let Ok(data) = self.data_input.next() {
            println!("JW-Debug CSVToEncodedDatasetNode has received data: {}.", data);
            self.data_object = Some(data);
        }

        if let Some(data) = &self.data_object {
            if let Ok(config) = self.config_input.next() {

                
                // convert String to DatasetBase
                let mut reader = ReaderBuilder::new()
                                                            .delimiter(config.separator)
                                                            .has_headers(config.has_feature_names)
                                                            .from_reader(data.as_bytes());

                println!("JW-Debug CSVToArrayNNode has received config.");
        
                // Skip the header row
                reader.headers().unwrap(); // Use unwrap() for simplicity, handle errors as needed
            
                let headers: Vec<String> = reader.headers().unwrap().into_iter().map(|header| header.to_string()).collect();
            
                let nominal_indices: HashSet<usize> = config.nominals.iter()
                    .filter_map(|nominal| headers.iter().position(|header| header == nominal))
                    .collect();
            
                let ordinal_indices: HashSet<usize> = config.ordinals.iter()
                    .filter_map(|ordinal| headers.iter().position(|header| header == ordinal))
                    .collect();

                let other_indices: HashSet<usize> = config.others.iter()
                    .filter_map(|other| headers.iter().position(|header| header == other))
                    .collect();

                let mut data_ndarray: Array2<String> = reader.deserialize_array2_dynamic().unwrap(); // Use unwrap() for simplicity, handle errors as needed
            

                // Separating Nominal and Other Data:
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

                let mut label_encoded_ordinals: Vec<Vec<T>> = Vec::new();
                for ordinal in ordinal_data.iter() {
                    label_encoded_ordinals.push(label_encode(&ordinal));
                }

                // Example usage:
                let mut one_hot_encoded_nominals: Vec<Array2<T>> = Vec::new();
                let mut one_hot_feature_names: Vec<String> = Vec::new();

                // Assuming `nominal_data` and `config.nominals` are defined somewhere
                for (nominal, feature_name) in nominal_data.iter().zip(config.nominals.iter()) {
                    let (one_hot_encoding, feature_names) = one_hot_encode(nominal, feature_name.to_string());
                    one_hot_encoded_nominals.push(one_hot_encoding);
                    one_hot_feature_names.extend(feature_names);
                }

                // Converting Data to ndarrays:
                let nominal_records: Array2<T> = concatenate_arrays(one_hot_encoded_nominals);
                let ordinal_records: Array2<T> = Array2::from_shape_vec((label_encoded_ordinals.len(), headers.len()), label_encoded_ordinals.into_iter().flatten().collect()).unwrap();
                let other_records: Array2<T> = Array2::from_shape_vec((other_data.len(), headers.len()), other_data.into_iter().flatten().collect()).unwrap();

                let combined_records: Array2<T> = concatenate![
                    Axis(0),
                    nominal_records,
                    ordinal_records,
                    other_records
                ];

                let mut combined_feature_names: Vec<String> = Vec::new();
                combined_feature_names.extend(one_hot_feature_names);
                combined_feature_names.extend(config.ordinals);
                combined_feature_names.extend(config.others);

                let encoded_dataset:DatasetBase<ArrayBase<OwnedRepr<T>, Dim<[usize; 2]>>, ArrayBase<OwnedRepr<()>, Dim<[usize; 1]>>> = DatasetBase::from(combined_records.t().to_owned()).with_feature_names(combined_feature_names);
                println!("Encoded dataset: {:?}", encoded_dataset);

                //////////////////////////////////////////////////////////////////////////////


                self.output.send(encoded_dataset).map_err(|e| UpdateError::Other(e.into()))?;
                Ok(())
            } else {
                Err(UpdateError::Other(anyhow::Error::msg("No config received!")))
            }

        } else {
            Err(UpdateError::Other(anyhow::Error::msg("No data received!")))
        }         
    }
}


#[test]
fn input_output_test() -> Result<(), UpdateError> {
    let change_observer = ChangeObserver::new();
    let test_data_input = String::from("Feate1,Feature2,Feature3,F4,F5,F6\n1,2,3,1,2,11\n4,5,6,1,5,3\n7,8,9,1,2,3\n10,8,9,1,6,4\n12,8,9,1,2,3\n7,8,9,1,2,3");
    let test_config_input = CSVToEncodedDatasetConfig{
        separator: b',',
        has_feature_names: true,
        nominals: vec!["Feate1".to_string(), "Feature2".to_string()],
        ordinals: vec!["F5".to_string(), "F6".to_string()],
        others: vec!["Feature3".to_string(), "F4".to_string()]
    };

    let mut and: CSVToEncodedDatasetNode<f64> = CSVToEncodedDatasetNode::new(Some(&change_observer));
    let mock_output = flowrs::connection::Edge::new();
    flowrs::connection::connect(and.output.clone(), mock_output.clone());
    and.data_input.send(test_data_input)?;
    and.config_input.send(test_config_input)?;
    and.on_update()?;

    let expected: Array2<f64> = array![[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 3.0, 1.0],
                                        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 6.0, 1.0],
                                        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 9.0, 1.0],
                                        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 2.0, 9.0, 1.0],
                                        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 9.0, 1.0],
                                        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 9.0, 1.0]];
    let actual: Array2<f64> = mock_output.next()?.records;
    
    Ok(assert!(expected == actual))
}

fn label_encode<T>(input: &Vec<String>) -> Vec<T> 
where
    T: Clone + Copy,
    f64: Into<T>
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
    f64: Into<T>
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
    T: Clone + Float
{
    // Check if the vector is not empty
    if arrays.is_empty() {
        panic!("Input vector is empty");
    }

    // Get the shape of the first array
    let (rows, cols) = (arrays[0].shape()[0], arrays.iter().map(|arr| arr.shape()[1]).sum());

    // Create a new Array2 to hold the concatenated data
    let mut result = Array2::zeros((rows, cols));

    // Iterate over the arrays and fill the result array
    for (col_index, array) in arrays.iter().flat_map(|arr| arr.axis_iter(Axis(1))).enumerate() {
        // Check if the array has the same number of rows as the first one
        if array.shape()[0] != rows {
            panic!("Inconsistent number of rows in arrays");
        }

        // Copy the data from the current array into the result
        result.slice_mut(s![.., col_index]).assign(&array);
    }

    result.t().to_owned()
}