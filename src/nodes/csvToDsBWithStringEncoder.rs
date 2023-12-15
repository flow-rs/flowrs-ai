
use flowrs::{node::{Node, UpdateError, ChangeObserver}, connection::{Input, Output}};
use flowrs::RuntimeConnectable;

use linfa::dataset::{DatasetBase, Float, Labels, Records};
use linfa_kernel::Inner;
use ndarray::{array, ArrayBase, OwnedRepr, Dim, Axis, Array1, Array2, s};
use csv::ReaderBuilder;
use ndarray_csv::Array2Reader;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::collections::{HashMap, HashSet};
use linfa::traits::Transformer;

use linfa::prelude::*;
use std::error::Error;




#[derive(Clone, Deserialize, Serialize)]
pub struct CSVToEncodedDatasetBaseConfig {
   pub separator: u8,
   pub has_feature_names: bool,
   nominals: Vec<String>,
   ordinals: Vec<String>,
   others: Vec<String>
}

#[derive(RuntimeConnectable, Deserialize, Serialize)]
pub struct CSVToEncodedDatasetBaseNode<T>
where
    T: Clone,
{
    #[input]
    pub config_input: Input<CSVToEncodedDatasetBaseConfig>,

    #[input]
    pub data_input: Input<String>,

    #[output]
    pub output: Output<DatasetBase<Array2<T>, Array1<()>>>,

    data_object: Option<String>
}

impl<T> CSVToEncodedDatasetBaseNode<T>
where
    T: Clone,
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

impl<T> Node for CSVToEncodedDatasetBaseNode<T>
where
    T: Clone + Send + DeserializeOwned,
{
    fn on_update(&mut self) -> Result<(), UpdateError> {
     
        if let Ok(data) = self.data_input.next() {
            println!("JW-Debug CSVToEncodedDatasetBaseNode has received data: {}.", data);
            self.data_object = Some(data);
        }

        if let Some(data) = &self.data_object {
            if let Ok(config) = self.config_input.next() {

                
                // convert String to DatasetBase
                let mut reader = ReaderBuilder::new()
                                                            .delimiter(config.separator)
                                                            .has_headers(config.has_feature_names)
                                                            .from_reader(data.as_bytes());
                let data_ndarray2: Array2<T> = reader.deserialize_array2_dynamic().map_err(|e| UpdateError::Other(e.into()))?;
                let dataset = DatasetBase::from(data_ndarray2);



                //////////////////////////////////////////////////////////////////////////////

                println!("JW-Debug CSVToArrayNNode has received config.");

                let mut reader = ReaderBuilder::new()
                .delimiter(config.separator)
                .has_headers(config.has_feature_names)
                .from_reader(data.as_bytes());
        
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
                let mut other_data: Vec<Vec<f64>> = Vec::new();
                
                for (col_idx, col) in data_ndarray.axis_iter_mut(Axis(1)).enumerate() {
                    if nominal_indices.contains(&col_idx) {
                        nominal_data.push(col.iter().map(|x| x.to_string()).collect());
                    } else if ordinal_indices.contains(&col_idx) {
                        ordinal_data.push(col.iter().map(|x| x.to_string()).collect());
                    } else if other_indices.contains(&col_idx) {
                        other_data.push(col.iter().map(|x| x.parse().unwrap()).collect());
                    }
                }

                let mut label_encoded_ordinals: Vec<Vec<f64>> = Vec::new();
                for ordinal in ordinal_data.iter() {
                    label_encoded_ordinals.push(label_encode(&ordinal));
                }

                // Example usage:
                let mut one_hot_encoded_nominals: Vec<Array2<f64>> = Vec::new();
                let mut one_hot_feature_names: Vec<String> = Vec::new();

                // Assuming `nominal_data` and `config.nominals` are defined somewhere
                for (nominal, feature_name) in nominal_data.iter().zip(config.nominals.iter()) {
                    let (one_hot_encoding, feature_names) = one_hot_encode(nominal, feature_name.to_string());
                    one_hot_encoded_nominals.push(one_hot_encoding);
                    one_hot_feature_names.extend(feature_names);
                }

                // Converting Data to ndarrays:
                let nominal_records = concatenate_arrays(one_hot_encoded_nominals);
                let ordinal_records: Array2<f64> = Array2::from_shape_vec((label_encoded_ordinals.len(), headers.len()), label_encoded_ordinals.into_iter().flatten().collect()).unwrap();
                let other_records: Array2<f64> = Array2::from_shape_vec((other_data.len(), headers.len()), other_data.into_iter().flatten().collect()).unwrap();
                                


                // Label Encoding ordinal records
                //let label_encoded_ordinals = label_encode_ordinal(&ordinal_records);

                // One hot encoding nominal_records
                //let (one_hot_encoded_nominals, nominal_feature_names) = one_hot_encode(&nominal_records);
                //println!("nominal features: {:?}", nominal_feature_names);
                //println!("One hot encoding: {:?}", one_hot_encoded_nominals);

                // Converting ndarrays to DatasetBase
                //let nominals_dbs:DatasetBase<ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>, ArrayBase<OwnedRepr<()>, Dim<[usize; 1]>>> = DatasetBase::from(one_hot_encoded_nominals.clone());
                //let nominals:DatasetBase<ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>, ArrayBase<OwnedRepr<()>, Dim<[usize; 1]>>> = nominals_dbs.with_feature_names(nominal_feature_names);
                //let ordinals_dbs:DatasetBase<ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>, ArrayBase<OwnedRepr<()>, Dim<[usize; 1]>>> = DatasetBase::from(label_encoded_ordinals.clone());
                //let ordinals:DatasetBase<ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>, ArrayBase<OwnedRepr<()>, Dim<[usize; 1]>>> = ordinals_dbs.with_feature_names(config.ordinals);
                //let others_dbs:DatasetBase<ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>, ArrayBase<OwnedRepr<()>, Dim<[usize; 1]>>> = DatasetBase::from(other_records.clone());
                //let others = others_dbs.with_feature_names(config.others);

                println!("Nominals: {:?}", nominal_records);
                println!("Feature names: {:?}", one_hot_feature_names);
                println!("Ordinal records: {:?}", ordinal_records);
                //println!("Ordinals: {:?}", ordinals);
                println!("Others: {:?}", other_records);

                //let nominal_dataset:DatasetBase<_, _> = DatasetBase::from(nominal_records);
                //let other_dataset: DatasetBase<f64, f64> = DatasetBase::from(Records::new(other_records, None));



                // load data in a vector
                let data: Vec<String> = vec!["hello".to_string(),
                "world".to_string(),
                "world".to_string(),
                "world".to_string(),
                "world".to_string(),
                "again".to_string(),
                "hello".to_string(),
                "again".to_string(),
                "goodbye".to_string()];

                // Apply label encoding
                let encoded_labels: Vec<f64> = label_encode(&data.clone());

                // Apply one-hot encoding
                let (one_hot_encoding, feature_names) = one_hot_encode(&data, "text".to_string());


                // Display the result
                println!("Encoded Labels: {:?}", encoded_labels);
                println!("Feature Names: {:?}", feature_names);
                println!("One-Hot Encoding:\n{:?}", one_hot_encoding);
                



                //////////////////////////////////////////////////////////////////////////////


                self.output.send(dataset).map_err(|e| UpdateError::Other(e.into()))?;
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
    let test_config_input = CSVToEncodedDatasetBaseConfig{
        separator: b',',
        has_feature_names: true,
        nominals: vec!["Feate1".to_string(), "Feature2".to_string()],
        ordinals: vec!["F5".to_string(), "F6".to_string()],
        others: vec!["Feature3".to_string(), "F4".to_string()]
    };

    let mut and: CSVToEncodedDatasetBaseNode<u32> = CSVToEncodedDatasetBaseNode::new(Some(&change_observer));
    let mock_output = flowrs::connection::Edge::new();
    flowrs::connection::connect(and.output.clone(), mock_output.clone());
    and.data_input.send(test_data_input)?;
    and.config_input.send(test_config_input)?;
    and.on_update()?;

    let expected: Array2<u32> = array![[1, 2, 3, 1, 2, 11], [4, 5, 6, 1, 5, 3], [7, 8, 9, 1, 2, 3], [10, 8, 9, 1, 6, 4], [12, 8, 9, 1, 2, 3], [7, 8, 9, 1, 2, 3]];
    let actual: Array2<u32> = mock_output.next()?.records;
    
    Ok(assert!(expected == actual))
}

fn label_encode(input: &Vec<String>) -> Vec<f64> {
    // Create a HashMap to store the mapping of unique strings to labels
    let mut label_mapping: HashMap<&String, f64> = HashMap::new();

    // Assign labels to unique strings
    let mut label_counter = 0.0;
    let labels: Vec<f64> = input
        .iter()
        .map(|s| {
            *label_mapping.entry(s).or_insert_with(|| {
                let label = label_counter;
                label_counter += 1.0;
                label
            })
        })
        .collect();

    labels
}

fn one_hot_encode(input: &Vec<String>, feature_name: String) -> (Array2<f64>, Vec<String>) {
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
    let mut encoding: ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>> = Array2::zeros((rows, cols));

    for (row_idx, s) in input.iter().enumerate() {
        if let Some(&col_idx) = column_mapping.get(s) {
            encoding[[row_idx, col_idx]] = 1.0;
        }
    }

    (encoding, feature_names)
}

fn concatenate_arrays(mut arrays: Vec<Array2<f64>>) -> Array2<f64> {
    // Check if the vector is not empty
    if arrays.is_empty() {
        panic!("Input vector is empty");
    }

    // Get the shape of the first array
    let (rows, cols) = (arrays[0].shape()[0], arrays.iter().map(|arr| arr.shape()[1]).sum());

    // Create a new Array2 to hold the concatenated data
    let mut result = Array2::zeros((rows, cols));

    // Starting column index in the result array
    let mut col_index = 0;

    // Iterate over the arrays and fill the result array
    for array in arrays {
        // Check if the array has the same number of rows as the first one
        if array.shape()[0] != rows {
            panic!("Inconsistent number of rows in arrays");
        }

        // Get the number of columns in the current array
        let current_cols = array.shape()[1];

        // Copy the data from the current array into the result
        result.slice_mut(s![.., col_index..col_index + current_cols]).assign(&array);

        // Update the column index for the next array
        col_index += current_cols;
    }

    result
}