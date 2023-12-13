
// use flowrs::{node::{Node, UpdateError, ChangeObserver}, connection::{Input, Output}};
// use flowrs::RuntimeConnectable;

use linfa::dataset::{DatasetBase, Float, Labels, Records};
use ndarray::{array, ArrayBase, OwnedRepr, Dim, Axis, Array1, Array2};
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
   ordinals: Vec<String>
}

// #[derive(RuntimeConnectable, Deserialize, Serialize)]
// pub struct CSVToEncodedDatasetBaseNode<T>
// where
//     T: Clone,
// {
//     #[input]
//     pub config_input: Input<CSVToEncodedDatasetBaseConfig>,

//     #[input]
//     pub data_input: Input<String>,

    #[output]
    pub output: Output<DatasetBase<Array2<T>, Array1<()>>>,

//     data_object: Option<String>
// }

// impl<T> CSVToEncodedDatasetBaseNode<T>
// where
//     T: Clone,
// {
//     pub fn new(change_observer: Option<&ChangeObserver>) -> Self {
//         Self {
//             data_input: Input::new(),
//             config_input: Input::new(),
//             output: Output::new(change_observer),
//             data_object : Option::None
//         }
//     }
// }

// impl<T> Node for CSVToEncodedDatasetBaseNode<T>
// where
//     T: Clone + Send + DeserializeOwned,
// {
//     fn on_update(&mut self) -> Result<(), UpdateError> {
     
//         if let Ok(data) = self.data_input.next() {
//             println!("JW-Debug CSVToEncodedDatasetBaseNode has received data: {}.", data);
//             self.data_object = Some(data);
//         }

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

//                 let mut reader = ReaderBuilder::new()
//                 .delimiter(config.separator)
//                 .has_headers(config.has_feature_names)
//                 .from_reader(data.as_bytes());
        
                // Skip the header row
                reader.headers().unwrap(); // Use unwrap() for simplicity, handle errors as needed
            
                let headers: Vec<String> = reader.headers().unwrap().into_iter().map(|header| header.to_string()).collect();
            
                let nominal_indices: HashSet<usize> = config.nominals.iter()
                    .filter_map(|nominal| headers.iter().position(|header| header == nominal))
                    .collect();
            
                let ordinal_indices: HashSet<usize> = config.ordinals.iter()
                    .filter_map(|ordinal| headers.iter().position(|header| header == ordinal))
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
                    } else {
                        other_data.push(col.iter().map(|x| x.parse().unwrap()).collect());
                    }
                }

                // Converting Data to Records:
                let nominal_records: Array2<String> = Array2::from_shape_vec((nominal_data.len(), headers.len()), nominal_data.into_iter().flatten().collect()).unwrap();
                let ordinal_records: Array2<String> = Array2::from_shape_vec((ordinal_data.len(), headers.len()), ordinal_data.into_iter().flatten().collect()).unwrap();
                let other_records: Array2<f64> = Array2::from_shape_vec((other_data.len(), headers.len()), other_data.into_iter().flatten().collect()).unwrap();
                                
                println!("Nominals: {:?}", nominal_records);
                println!("Ordinals: {:?}", ordinal_records);
                println!("Others: {:?}", other_records);

                //let nominal_dataset:DatasetBase<_, _> = DatasetBase::from(nominal_records);
                //let other_dataset: DatasetBase<f64, f64> = DatasetBase::from(Records::new(other_records, None));



                //////////////////////////////////////////////////////////////////////////////


                self.output.send(dataset).map_err(|e| UpdateError::Other(e.into()))?;
                Ok(())
            } else {
                Err(UpdateError::Other(anyhow::Error::msg("No config received!")))
            }

//         } else {
//             Err(UpdateError::Other(anyhow::Error::msg("No data received!")))
//         }         
//     }
// }


#[test]
fn input_output_test() -> Result<(), UpdateError> {
    let change_observer = ChangeObserver::new();
    let test_data_input = String::from("Feate1,Feature2,Feature3,F4,F5,F6\n1,2,3,1,2,3\n4,5,6,1,2,3\n7,8,9,1,2,3\n7,8,9,1,2,3\n7,8,9,1,2,3\n7,8,9,1,2,3");
    let test_config_input = CSVToEncodedDatasetBaseConfig{
        separator: b',',
        has_feature_names: true,
        nominals: vec!["Feature2".to_string(), "F5".to_string(), "F6".to_string()],
        ordinals: vec!["Feate1".to_string()]
    };

//     let mut and: CSVToEncodedDatasetBaseNode<u32> = CSVToEncodedDatasetBaseNode::new(Some(&change_observer));
//     let mock_output = flowrs::connection::Edge::new();
//     flowrs::connection::connect(and.output.clone(), mock_output.clone());
//     and.data_input.send(test_data_input)?;
//     and.config_input.send(test_config_input)?;
//     and.on_update()?;

    let expected: Array2<u32> = array![[1, 2, 3, 1, 2, 3], [4, 5, 6, 1, 2, 3], [7, 8, 9, 1, 2, 3], [7, 8, 9, 1, 2, 3], [7, 8, 9, 1, 2, 3], [7, 8, 9, 1, 2, 3]];
    let actual: Array2<u32> = mock_output.next()?.records;
    
    Ok(assert!(expected == actual))
}

