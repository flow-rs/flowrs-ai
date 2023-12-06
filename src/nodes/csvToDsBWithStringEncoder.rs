
use flowrs::{node::{Node, UpdateError, ChangeObserver}, connection::{Input, Output}};
use flowrs::RuntimeConnectable;

use linfa::DatasetBase;
use ndarray::{array, ArrayBase, OwnedRepr, Dim, Axis};
use ndarray::Array2;
use csv::ReaderBuilder;
use ndarray_csv::Array2Reader;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::collections::HashSet;


#[derive(Clone, Deserialize, Serialize)]
pub struct CSVToEncodedDatasetBaseConfig {
   pub separator: u8,
   pub has_feature_names: bool,
   nominals: Vec<String>,
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
    pub output: Output<DatasetBase<Array2<T>, ()>>,

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
                println!("JW-Debug CSVToArrayNNode has received config.");

                let mut reader = ReaderBuilder::new()
                .delimiter(config.separator)
                .has_headers(config.has_feature_names)
                .from_reader(data.as_bytes());
        
                let headers: Vec<String> = reader.headers()
                                                    .map_err(|e| UpdateError::Other(e.into()))?
                                                    .into_iter()
                                                    .map(|header| header.to_string()) // Convert &str to String
                                                    .collect();
            
                let nominal_indices: HashSet<usize> = config.nominals.iter().filter_map(|nominal| headers.iter().position(|header| header == nominal)).collect();
            
                let data_ndarray: Array2<i32> = reader.deserialize_array2_dynamic().map_err(|e| UpdateError::Other(e.into()))?;
            
                let mut nominal_data = Vec::new();
                let mut other_data = Vec::new();
            
                for (col_idx, col) in data_ndarray.axis_iter(Axis(1)).enumerate() {
                    if nominal_indices.contains(&col_idx) {
                        nominal_data.push(col.to_owned());
                    } else {
                        other_data.push(col.to_owned());
                    }
                }
            
                let nominal_records: Vec<Vec<f64>> = nominal_data.iter().map(|col| col.iter().map(|&x| x as f64).collect()).collect();
                let other_records: Vec<Vec<f64>> = other_data.iter().map(|col| col.iter().map(|&x| x as f64).collect()).collect();
                
                let nominal_dataset: DatasetBase<f64, f64> = DatasetBase::from(Records::new(nominal_records, None));
                let other_dataset: DatasetBase<f64, f64> = DatasetBase::from(Records::new(other_records, None));

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
    let test_data_input = String::from("Feate1,Feature2,Feature3\n1,2,3\n4,5,6\n7,8,9");
    let test_config_input = CSVToEncodedDatasetBaseConfig{
        separator: b',',
        has_feature_names: true
    };

    let mut and: CSVToEncodedDatasetBaseNode<u32> = CSVToEncodedDatasetBaseNode::new(Some(&change_observer));
    let mock_output = flowrs::connection::Edge::new();
    flowrs::connection::connect(and.output.clone(), mock_output.clone());
    and.data_input.send(test_data_input)?;
    and.config_input.send(test_config_input)?;
    and.on_update()?;

    let expected: Array2<u32> = array![[1, 2, 3], [4, 5, 6], [7, 8, 9]];
    let actual: Array2<u32> = mock_output.next()?.records;
    
    Ok(assert!(expected == actual))
}