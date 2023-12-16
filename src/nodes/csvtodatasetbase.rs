
use flowrs::{node::{Node, UpdateError, ChangeObserver}, connection::{Input, Output}};
use flowrs::RuntimeConnectable;

use linfa::DatasetBase;
use ndarray::{array, Array1};
use ndarray::Array2;
use csv::ReaderBuilder;
use ndarray_csv::Array2Reader;
use serde::{Deserialize, Serialize, de::DeserializeOwned};


#[derive(Clone, Deserialize, Serialize)]
pub struct CSVToDatasetBaseConfig {
   pub separator: u8,
   pub has_feature_names: bool
}


impl CSVToDatasetBaseConfig {
    pub fn new(separator: u8, has_feature_names: bool) -> Self {
        CSVToDatasetBaseConfig {
            separator: separator,
            has_feature_names: has_feature_names
        }
    }
}


#[derive(RuntimeConnectable, Deserialize, Serialize)]
pub struct CSVToDatasetBaseNode<T>
where
    T: Clone
{
    #[input]
    pub config_input: Input<CSVToDatasetBaseConfig>,

    #[input]
    pub data_input: Input<String>,

    #[output]
    pub output: Output<DatasetBase<Array2<T>, Array1<()>>>,

    config: CSVToDatasetBaseConfig
}


impl<T> CSVToDatasetBaseNode<T>
where
    T: Clone
{
    pub fn new(change_observer: Option<&ChangeObserver>) -> Self {
        Self {
            data_input: Input::new(),
            config_input: Input::new(),
            output: Output::new(change_observer),
            config: CSVToDatasetBaseConfig::new(b',', false)
        }
    }
}


impl<T> Node for CSVToDatasetBaseNode<T>
where
    T: Clone + Send + DeserializeOwned
{
    fn on_update(&mut self) -> Result<(), UpdateError> {
        println!("JW-Debug: CSVToDatasetBaseNode has received an update!");
     
        if let Ok(config) = self.config_input.next() {
            println!("JW-Debug: CSVToDatasetBaseNode has received config: {}, {}", config.separator, config.has_feature_names);
            self.config = config;
        }

        if let Ok(data) = self.data_input.next() {
            println!("JW-Debug: CSVToDatasetBaseNode has received data!");
            
            // convert String to DatasetBase
            let mut reader = ReaderBuilder::new()
                .delimiter(self.config.separator)
                .has_headers(self.config.has_feature_names)
                .from_reader(data.as_bytes());
            let data_ndarray: Array2<T> = reader.deserialize_array2_dynamic().map_err(|e| UpdateError::Other(e.into()))?;
            let dataset = DatasetBase::from(data_ndarray);
            
            // get feature names
            if self.config.has_feature_names {
                let mut feature_names : Vec<String> = Vec::new();
                for element in reader.headers().map_err(|e| UpdateError::Other(e.into()))?.into_iter() {
                    feature_names.push(String::from(element));
                };
                let dataset_with_features = dataset.with_feature_names(feature_names);

                self.output.send(dataset_with_features).map_err(|e| UpdateError::Other(e.into()))?;
                return Ok(());
            }
            self.output.send(dataset).map_err(|e| UpdateError::Other(e.into()))?;
        }
        Ok(())
    }
}


#[test]
fn input_output_test() -> Result<(), UpdateError> {
    let change_observer = ChangeObserver::new();
    let test_data_input = String::from("Feate1,Feature2,Feature3\n1,2,3\n4,5,6\n7,8,9");
    let test_config_input = CSVToDatasetBaseConfig{
        separator: b',',
        has_feature_names: true
    };

    let mut and: CSVToDatasetBaseNode<u32> = CSVToDatasetBaseNode::new(Some(&change_observer));
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
    let mut node: CSVToDatasetBaseNode<f32> = CSVToDatasetBaseNode::new(Some(&change_observer));
    let mock_output = flowrs::connection::Edge::new();
    flowrs::connection::connect(node.output.clone(), mock_output.clone());

    let test_data_input = String::from("1,2,3,4\n3,4,5,6\n5,6,7,8\n7,4,1,9");

    node.data_input.send(test_data_input.clone())?;
    node.on_update()?;

    let expected = array![[1.0, 2.0, 3.0, 4.0],
    [3.0, 4.0, 5.0, 6.0],
    [5.0, 6.0, 7.0, 8.0],
    [7.0, 4.0, 1.0, 9.0]];
    let actual = mock_output.next()?.records;

    Ok(assert!(expected == actual))
}


#[test]
fn test_f64() -> Result<(), UpdateError> {
    let change_observer = ChangeObserver::new();
    let mut node: CSVToDatasetBaseNode<f64> = CSVToDatasetBaseNode::new(Some(&change_observer));
    let mock_output = flowrs::connection::Edge::new();
    flowrs::connection::connect(node.output.clone(), mock_output.clone());

    let test_data_input = String::from("1,2,3,4\n3,4,5,6\n5,6,7,8\n7,4,1,9");

    node.data_input.send(test_data_input.clone())?;
    node.on_update()?;

    let expected = array![[1.0, 2.0, 3.0, 4.0],
    [3.0, 4.0, 5.0, 6.0],
    [5.0, 6.0, 7.0, 8.0],
    [7.0, 4.0, 1.0, 9.0]];
    let actual = mock_output.next()?.records;

    Ok(assert!(expected == actual))
}