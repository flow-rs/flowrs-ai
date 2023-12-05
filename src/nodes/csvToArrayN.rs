
use flowrs::{node::{Node, UpdateError, ChangeObserver}, connection::{Input, Output}};
use flowrs::RuntimeConnectable;

use linfa::DatasetBase;
use ndarray::{array, Array1, ArrayBase, OwnedRepr, Dim};
use ndarray::Array2;
use csv::ReaderBuilder;
use ndarray_csv::Array2Reader;
use serde::{Deserialize, Serialize, de::DeserializeOwned};


#[derive(Clone, Deserialize, Serialize)]
pub struct CSVToDatasetBaseConfig {
   pub separator: u8,
   pub has_feature_names: bool
}

#[derive(RuntimeConnectable, Deserialize, Serialize)]
pub struct CSVToDatasetBaseNode<T>
where
    T: Clone,
{
    #[input]
    pub config_input: Input<CSVToDatasetBaseConfig>,

    #[input]
    pub data_input: Input<String>,

    #[output]
    pub output: Output<DatasetBase<Array2<T>, Array1<()>>>,

    data_object: Option<String>
}

impl<T> CSVToDatasetBaseNode<T>
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

impl<T> Node for CSVToDatasetBaseNode<T>
where
    T: Clone + Send + DeserializeOwned,
{
    fn on_update(&mut self) -> Result<(), UpdateError> {
     
        if let Ok(data) = self.data_input.next() {
            println!("JW-Debug CSVToDatasetBaseNode has received data: {}.", data);
            self.data_object = Some(data);
        }

        if let Some(data) = &self.data_object {
            if let Ok(config) = self.config_input.next() {
                println!("JW-Debug CSVToArrayNNode has received config.");

                // convert String to DatasetBase
                let mut reader = ReaderBuilder::new()
                                                            .delimiter(config.separator)
                                                            .has_headers(config.has_feature_names)
                                                            .from_reader(data.as_bytes());
                let data_ndarray: Array2<T> = reader.deserialize_array2_dynamic().map_err(|e| UpdateError::Other(e.into()))?;
                let dataset = DatasetBase::from(data_ndarray);

                // get feature names
                if config.has_feature_names {
                    let mut feature_names : Vec<String> = Vec::new();
                    for element in reader.headers().map_err(|e| UpdateError::Other(e.into()))?.into_iter() {
                        feature_names.push(String::from(element));
                    };
                    let dataset_with_features = dataset.with_feature_names(feature_names);

                    self.output.send(dataset_with_features).map_err(|e| UpdateError::Other(e.into()))?;
                    return Ok(());
                }

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