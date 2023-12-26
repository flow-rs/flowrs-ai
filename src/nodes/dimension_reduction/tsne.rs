use flowrs::{node::{Node, UpdateError, ChangeObserver}, connection::{Input, Output}};
use flowrs::RuntimeConnectable;

use ndarray::{Array2, Array1, array};
use linfa::{traits::Transformer, DatasetBase, Float};
use linfa_tsne::TSneParams;
use serde::{Deserialize, Serialize};


#[derive(Clone, Deserialize, Serialize)]
pub struct TsneConfig {
   pub embedding_size: usize,
   pub perplexity: f64,
   pub approx_threshold: f64,
}  


impl TsneConfig {
    pub fn new(embedding_size: usize, perplexity: f64, approx_threshold: f64) -> Self {
        TsneConfig {
            embedding_size,
            perplexity,
            approx_threshold
        }
    }
}


#[derive(RuntimeConnectable, Deserialize, Serialize)]
pub struct TsneNode<T> 
where
    T: Clone + Float
{
    #[output]
    pub output: Output<DatasetBase<Array2<T>, Array1<()>>>,

    #[input]
    pub data_input: Input<DatasetBase<Array2<T>, Array1<()>>>,

    #[input]
    pub config_input: Input<TsneConfig>,
    
    config: TsneConfig
}


impl<T> TsneNode<T> 
where 
    T: Clone + Float
{
    pub fn new(change_observer: Option<&ChangeObserver>) -> Self {
        Self {
            output: Output::new(change_observer),
            data_input: Input::new(),
            config_input: Input::new(),
            config: TsneConfig::new(2, 1., 0.1)
        }
    }
}


impl<T> Node for TsneNode<T>
where
    T: Clone + Send + Float
{
    fn on_update(&mut self) -> Result<(), UpdateError> {

        // received config
        if let Ok(config) = self.config_input.next() {
            println!("[DEBUG::TsneNode] New Config:\n embedding_size: {},\n steps: {},\n approx_threshold: {}", config.embedding_size, config.perplexity, config.approx_threshold);
            self.config = config;
        }

        // received data
        if let Ok(data) = self.data_input.next() {
            println!("[DEBUG::TsneNode] Received Data:\n {}", data.records.clone());

            let red_dataset = TSneParams::embedding_size(self.config.embedding_size)
                .perplexity(T::from(self.config.perplexity).unwrap())
                .approx_threshold(T::from(self.config.approx_threshold).unwrap())
                .transform(data.clone())
                .unwrap();

            println!("[DEBUG::TsneNode] Sent Data:\n {}", red_dataset.records.clone());
            self.output.send(red_dataset).map_err(|e| UpdateError::Other(e.into()))?;
        }
        Ok(())
    }
}


#[test]
fn input_output_test() -> Result<(), UpdateError> {
    let change_observer = ChangeObserver::new();
    let test_config_input = TsneConfig{
        embedding_size: 2,
        perplexity: 1.0,
        approx_threshold: 0.1
    };
    let test_input: Array2<f64> = array![[1.1, 2.5, 3.2, 4.6, 5.2, 6.7], 
                                         [7.8, 8.2, 9.5, 10.3, 11.0, 12.0], 
                                         [13.0, 14.0, 15.0, 1.0, 2.0, 3.0], 
                                         [4.0, 5.0, 6.0, 7.0, 8.0, 9.0], 
                                         [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
                                         [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 
                                         [7.0, 8.0, 9.0, 10.0, 11.0, 12.0], 
                                         [13.0, 14.0, 15.0, 1.0, 2.0, 3.0], 
                                         [4.0, 5.0, 6.0, 7.0, 8.0, 9.0], 
                                         [10.0, 11.0, 12.0, 13.0, 14.0, 15.0]];
    let dataset = DatasetBase::from(test_input.clone());
    let mut and: TsneNode<f64> = TsneNode::new(Some(&change_observer));
    let mock_output = flowrs::connection::Edge::new();
    flowrs::connection::connect(and.output.clone(), mock_output.clone());
    and.data_input.send(dataset)?;
    and.config_input.send(test_config_input)?;
    and.on_update()?;

    let actual = mock_output.next()?.records;
    
    let expected_rows = 10;
    let expected_cols = 2;

    if actual.shape()[0] == expected_rows && actual.shape()[1] == expected_cols {
        Ok(())
    } else {
        Err(UpdateError::RecvError { message: "Actual has wrong size".to_string() })
    }
}


#[test]
fn default_config_test() -> Result<(), UpdateError> {
    let change_observer = ChangeObserver::new();

    let test_input: Array2<f64> = array![[1.1, 2.5, 3.2, 4.6, 5.2, 6.7], 
                                         [7.8, 8.2, 9.5, 10.3, 11.0, 12.0], 
                                         [13.0, 14.0, 15.0, 1.0, 2.0, 3.0], 
                                         [4.0, 5.0, 6.0, 7.0, 8.0, 9.0], 
                                         [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
                                         [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 
                                         [7.0, 8.0, 9.0, 10.0, 11.0, 12.0], 
                                         [13.0, 14.0, 15.0, 1.0, 2.0, 3.0], 
                                         [4.0, 5.0, 6.0, 7.0, 8.0, 9.0], 
                                         [10.0, 11.0, 12.0, 13.0, 14.0, 15.0]];
    let dataset = DatasetBase::from(test_input.clone());
    let mut and: TsneNode<f64> = TsneNode::new(Some(&change_observer));
    let mock_output = flowrs::connection::Edge::new();
    flowrs::connection::connect(and.output.clone(), mock_output.clone());
    and.data_input.send(dataset)?;
    and.on_update()?;

    let actual = mock_output.next()?.records;

    let expected_rows = 10;
    let expected_cols = 2;

    if actual.shape()[0] == expected_rows && actual.shape()[1] == expected_cols {
        Ok(())
    } else {
        Err(UpdateError::RecvError { message: "Actual has wrong size".to_string() })
    }
}


#[test]
fn test_f32() -> Result<(), UpdateError> {
    let change_observer = ChangeObserver::new();
    let mut node: TsneNode<f32> = TsneNode::new(Some(&change_observer));
    let mock_output = flowrs::connection::Edge::new();
    flowrs::connection::connect(node.output.clone(), mock_output.clone());

    let test_data = array![[1.0, 2.0, 3.0, 4.0],
    [3.0, 4.0, 5.0, 6.0],
    [5.0, 6.0, 7.0, 8.0],
    [7.0, 4.0, 1.0, 9.0]];
    let test_data_input = DatasetBase::from(test_data);

    node.data_input.send(test_data_input)?;
    node.on_update()?;

    let expected = array![[170.71805, -202.21175],
    [30.475708, 4.392071],
    [105.48429, 242.53217],
    [-306.67807, -44.712498]];

    let actual = mock_output.next()?.records;
    
    Ok(assert!(expected == actual))
}


#[test]
fn test_f64() -> Result<(), UpdateError> {
    let change_observer: ChangeObserver = ChangeObserver::new();
    let mut node: TsneNode<f64> = TsneNode::new(Some(&change_observer));
    let mock_output = flowrs::connection::Edge::new();
    flowrs::connection::connect(node.output.clone(), mock_output.clone());

    let test_data = array![[1.0, 2.0, 3.0, 4.0],
    [3.0, 4.0, 5.0, 6.0],
    [5.0, 6.0, 7.0, 8.0],
    [7.0, 4.0, 1.0, 9.0]];
    let test_data_input = DatasetBase::from(test_data);

    node.data_input.send(test_data_input)?;
    node.on_update()?;

    let expected = array![[22.535654114711726, 257.3893360361801],
    [26.93970505811448, 13.644883900171767],
    [220.8346844675358, -134.12344265050243],
    [-270.310043640362, -136.91077728584943]];

    let actual = mock_output.next()?.records;

    Ok(assert!(expected == actual))
}