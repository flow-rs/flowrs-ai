use flowrs::{node::{Node, UpdateError, ChangeObserver}, connection::{Input, Output}};
use flowrs::RuntimeConnectable;

use ndarray::{Array2, OwnedRepr};
use ndarray::prelude::*;
use linfa::{traits::Transformer, DatasetBase, Dataset};
use linfa_tsne::TSneParams;
use serde::{Deserialize, Serialize};

#[derive(Clone, Deserialize, Serialize)]
pub struct TsneConfig<T> 
where
    T: Clone + linfa::Float,
{
   pub embedding_size: usize,
   pub perplexity: T,
   pub approx_threshold: T,
}  

impl<T> TsneConfig<T>
where 
T: Clone + linfa::Float,
{
    pub fn new(embedding_size: usize, perplexity: T, approx_threshold: T) -> Self {
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
    T: Clone + linfa::Float,
{
    #[output]
    pub output: Output<DatasetBase<Array2<T>, Array1<()>>>,

    #[input]
    pub input: Input<DatasetBase<Array2<T>, Array1<()>>>,

    #[input]
    pub config_input: Input<TsneConfig<T>>,
    
    config: TsneConfig<T>
}

impl<T> TsneNode<T> 
where 
    T: Clone + linfa::Float,
{
    pub fn new(change_observer: Option<&ChangeObserver>) -> Self {
        let perplexity: T = T::from(1.0).unwrap();
        let approx_threshold: T = T::from(0.1).unwrap();

        Self {
            output: Output::new(change_observer),
            input: Input::new(),
            config_input: Input::new(),
            config: TsneConfig::new(2, perplexity, approx_threshold)
        }
    }
}

impl<T> Node for TsneNode<T>
where
    T: Clone + Send + linfa::Float,
{
    fn on_update(&mut self) -> Result<(), UpdateError> {
        println!("JW-Debug: TsneNode has received an update!");

        // Neue Config kommt an
        if let Ok(config) = self.config_input.next() {
            println!("JW-Debug: TsneNode has received config: {}, {}, {}", config.embedding_size, config.perplexity, config.approx_threshold);

            self.config = config;
        }

        // Daten kommen an
        if let Ok(data) = self.input.next() {
            println!("JW-Debug: TsneNode has received data!");

            let dataset = TSneParams::embedding_size(self.config.embedding_size)
            .perplexity(self.config.perplexity)
            .approx_threshold(self.config.approx_threshold)
            .transform(data.clone())
            .unwrap();

            self.output.send(dataset).map_err(|e| UpdateError::Other(e.into()))?;
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
    let dataset: DatasetBase<ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>, ArrayBase<OwnedRepr<()>, Dim<[usize; 1]>>> = Dataset::from(test_input.clone());
    let mut and: TsneNode<f64> = TsneNode::new(Some(&change_observer));
    let mock_output = flowrs::connection::Edge::new();
    flowrs::connection::connect(and.output.clone(), mock_output.clone());
    and.input.send(dataset)?;
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
    let dataset: DatasetBase<ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>, ArrayBase<OwnedRepr<()>, Dim<[usize; 1]>>> = Dataset::from(test_input.clone());
    let mut and: TsneNode<f64> = TsneNode::new(Some(&change_observer));
    let mock_output = flowrs::connection::Edge::new();
    flowrs::connection::connect(and.output.clone(), mock_output.clone());
    and.input.send(dataset)?;
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