use flowrs::{node::{Node, UpdateError, ChangeObserver}, connection::{Input, Output}};
use flowrs::RuntimeConnectable;

use linfa::prelude::*;
use ndarray::{prelude::*, OwnedRepr};
use linfa::traits::{Fit, Predict};
use linfa_clustering::KMeans;
use serde::{Deserialize, Serialize};


#[derive(Clone)]
pub struct KmeansConfig {
   pub num_of_dim: usize,
   pub max_n_iterations: u64,
   pub tolerance: f64
}

impl KmeansConfig {
    pub fn new(num_of_dim: usize, max_n_iterations: u64, tolerance: f64) -> Self {
        KmeansConfig {
            num_of_dim,
            max_n_iterations,
            tolerance,
        }
    }
}

#[derive(RuntimeConnectable)]
pub struct KmeansNode {
    #[input]
    pub config_input: Input<KmeansConfig>,

    #[output]
    pub output: Output<DatasetBase<Array2<f64>, Array1<usize>>>,

    #[input]
    pub input: Input<DatasetBase<Array2<f64>, Array1<()>>>, 

    config: KmeansConfig
}

impl KmeansNode {
    pub fn new(change_observer: Option<&ChangeObserver>) -> Self {
        Self {
            output: Output::new(change_observer),
            input: Input::new(),
            config_input: Input::new(),
            config: KmeansConfig::new(3, 200, 1e-5)
        }
    }
}

impl Node for KmeansNode {
    fn on_update(&mut self) -> Result<(), UpdateError> {
        println!("JW-Debug: KmeansNode has received an update!");

        // Neue Config kommt an
        if let Ok(config) = self.config_input.next() {
            println!("JW-Debug: KmeansNode has received config: {}, {}, {}", config.max_n_iterations, config.num_of_dim, config.tolerance);

            self.config = config;
        }

        // Daten kommen an
        if let Ok(data) = self.input.next() {
            println!("JW-Debug: KmeansNode has received data!");

            let records = data.records.clone();

            let model = KMeans::params(self.config.num_of_dim)
            .max_n_iterations(self.config.max_n_iterations)
            .tolerance(self.config.tolerance)
            .fit(&data)
            .expect("Error while fitting KMeans to the dataset");

            let result = model.predict(data);

            let myoutput: DatasetBase<Array2<f64>, Array1<usize>> = DatasetBase::new(records, result.targets.clone());

            self.output.send(myoutput).map_err(|e| UpdateError::Other(e.into()))?;
            println!("JW-Debug: KmeansNode has sent an output!");
        }        
        Ok(())
    }
}


#[test]
fn new_config_test() -> Result<(), UpdateError> {
    let change_observer = ChangeObserver::new();
    let test_config_input = KmeansConfig{
        num_of_dim: 3,
        max_n_iterations: 200,
        tolerance: 1e-5
    };
    
    let record_input: Array2<f64> = array![[1.1, 2.5, 3.2, 4.6, 5.2, 6.7], 
                                         [7.8, 8.2, 9.5, 10.3, 11.0, 12.0], 
                                         [13.0, 14.0, 15.0, 1.0, 2.0, 3.0], 
                                         [4.0, 5.0, 6.0, 7.0, 8.0, 9.0], 
                                         [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
                                         [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 
                                         [7.0, 8.0, 9.0, 10.0, 11.0, 12.0], 
                                         [13.0, 14.0, 15.0, 1.0, 2.0, 3.0], 
                                         [4.0, 5.0, 6.0, 7.0, 8.0, 9.0], 
                                         [10.0, 11.0, 12.0, 13.0, 14.0, 15.0]];

    let input_data = DatasetBase::from(record_input.clone());

    let mut and: KmeansNode<> = KmeansNode::new(Some(&change_observer));
    let mock_output = flowrs::connection::Edge::new();
    flowrs::connection::connect(and.output.clone(), mock_output.clone());
    and.input.send(input_data)?;
    and.config_input.send(test_config_input)?;
    and.on_update()?;

    let expected = array![2, 0, 1, 2, 0, 2, 0, 1, 2, 0];
    let actual = mock_output.next()?;

    Ok(assert!(expected == actual.targets()))
}

#[test]
fn default_config_test() -> Result<(), UpdateError> {
    let change_observer = ChangeObserver::new();
    
    let record_input: Array2<f64> = array![[1.1, 2.5, 3.2, 4.6, 5.2, 6.7], 
                                         [7.8, 8.2, 9.5, 10.3, 11.0, 12.0], 
                                         [13.0, 14.0, 15.0, 1.0, 2.0, 3.0], 
                                         [4.0, 5.0, 6.0, 7.0, 8.0, 9.0], 
                                         [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
                                         [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 
                                         [7.0, 8.0, 9.0, 10.0, 11.0, 12.0], 
                                         [13.0, 14.0, 15.0, 1.0, 2.0, 3.0], 
                                         [4.0, 5.0, 6.0, 7.0, 8.0, 9.0], 
                                         [10.0, 11.0, 12.0, 13.0, 14.0, 15.0]];

    let input_data = DatasetBase::from(record_input.clone());

    let mut and: KmeansNode<> = KmeansNode::new(Some(&change_observer));
    let mock_output = flowrs::connection::Edge::new();
    flowrs::connection::connect(and.output.clone(), mock_output.clone());
    and.input.send(input_data)?;
    and.on_update()?;

    let expected = array![2, 0, 1, 2, 0, 2, 0, 1, 2, 0];
    let actual = mock_output.next()?;

    Ok(assert!(expected == actual.targets()))
}