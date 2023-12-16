use flowrs::{node::{Node, UpdateError, ChangeObserver}, connection::{Input, Output}};
use flowrs::RuntimeConnectable;

use ndarray::{Array2, Array1, array};
use linfa::{dataset::DatasetBase, Float};
use linfa::traits::Transformer;
use linfa_clustering::Dbscan;
use serde::{Deserialize, Serialize};


#[derive(Clone, Deserialize, Serialize)]
pub struct DbscanConfig<T>
where
    T: Clone + Float
{
    pub min_points: usize,
    pub tolerance: T
}


impl<T> DbscanConfig<T>
where
    T: Clone + Float
{
pub fn new(min_points: usize, tolerance: T) -> Self {
        DbscanConfig {
            min_points: min_points,
            tolerance: tolerance,
        }
    }
}


#[derive(RuntimeConnectable, Deserialize, Serialize)]
pub struct DbscanNode<T> 
where
    T: Clone + Float
{
    #[input]
    pub config_input: Input<DbscanConfig<T>>,
    
    #[output]
    pub output: Output<DatasetBase<Array2<T>, Array1<Option<usize>>>>,

    #[input]
    pub data_input: Input<DatasetBase<Array2<T>, Array1<()>>>, 

    config: DbscanConfig<T>
}


impl<T> DbscanNode<T> 
where
    T: Clone + Float
{
    pub fn new(change_observer: Option<&ChangeObserver>) -> Self {
        let default_tolerance: T = T::from(0.5).unwrap();

        Self {
            config_input: Input::new(),
            data_input: Input::new(),
            output: Output::new(change_observer),
            config: DbscanConfig::new(2, default_tolerance)
        }
    }
}


impl<T> Node for DbscanNode<T>
where
    T: Clone + Send + Float
{
    fn on_update(&mut self) -> Result<(), UpdateError> {
        println!("JW-Debug: DbscanNode has received an update!");

        // Neue Config kommt an
        if let Ok(config) = self.config_input.next() {
            println!("JW-Debug: DbscanNode has received config: {}, {}", config.min_points, config.tolerance);

            self.config = config;
        }

        // Daten kommen an
        if let Ok(data) = self.data_input.next() {
            println!("JW-Debug: DbscanNode has received data!"); //: \n Records: {} \n Targets: {}.", dataset.records, dataset.targets);

            let clusters = Dbscan::params(self.config.min_points)
                .tolerance(self.config.tolerance)
                .transform(data)
                .unwrap();

            self.output.send(clusters).map_err(|e| UpdateError::Other(e.into()))?;
            println!("JW-Debug: DbscanNode has sent an output!");
        }

        Ok(())
    }
}


#[test]
fn new_config_test() -> Result<(), UpdateError> {
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
    let input_data = DatasetBase::from(record_input);
    let test_config_input: DbscanConfig<f64> = DbscanConfig{
        min_points: 2,
        tolerance: 0.5
    };

    let mut and: DbscanNode<f64> = DbscanNode::new(Some(&change_observer));
    let mock_output = flowrs::connection::Edge::new();
    flowrs::connection::connect(and.output.clone(), mock_output.clone());
    and.data_input.send(input_data)?;
    and.config_input.send(test_config_input)?;
    and.on_update()?;

    let expected: Array1<Option<usize>> = array![None, None, Some(0), Some(1), Some(2), None, None, Some(0), Some(1), Some(2)];
    let actual: DatasetBase<Array2<f64>, Array1<Option<usize>>> = mock_output.next()?;

    Ok(assert!(expected == actual.targets))
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
    let input_data = DatasetBase::from(record_input);

    let mut and: DbscanNode<f64> = DbscanNode::new(Some(&change_observer));
    let mock_output = flowrs::connection::Edge::new();
    flowrs::connection::connect(and.output.clone(), mock_output.clone());
    and.data_input.send(input_data)?;
    and.on_update()?;

    let expected: Array1<Option<usize>> = array![None, None, Some(0), Some(1), Some(2), None, None, Some(0), Some(1), Some(2)];
    let actual: DatasetBase<Array2<f64>, Array1<Option<usize>>> = mock_output.next()?;

    Ok(assert!(expected == actual.targets))
}


#[test]
fn test_f32() -> Result<(), UpdateError> {
    let change_observer = ChangeObserver::new();
    let mut node: DbscanNode<f32> = DbscanNode::new(Some(&change_observer));
    let mock_output = flowrs::connection::Edge::new();
    flowrs::connection::connect(node.output.clone(), mock_output.clone());

    let test_data = array![[1.0, 2.0, 3.0, 4.0],
    [3.0, 4.0, 5.0, 6.0],
    [5.0, 6.0, 7.0, 8.0],
    [7.0, 4.0, 1.0, 9.0]];
    let test_data_input = DatasetBase::from(test_data.clone());

    node.data_input.send(test_data_input)?;
    node.on_update()?;

    let expected_targets = array![None, None, None, None];

    let actual = mock_output.next()?;
    let actual_records = actual.records;
    let actual_targets = actual.targets;

    Ok(assert!((test_data == actual_records) && (expected_targets == actual_targets)))
}


#[test]
fn test_f64() -> Result<(), UpdateError> {
    let change_observer = ChangeObserver::new();
    let mut node: DbscanNode<f64> = DbscanNode::new(Some(&change_observer));
    let mock_output = flowrs::connection::Edge::new();
    flowrs::connection::connect(node.output.clone(), mock_output.clone());

    let test_data = array![[1.0, 2.0, 3.0, 4.0],
    [3.0, 4.0, 5.0, 6.0],
    [5.0, 6.0, 7.0, 8.0],
    [7.0, 4.0, 1.0, 9.0]];
    let test_data_input = DatasetBase::from(test_data.clone());

    node.data_input.send(test_data_input)?;
    node.on_update()?;

    let expected_targets = array![None, None, None, None];

    let actual = mock_output.next()?;
    let actual_records = actual.records;
    let actual_targets = actual.targets;

    Ok(assert!((test_data == actual_records) && (expected_targets == actual_targets)))
}