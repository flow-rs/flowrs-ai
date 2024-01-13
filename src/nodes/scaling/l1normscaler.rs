use std::time::{Duration, Instant};

use flowrs::{node::{Node, UpdateError, ChangeObserver}, connection::{Input, Output}};
use flowrs::RuntimeConnectable;

use ndarray::{Array2, array, Array1};
use linfa::{traits::Transformer, DatasetBase, Float};
use linfa_preprocessing::norm_scaling::NormScaler;
use serde::{Deserialize, Serialize};


#[derive(RuntimeConnectable, Deserialize, Serialize)]
pub struct L1NormScalerNode<T>
where
    T: Clone
{
    #[output]
    pub output: Output<DatasetBase<Array2<T>, Array1<()>>>,

    #[input]
    pub data_input: Input<DatasetBase<Array2<T>, Array1<()>>>,

    cum_time: Duration,
    counter: usize
}


impl<T> L1NormScalerNode<T> 
where
    T: Clone
{
    pub fn new(change_observer: Option<&ChangeObserver>) -> Self {
        Self {
            output: Output::new(change_observer),
            data_input: Input::new(),
            cum_time: Duration::new(0, 0),
            counter: 0
        }
    }
}


impl<T> Node for L1NormScalerNode<T>
where
    T: Clone + Send + Float
{
    fn on_update(&mut self) -> Result<(), UpdateError> {

        // receiving data
        if let Ok(data) = self.data_input.next() {
            let start_time = Instant::now();

            let scaler = NormScaler::l1();
            let normalized_data = scaler.transform(data);
    
            self.output.send(normalized_data).map_err(|e| UpdateError::Other(e.into()))?;

            let end_time = Instant::now();
            self.cum_time = self.cum_time.saturating_add(end_time - start_time);
            self.counter = self.counter + 1;
            if self.counter == 10 {
                println!("[L1NormScalerNode] Cum_Time: {:?}", self.cum_time);
                #[cfg(target_arch = "wasm32")]
                crate::log(format!("[L1NormScalerNode] Cum_Time: {:?}", self.cum_time).as_str());
            }
        }

        Ok(())
    }
}


#[test]
fn input_output_test() -> Result<(), UpdateError> {
    let change_observer = ChangeObserver::new();
    let test_input: Array2<f64> = array![[1.0, 2.0, 3.0, 4.0],
    [3.0, 4.0, 5.0, 6.0],
    [5.0, 6.0, 7.0, 8.0],
    [7.0, 4.0, 1.0, 9.0]];

    let dataset = DatasetBase::from(test_input.clone());

    let mut and: L1NormScalerNode<f64> = L1NormScalerNode::new(Some(&change_observer));
    let mock_output = flowrs::connection::Edge::new();
    flowrs::connection::connect(and.output.clone(), mock_output.clone());
    and.data_input.send(dataset)?;
    and.on_update()?;

    let expected_data = array![[0.1, 0.2, 0.3, 0.4],
    [0.16666666666666666, 0.2222222222222222, 0.2777777777777778, 0.3333333333333333],
    [0.19230769230769232, 0.23076923076923078, 0.2692307692307692, 0.3076923076923077],
    [0.3333333333333333, 0.19047619047619047, 0.047619047619047616, 0.42857142857142855]];

    let actual = mock_output.next()?;
    let expected = DatasetBase::new(expected_data.clone(), expected_data.clone());

    Ok(assert!(expected.records == actual.records))
}


#[test]
fn test_f32() -> Result<(), UpdateError> {
    let change_observer = ChangeObserver::new();
    let mut node: L1NormScalerNode<f32> = L1NormScalerNode::new(Some(&change_observer));
    let mock_output = flowrs::connection::Edge::new();
    flowrs::connection::connect(node.output.clone(), mock_output.clone());

    let test_data = array![[1.0, 2.0, 3.0, 4.0],
    [3.0, 4.0, 5.0, 6.0],
    [5.0, 6.0, 7.0, 8.0],
    [7.0, 4.0, 1.0, 9.0]];
    let test_data_input = DatasetBase::from(test_data);

    node.data_input.send(test_data_input)?;
    node.on_update()?;

    let expected = array![[0.1, 0.2, 0.3, 0.4],
    [0.16666666666666666, 0.2222222222222222, 0.2777777777777778, 0.3333333333333333],
    [0.19230769230769232, 0.23076923076923078, 0.2692307692307692, 0.3076923076923077],
    [0.3333333333333333, 0.19047619047619047, 0.047619047619047616, 0.42857142857142855]];

    let actual = mock_output.next()?.records;

    Ok(assert!(expected == actual))
}


#[test]
fn test_f64() -> Result<(), UpdateError> {
    let change_observer = ChangeObserver::new();
    let mut node: L1NormScalerNode<f64> = L1NormScalerNode::new(Some(&change_observer));
    let mock_output = flowrs::connection::Edge::new();
    flowrs::connection::connect(node.output.clone(), mock_output.clone());

    let test_data = array![[1.0, 2.0, 3.0, 4.0],
    [3.0, 4.0, 5.0, 6.0],
    [5.0, 6.0, 7.0, 8.0],
    [7.0, 4.0, 1.0, 9.0]];
    let test_data_input = DatasetBase::from(test_data);

    node.data_input.send(test_data_input)?;
    node.on_update()?;

    let expected = array![[0.1, 0.2, 0.3, 0.4],
    [0.16666666666666666, 0.2222222222222222, 0.2777777777777778, 0.3333333333333333],
    [0.19230769230769232, 0.23076923076923078, 0.2692307692307692, 0.3076923076923077],
    [0.3333333333333333, 0.19047619047619047, 0.047619047619047616, 0.42857142857142855]];

    let actual = mock_output.next()?.records;

    Ok(assert!(expected == actual))
}