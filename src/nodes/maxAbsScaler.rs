use flowrs::{node::{Node, UpdateError, ChangeObserver}, connection::{Input, Output}};
use flowrs::RuntimeConnectable;

use ndarray::Array2;
use linfa::dataset::DatasetBase;
use linfa::traits::{Fit, Transformer};
use ndarray::arr2;
use linfa_preprocessing::linear_scaling::LinearScaler;
use serde::{Deserialize, Serialize};


#[derive(RuntimeConnectable, Deserialize, Serialize)]
pub struct MaxAbsSclerNode {
    #[output]
    pub output: Output<Array2<f64>>,

    #[input]
    pub input: Input<Array2<f64>>,
}

impl MaxAbsSclerNode {
    pub fn new(change_observer: Option<&ChangeObserver>) -> Self {
        Self {
            output: Output::new(change_observer),
            input: Input::new()
        }
    }
}

impl Node for MaxAbsSclerNode {
    fn on_update(&mut self) -> Result<(), UpdateError> {

        if let Ok(data) = self.input.next() {
            println!("JW-Debug MaxAbsScalerNode has received: {}.", data);

            let dataset = DatasetBase::from(data);
            // apply PCA projection along a line which maximizes the spread of the data
            // Learn scaling parameters
            let scaler = LinearScaler::max_abs().fit(&dataset).unwrap();
            // scale dataset according to parameters
            let dataset = scaler.transform(dataset);

            println!("{}", dataset.records);

            self.output.send(dataset.records).map_err(|e| UpdateError::Other(e.into()))?;
        }
        Ok(())
    }
}

#[test]
fn input_output_test() -> Result<(), UpdateError> {
    let change_observer = ChangeObserver::new();
    let test_input: Array2<f64> = arr2(&[[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]);

    let mut and: MaxAbsSclerNode<> = MaxAbsSclerNode::new(Some(&change_observer));
    let mock_output = flowrs::connection::Edge::new();
    flowrs::connection::connect(and.output.clone(), mock_output.clone());
    and.input.send(test_input)?;
    and.on_update()?;

    let expected: Array2<f64> = arr2(&[[0.14285714285714285, 0.25, 0.3333333333333333],[0.5714285714285714, 0.625, 0.6666666666666666],[1., 1., 1.]]);
    let actual: Array2<f64> = mock_output.next()?;

    Ok(assert!(expected == actual))
}