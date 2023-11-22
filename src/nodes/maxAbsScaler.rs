use flowrs::{node::{Node, UpdateError, ChangeObserver}, connection::{Input, Output}};
use flowrs::RuntimeConnectable;

use ndarray::{Array2, array};
use linfa::dataset::DatasetBase;
use linfa::traits::{Fit, Transformer};
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

            // transform to DatasetBase
            let dataset = DatasetBase::from(data);

            // max abs scaling
            let scaler = LinearScaler::max_abs().fit(&dataset).unwrap();
            let dataset = scaler.transform(dataset);

            // debug
            println!("Scaled data: {}", dataset.records);

            self.output.send(dataset.records).map_err(|e| UpdateError::Other(e.into()))?;
        }
        Ok(())
    }
}

#[test]
fn input_output_test() -> Result<(), UpdateError> {
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

    let mut and: MaxAbsSclerNode<> = MaxAbsSclerNode::new(Some(&change_observer));
    let mock_output = flowrs::connection::Edge::new();
    flowrs::connection::connect(and.output.clone(), mock_output.clone());
    and.input.send(test_input)?;
    and.on_update()?;

    let expected: Array2<f64> = array![[0.08461538461538462, 0.17857142857142855, 0.21333333333333335, 0.35384615384615387, 0.37142857142857144, 0.44666666666666666],
                                       [0.6, 0.5857142857142856, 0.6333333333333333, 0.7923076923076924, 0.7857142857142857, 0.8],
                                       [1., 1., 1., 0.07692307692307693, 0.14285714285714285, 0.2],
                                       [0.3076923076923077, 0.3571428571428571, 0.4, 0.5384615384615385, 0.5714285714285714, 0.6],
                                       [0.7692307692307693, 0.7857142857142857, 0.8, 1., 1., 1.],
                                       [0.07692307692307693, 0.14285714285714285, 0.2, 0.3076923076923077, 0.3571428571428571, 0.4],
                                       [0.5384615384615385, 0.5714285714285714, 0.6, 0.7692307692307693, 0.7857142857142857, 0.8],
                                       [1., 1., 1., 0.07692307692307693, 0.14285714285714285, 0.2],
                                       [0.3076923076923077, 0.3571428571428571, 0.4, 0.5384615384615385, 0.5714285714285714, 0.6],
                                       [0.7692307692307693, 0.7857142857142857, 0.8, 1., 1., 1.]];
    let actual: Array2<f64> = mock_output.next()?;

    Ok(assert!(expected == actual))
}