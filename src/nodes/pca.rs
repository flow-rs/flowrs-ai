use flowrs::{node::{Node, UpdateError, ChangeObserver}, connection::{Input, Output}};
use flowrs::RuntimeConnectable;

use ndarray::Array2;
use linfa::dataset::DatasetBase;
use linfa_reduction::Pca;
use linfa::traits::{Fit, Predict};
use ndarray::arr2;
use serde::{Deserialize, Serialize};


#[derive(RuntimeConnectable, Deserialize, Serialize)]
pub struct PCANode {
    #[output]
    pub output: Output<Array2<f64>>,

    #[input]
    pub input: Input<Array2<f64>>,
}

impl PCANode {
    pub fn new(change_observer: Option<&ChangeObserver>) -> Self {
        Self {
            output: Output::new(change_observer),
            input: Input::new()
        }
    }
}

impl Node for PCANode {
    fn on_update(&mut self) -> Result<(), UpdateError> {

        if let Ok(data) = self.input.next() {
            println!("JW-Debug PCANode has received: {}.", data);

            let dataset = DatasetBase::from(data);
            // apply PCA projection along a line which maximizes the spread of the data
            let embedding = Pca::params(1)
                .fit(&dataset)
                .unwrap();

            // reduce dimensionality of the dataset
            let reduced_dataset = embedding.predict(dataset);
            println!("{}", reduced_dataset.targets);

            self.output.send(reduced_dataset.targets).map_err(|e| UpdateError::Other(e.into()))?;
        }
        Ok(())
    }
}

#[test]
fn input_output_test() -> Result<(), UpdateError> {
    let change_observer = ChangeObserver::new();
    let test_input: Array2<f64> = arr2(&[[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]);

    let mut and: PCANode<> = PCANode::new(Some(&change_observer));
    let mock_output = flowrs::connection::Edge::new();
    flowrs::connection::connect(and.output.clone(), mock_output.clone());
    and.input.send(test_input)?;
    and.on_update()?;

    let expected: Array2<f64> = arr2(&[[-5.19615242270663],[0.],[5.19615242270663]]);
    let actual: Array2<f64> = mock_output.next()?;

    Ok(assert!(expected == actual))
}