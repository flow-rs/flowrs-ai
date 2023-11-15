use flowrs::{node::{Node, UpdateError, ChangeObserver}, connection::{Input, Output}};
use flowrs::RuntimeConnectable;

use ndarray::{Array2, array};
use linfa::dataset::DatasetBase;
use linfa_reduction::Pca;
use linfa::traits::{Fit, Predict};
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

            // transform to DatasetBase
            let dataset = DatasetBase::from(data);

            // parameter
            let embedding_size = 2;
            // pca
            let embedding = Pca::params(embedding_size)
                .fit(&dataset)
                .unwrap();
            let reduced_dataset = embedding.predict(dataset);
            
            // debug
            println!("Pca: {}", reduced_dataset.targets);

            self.output.send(reduced_dataset.targets).map_err(|e| UpdateError::Other(e.into()))?;
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

    let mut and: PCANode<> = PCANode::new(Some(&change_observer));
    let mock_output = flowrs::connection::Edge::new();
    flowrs::connection::connect(and.output.clone(), mock_output.clone());
    and.input.send(test_input)?;
    and.on_update()?;

    let expected: Array2<f64> = array![[-3.076047733203457, -10.562293260063301],
                                       [-3.561730416569943, 3.951032231750752],
                                       [14.63575200500477, 1.1072539713398344], 
                                       [-3.347031741680441, -4.147375003300382],
                                       [-4.622799446757189, 10.4931265494172],
                                       [-2.709147889142067, -11.467625779659173],
                                       [-3.984915594218815, 3.1728757730584096],
                                       [14.63575200500477, 1.1072539713398344],
                                       [-3.347031741680441, -4.147375003300382],
                                       [-4.622799446757189, 10.4931265494172]];
    let actual: Array2<f64> = mock_output.next()?;

    Ok(assert!(expected == actual))
}