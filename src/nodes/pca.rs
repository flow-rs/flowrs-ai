use std::arch::x86_64::_MM_EXCEPT_DENORM;

use flowrs::{node::{Node, UpdateError, ChangeObserver}, connection::{Input, Output}};
use flowrs::RuntimeConnectable;

use ndarray::{Array2, array, ArrayBase, Dim, OwnedRepr};
use linfa::{Dataset, DatasetBase};
use linfa_reduction::Pca;
use linfa::traits::{Fit, Predict};
use serde::{Deserialize, Serialize};


#[derive(Clone)]
pub struct PCAConfig {
   pub embedding_size: usize
}

#[derive(RuntimeConnectable)]
pub struct PCANode {
    #[input]
    pub config_input: Input<PCAConfig>,

    #[output]
    pub output: Output<DatasetBase<ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>, ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>>>,

    #[input]
    pub dataset_input: Input<DatasetBase<ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>, ArrayBase<OwnedRepr<()>, Dim<[usize; 1]>>>>
}

impl PCANode {
    pub fn new(change_observer: Option<&ChangeObserver>) -> Self {
        Self {
            config_input: Input::new(),
            dataset_input: Input::new(),
            output: Output::new(change_observer)
        }
    }
}

impl Node for PCANode {
    fn on_update(&mut self) -> Result<(), UpdateError> {

        if let Ok(dataset) = self.dataset_input.next() {
            println!("JW-Debug PCANode has received: {}.", dataset.records);
            
            if let Ok(config) = self.config_input.next() {
                println!("JW-Debug PCANode has received config.");
            
                // parameter
                let embedding_size = 2;
                // pca
                let embedding = Pca::params(embedding_size)
                    .fit(&dataset)
                    .unwrap();
                let red_dataset = embedding.predict(dataset);
                
                let myoutput: DatasetBase<ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>, ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>> = DatasetBase::new(red_dataset.records.clone(), red_dataset.targets.clone());

                println!("DatasetBase\n");
                println!("Records:\n {}\n", red_dataset.records.clone());
                println!("Targets:\n {:?}\n", red_dataset.targets.clone());
                println!("Feature names:\n {:?}\n", red_dataset.feature_names().clone());

                self.output.send(myoutput).map_err(|e| UpdateError::Other(e.into()))?;
            }
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
    let dataset = Dataset::from(test_input.clone());
    let test_config_input = PCAConfig{
        embedding_size: 2,
    };
    let mut and: PCANode<> = PCANode::new(Some(&change_observer));
    let mock_output = flowrs::connection::Edge::new();
    flowrs::connection::connect(and.output.clone(), mock_output.clone());
    and.dataset_input.send(dataset)?;
    and.config_input.send(test_config_input);
    and.on_update()?;

    let expected_data: Array2<f64> = array![[-3.076047733203457, -10.562293260063301],
                                       [-3.561730416569943, 3.951032231750752],
                                       [14.63575200500477, 1.1072539713398344], 
                                       [-3.347031741680441, -4.147375003300382],
                                       [-4.622799446757189, 10.4931265494172],
                                       [-2.709147889142067, -11.467625779659173],
                                       [-3.984915594218815, 3.1728757730584096],
                                       [14.63575200500477, 1.1072539713398344],
                                       [-3.347031741680441, -4.147375003300382],
                                       [-4.622799446757189, 10.4931265494172]];
    let expected: DatasetBase<ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>, ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>> = DatasetBase::new(test_input.clone(), expected_data.clone());

    let actual = mock_output.next()?;


    println!("Actual\n");
    println!("Records:\n {}\n", actual.records.clone());
    println!("Targets:\n {:?}\n", actual.targets.clone());
    println!("Feature names:\n {:?}\n", actual.feature_names().clone());

    println!("Expected\n");
    println!("Records:\n {}\n", expected.records.clone());
    println!("Targets:\n {:?}\n", expected.targets.clone());
    println!("Feature names:\n {:?}\n", expected.feature_names().clone());


    Ok(assert!(expected.targets == actual.targets))
}