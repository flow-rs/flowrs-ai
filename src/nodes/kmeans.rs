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

// Definition eines Structs
#[derive(RuntimeConnectable)]
pub struct KmeansNode<T>
where
    T: Clone,
{
    #[input]
    pub config_input: Input<KmeansConfig>,

    #[output]
    pub output: Output<DatasetBase<Array2<T>, Array1<usize>>>,

    #[input]
    pub input: Input<DatasetBase<Array2<T>, Array1<()>>>, 
}

impl<T> KmeansNode<T> 
where
    T: Clone,
{
    // Hier will der Konstruktur als einzigen Parameter einen ChangeObserver
    pub fn new(change_observer: Option<&ChangeObserver>) -> Self {
        Self {
            output: Output::new(change_observer),
            input: Input::new(),
            config_input: Input::new()
        }
    }
}

// Hier befinden sich die Methoden von unserer Node.
impl Node for KmeansNode<f64> {
    // on_update wird von der Pipeline automatisch getriggert, wenn diese Node einen Input bekommt.
    fn on_update(&mut self) -> Result<(), UpdateError> {

        // Hier überprüfen wir nur, ob ein input da ist und der passt
        if let Ok(node_data) = self.input.next() {
            println!("JW-Debug: KmeansNode has received: \n Records: {}.", node_data.records);
            
            if let Ok(config) = self.config_input.next() {
                println!("JW-Debug CSVToArrayNNode has received config.");

                let records = node_data.records.clone();

                let model = KMeans::params(config.num_of_dim)
                .max_n_iterations(config.max_n_iterations)
                .tolerance(config.tolerance)
                .fit(&node_data)
                .expect("Error while fitting KMeans to the dataset");
    
                // Predict cluster assignments
                let result = model.predict(node_data);
                println!("Result: {:?}\n", result);
    
                let myoutput: DatasetBase<Array2<f64>, Array1<usize>> = DatasetBase::new(records, result.targets.clone());
    
                // Hier schicken wir node_data als output an die nächste node bzw. den output
                self.output.send(myoutput).map_err(|e| UpdateError::Other(e.into()))?;

            } else {
                //Err(UpdateError::Other(anyhow::Error::msg("No config received!")));
            }         
        }
        Ok(())
    }
}


// #############################################################################
// #############################################################################
// Test, um die Node zu testen
// Hier auf "|> Run Test" drücken, was unter "#[test" angezeigt wird
// #############################################################################
// #############################################################################
#[test]
fn input_output_test() -> Result<(), UpdateError> {
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
    let target_input: Array2<f64> = array![[-3.076047733203457, -10.562293260063301],
    [-3.561730416569943, 3.951032231750752],
    [14.63575200500477, 1.1072539713398344], 
    [-3.347031741680441, -4.147375003300382],
    [-4.622799446757189, 10.4931265494172],
    [-2.709147889142067, -11.467625779659173],
    [-3.984915594218815, 3.1728757730584096],
    [14.63575200500477, 1.1072539713398344],
    [-3.347031741680441, -4.147375003300382],
    [-4.622799446757189, 10.4931265494172]];
    let input_data = DatasetBase::from(record_input.clone());

    let mut and: KmeansNode<f64> = KmeansNode::new(Some(&change_observer));
    let mock_output = flowrs::connection::Edge::new();
    flowrs::connection::connect(and.output.clone(), mock_output.clone());
    and.input.send(input_data)?;
    and.config_input.send(test_config_input)?;
    and.on_update()?;

    let expected = array![2, 0, 1, 2, 0, 2, 0, 1, 2, 0];
    let actual = mock_output.next()?;

    Ok(assert!(expected == actual.targets()))
}