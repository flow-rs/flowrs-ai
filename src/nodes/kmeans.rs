use flowrs::{node::{Node, UpdateError, ChangeObserver}, connection::{Input, Output}};
use flowrs::RuntimeConnectable;

use linfa::prelude::*;
use ndarray::{prelude::*, OwnedRepr};
use linfa::traits::{Fit, Predict};
use linfa_clustering::KMeans;
use serde::{Deserialize, Serialize};


// Definition eines Structs
#[derive(RuntimeConnectable, Deserialize, Serialize)]
pub struct KmeansNode { // <--- Wenn man eine neue Node anlegt, einfach alles kopieren und hier den Namen ändern
    #[output]
    pub output: Output<DatasetBase<ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>, ArrayBase<OwnedRepr<usize>, Dim<[usize; 1]>>>>, // <--- Wir haben in diesem Fall eine Output-Variable vom Typ Array2<u8>

    #[input]
    pub input: Input<DatasetBase<ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>, ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>>>, // <--- Wir haben in diesem Fall eine Input-Variable vom Typ Array2<u8>

    // Das bedeutet, unsere Node braucht als Input einen Array2<u8> und liefert als Output einen Array2<u8>
}

// Das ist einfach der Konstruktur
impl KmeansNode {
    // Hier will der Konstruktur als einzigen Parameter einen ChangeObserver
    pub fn new(change_observer: Option<&ChangeObserver>) -> Self {
        Self {
            output: Output::new(change_observer),
            input: Input::new()
        }
    }
}

// Hier befinden sich die Methoden von unserer Node. Wir verwenden erstmal nur die Methoden, welche wir implementieren müssen, da diese von "Node" vorgegeben werden.
impl Node for KmeansNode {
    // on_update wird von der Pipeline automatisch getriggert, wenn diese Node einen Input bekommt.
    fn on_update(&mut self) -> Result<(), UpdateError> {

        // Hier überprüfen wir nur, ob ein input da ist und der passt
        if let Ok(node_data) = self.input.next() {
            println!("JW-Debug: KmeansNode has received: \n Records: {} \n Targets: {}.", node_data.records, node_data.targets);

            let model = KMeans::params(3)
            .max_n_iterations(200)
            .tolerance(1e-5)
            .fit(&node_data)
            .expect("Error while fitting KMeans to the dataset");

            // Predict cluster assignments
            let result = model.predict(node_data);
            println!("Result: {:?}\n", result);

            let myoutput: DatasetBase<ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>, ArrayBase<OwnedRepr<usize>, Dim<[usize; 1]>>> = DatasetBase::new(result.records.clone(), result.targets.clone());


            // Hier schicken wir node_data als output an die nächste node bzw. den output
            self.output.send(myoutput).map_err(|e| UpdateError::Other(e.into()))?;
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
    let input_data: DatasetBase<ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>, ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>> = DatasetBase::new(record_input.clone(), target_input.clone());

    let mut and: KmeansNode<> = KmeansNode::new(Some(&change_observer));
    let mock_output = flowrs::connection::Edge::new();
    flowrs::connection::connect(and.output.clone(), mock_output.clone());
    and.input.send(input_data)?;
    and.on_update()?;

    let expected = array![2, 0, 1, 2, 0, 2, 0, 1, 2, 0];
    let actual: DatasetBase<ArrayBase<ndarray::OwnedRepr<f64>, Dim<[usize; 2]>>, ArrayBase<ndarray::OwnedRepr<usize>, Dim<[usize; 1]>>> = mock_output.next()?;

    Ok(assert!(expected == actual.targets()))
}