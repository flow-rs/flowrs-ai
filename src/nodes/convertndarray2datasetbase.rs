use flowrs::{node::{Node, UpdateError, ChangeObserver}, connection::{Input, Output}};
use flowrs::RuntimeConnectable;

use ndarray::{Array2, array, OwnedRepr};
use ndarray::prelude::*;
use serde::{Deserialize, Serialize};
use linfa::prelude::*;


// Definition eines Structs
#[derive(RuntimeConnectable, Deserialize, Serialize)]
pub struct ConvertNdarray2DatasetBase { // <--- Wenn man eine neue Node anlegt, einfach alles kopieren und hier den Namen ändern
    #[input]
    pub input: Input<Array2<f64>>, // <--- Wir haben in diesem Fall eine Output-Variable vom Typ Array2<u8>

    #[output]
    //pub output: Output<DatasetBase<(), ndarray::ArrayBase<ndarray::OwnedRepr<f64>, ndarray::Dim<[usize; 2]>>>>, // <--- Wir haben in diesem Fall eine Output-Variable vom Typ Array2<u8>
    pub output: Output<DatasetBase<ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>, ArrayBase<OwnedRepr<()>, Dim<[usize; 1]>>>>
}

// Das ist einfach der Konstruktur
impl ConvertNdarray2DatasetBase {
    // Hier will der Konstruktur als einzigen Parameter einen ChangeObserver
    pub fn new(change_observer: Option<&ChangeObserver>) -> Self {
        Self {
            output: Output::new(change_observer),
            input: Input::new()
        }
    }
}

// Hier befinden sich die Methoden von unserer Node. Wir verwenden erstmal nur die Methoden, welche wir implementieren müssen, da diese von "Node" vorgegeben werden.
impl Node for ConvertNdarray2DatasetBase {
    // on_update wird von der Pipeline automatisch getriggert, wenn diese Node einen Input bekommt.
    fn on_update(&mut self) -> Result<(), UpdateError> {

        // Hier überprüfen wir nur, ob ein input da ist und der passt
        if let Ok(node_data) = self.input.next() {
            println!("JW-Debug: ConvertNdarray2DatasetBase has received: {}.", node_data);

            let dataset = Dataset::from(node_data.clone());

            println!("DatasetBase\n");
            println!("Records:\n {}\n", dataset.records);
            println!("Targets:\n {:?}\n", dataset.targets);
            println!("Feature names:\n {:?}\n", dataset.feature_names());

            // Hier schicken wir node_data als output an die nächste node bzw. den output
            self.output.send(dataset).map_err(|e| UpdateError::Other(e.into()))?;
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
    let test_input: Array2<f64> = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];

    let mut test_node: ConvertNdarray2DatasetBase<> = ConvertNdarray2DatasetBase::new(Some(&change_observer));
    let mock_output = flowrs::connection::Edge::new();
    flowrs::connection::connect(test_node.output.clone(), mock_output.clone());
    test_node.input.send(test_input.clone())?;
    test_node.on_update()?;

    let actual: DatasetBase<ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>, ArrayBase<OwnedRepr<()>, Dim<[usize; 1]>>> = mock_output.next()?;



    Ok(assert!(true))
}