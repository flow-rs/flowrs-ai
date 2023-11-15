use flowrs::{node::{Node, UpdateError, ChangeObserver}, connection::{Input, Output}};
use flowrs::RuntimeConnectable;

use ndarray::Array2;
use ndarray::prelude::*;
use linfa::traits::Transformer;
use linfa_preprocessing::norm_scaling::NormScaler;
use serde::{Deserialize, Serialize};


// Definition eines Structs
#[derive(RuntimeConnectable, Deserialize, Serialize)]
pub struct NormscalerNode { // <--- Wenn man eine neue Node anlegt, einfach alles kopieren und hier den Namen ändern
    #[output]
    pub output: Output<Array2<f64>>, // <--- Wir haben in diesem Fall eine Output-Variable vom Typ Array2<u8>

    #[input]
    pub input: Input<Array2<f64>>, // <--- Wir haben in diesem Fall eine Input-Variable vom Typ Array2<u8>

    // Das bedeutet, unsere Node braucht als Input einen Array2<u8> und liefert als Output einen Array2<u8>
}

// Das ist einfach der Konstruktur
impl NormscalerNode {
    // Hier will der Konstruktur als einzigen Parameter einen ChangeObserver
    pub fn new(change_observer: Option<&ChangeObserver>) -> Self {
        Self {
            output: Output::new(change_observer),
            input: Input::new()
        }
    }
}

// Hier befinden sich die Methoden von unserer Node. Wir verwenden erstmal nur die Methoden, welche wir implementieren müssen, da diese von "Node" vorgegeben werden.
impl Node for NormscalerNode {
    // on_update wird von der Pipeline automatisch getriggert, wenn diese Node einen Input bekommt.
    fn on_update(&mut self) -> Result<(), UpdateError> {

        // Hier überprüfen wir nur, ob ein input da ist und der passt
        if let Ok(node_data) = self.input.next() {
            println!("JW-Debug: NormscalerNode has received: {}.", node_data);

            // #############################################################################
            // #############################################################################
            // Here begins the linfa_lib code
            // #############################################################################
            // #############################################################################

            // impl<F: Float> Transformer<ArrayBase<OwnedRepr<F>, Dim<[usize; 2]>>, ArrayBase<OwnedRepr<F>, Dim<[usize; 2]>>> for NormScaler
            // fn transform(&self, x: Array2<F>) -> Array2<F>
            // Scales all samples in the array of shape (nsamples, nfeatures) to have unit norm.
            // parameters: l1 or l2 norm

            let scaler = NormScaler::l2();
            let normalized_data = scaler.transform(node_data.clone());
            println!("Data:\n{:?}\n", normalized_data);

            // #############################################################################
            // #############################################################################
            // Here ends the linfa_libe code
            // #############################################################################
            // #############################################################################

            // Hier schicken wir node_data als output an die nächste node bzw. den output
            self.output.send(normalized_data).map_err(|e| UpdateError::Other(e.into()))?;
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

    let mut and: NormscalerNode<> = NormscalerNode::new(Some(&change_observer));
    let mock_output = flowrs::connection::Edge::new();
    flowrs::connection::connect(and.output.clone(), mock_output.clone());
    and.input.send(test_input)?;
    and.on_update()?;

    let expected: Array2<f64> = array![[0.4472135954999579, 0.8944271909999159], [0.6, 0.8], [0.6401843996644799, 0.7682212795973759]];
    let actual: Array2<f64> = mock_output.next()?;

    Ok(assert!(expected == actual))
}