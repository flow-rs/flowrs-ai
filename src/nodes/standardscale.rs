use flowrs::{node::{Node, UpdateError, ChangeObserver}, connection::{Input, Output}};
use flowrs::RuntimeConnectable;

use linfa::prelude::*;
use ndarray::prelude::*;
use linfa::traits::{Fit, Transformer};
use linfa_preprocessing::linear_scaling::LinearScaler;
use serde::{Deserialize, Serialize};


// Definition eines Structs
#[derive(RuntimeConnectable, Deserialize, Serialize)]
pub struct StandardscaleNode { // <--- Wenn man eine neue Node anlegt, einfach alles kopieren und hier den Namen ändern
    #[output]
    pub output: Output<Array2<f64>>, // <--- Wir haben in diesem Fall eine Output-Variable vom Typ Array2<u8>

    #[input]
    pub input: Input<Array2<f64>>, // <--- Wir haben in diesem Fall eine Input-Variable vom Typ Array2<u8>

    // Das bedeutet, unsere Node braucht als Input einen Array2<u8> und liefert als Output einen Array2<u8>
}

// Das ist einfach der Konstruktur
impl StandardscaleNode {
    // Hier will der Konstruktur als einzigen Parameter einen ChangeObserver
    pub fn new(change_observer: Option<&ChangeObserver>) -> Self {
        Self {
            output: Output::new(change_observer),
            input: Input::new()
        }
    }
}

// Hier befinden sich die Methoden von unserer Node. Wir verwenden erstmal nur die Methoden, welche wir implementieren müssen, da diese von "Node" vorgegeben werden.
impl Node for StandardscaleNode {
    // on_update wird von der Pipeline automatisch getriggert, wenn diese Node einen Input bekommt.
    fn on_update(&mut self) -> Result<(), UpdateError> {

        // Hier überprüfen wir nur, ob ein input da ist und der passt
        if let Ok(node_data) = self.input.next() {
            println!("JW-Debug: StandardscaleNode has received: {}.", node_data);

            // #############################################################################
            // #############################################################################
            // Here begins the linfa_lib code
            // #############################################################################
            // #############################################################################

            // impl<F: Float, D: Data<Elem = F>, T: AsTargets> Fit<ArrayBase<D, Dim<[usize; 2]>>, T, PreprocessingError> for LinearScalerParams<F>
            // fn fit(&self, x: &DatasetBase<ArrayBase<D, Ix2>, T>) -> Result<Self::Object>
            // Fits the input dataset accordng to the scaler method. Will return an error if the dataset does not contain any samples or
            // (in the case of MinMax scaling) if the specified range is not valid.

            // impl<F: Float> Transformer<ArrayBase<OwnedRepr<F>, Dim<[usize; 2]>>, ArrayBase<OwnedRepr<F>, Dim<[usize; 2]>>> for LinearScaler<F>
            // fn transform(&self, x: Array2<F>) -> Array2<F>
            // Scales an array of size (nsamples, nfeatures) according to the scaler’s offsets and scales. 
            // Panics if the shape of the input array is not compatible with the shape of the dataset used for fitting.

            // Needs DatasetBase für fitting
            let dataset: DatasetBase<ArrayBase<ndarray::OwnedRepr<f64>, Dim<[usize; 2]>>, ArrayBase<ndarray::OwnedRepr<()>, Dim<[usize; 1]>>> = DatasetBase::from(node_data.clone());
            // Learn scaling parameters
            let scaler = LinearScaler::standard().fit(&dataset).unwrap();
            // scale dataset according to parameters
            let standard_scaled_data = scaler.transform(node_data);
            println!("Data:\n{:?}\n", standard_scaled_data);

            // #############################################################################
            // #############################################################################
            // Here ends the linfa_libe code
            // #############################################################################
            // #############################################################################

            // Hier schicken wir node_data als output an die nächste node bzw. den output
            self.output.send(standard_scaled_data).map_err(|e| UpdateError::Other(e.into()))?;
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

    let mut and: StandardscaleNode<> = StandardscaleNode::new(Some(&change_observer));
    let mock_output = flowrs::connection::Edge::new();
    flowrs::connection::connect(and.output.clone(), mock_output.clone());
    and.input.send(test_input)?;
    and.on_update()?;

    let expected: Array2<f64> = array![[-1.224744871391589, -1.224744871391589],
                                        [0.0, 0.0],
                                        [1.224744871391589, 1.224744871391589]];
    let actual: Array2<f64> = mock_output.next()?;

    Ok(assert!(expected == actual))
}