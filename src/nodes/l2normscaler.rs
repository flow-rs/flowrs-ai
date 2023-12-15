use flowrs::{node::{Node, UpdateError, ChangeObserver}, connection::{Input, Output}};
use flowrs::RuntimeConnectable;

use ndarray::{Array2, Array1, array, ArrayBase, OwnedRepr, Dim};
use linfa::traits::Transformer;
use linfa_preprocessing::norm_scaling::NormScaler;
use serde::{Deserialize, Serialize};
use linfa::prelude::*;


// Definition eines Structs
#[derive(RuntimeConnectable, Deserialize, Serialize)]
pub struct L2NormscalerNode<T>
where
    T: Clone,
{ // <--- Wenn man eine neue Node anlegt, einfach alles kopieren und hier den Namen ändern
    #[output]
    pub output: Output<DatasetBase<Array2<T>, Array1<()>>>, // <--- Wir haben in diesem Fall eine Output-Variable vom Typ Array2<u8>

    #[input]
    pub input: Input<DatasetBase<Array2<T>, Array1<()>>>, // <--- Wir haben in diesem Fall eine Input-Variable vom Typ Array2<u8>

    // Das bedeutet, unsere Node braucht als Input einen Array2<u8> und liefert als Output einen Array2<u8>
}

// Das ist einfach der Konstruktur
impl<T> L2NormscalerNode<T> 
where
    T: Clone,
{
    // Hier will der Konstruktur als einzigen Parameter einen ChangeObserver
    pub fn new(change_observer: Option<&ChangeObserver>) -> Self {
        Self {
            output: Output::new(change_observer),
            input: Input::new()
        }
    }
}

// Hier befinden sich die Methoden von unserer Node. Wir verwenden erstmal nur die Methoden, welche wir implementieren müssen, da diese von "Node" vorgegeben werden.
impl<T> Node for L2NormscalerNode<T> 
where
    T: Clone + Send + linfa::Float
{
    // on_update wird von der Pipeline automatisch getriggert, wenn diese Node einen Input bekommt.
    fn on_update(&mut self) -> Result<(), UpdateError> {

        // Hier überprüfen wir nur, ob ein input da ist und der passt
        if let Ok(node_data) = self.input.next() {
            println!("JW-Debug: L2NormscalerNode has received: {}.", node_data.records);

            let scaler = NormScaler::l1();
            let normalized_data = scaler.transform(node_data.clone());
            println!("Data:\n{:?}\n", normalized_data);
    
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
    let test_input: Array2<f64> = array![[1.0, 2.0, 3.0, 4.0],
    [3.0, 4.0, 5.0, 6.0],
    [5.0, 6.0, 7.0, 8.0],
    [7.0, 4.0, 1.0, 9.0]];

    let dataset = Dataset::from(test_input.clone());

    let mut test_node: L2NormscalerNode<f64> = L2NormscalerNode::new(Some(&change_observer));
    let mock_output = flowrs::connection::Edge::new();
    flowrs::connection::connect(test_node.output.clone(), mock_output.clone());
    test_node.input.send(dataset)?;
    test_node.on_update()?;

    let expected_data = array![[0.1, 0.2, 0.3, 0.4],
    [0.16666666666666666, 0.2222222222222222, 0.2777777777777778, 0.3333333333333333],
    [0.19230769230769232, 0.23076923076923078, 0.2692307692307692, 0.3076923076923077],
    [0.3333333333333333, 0.19047619047619047, 0.047619047619047616, 0.42857142857142855]];

    let actual: DatasetBase<ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>, ArrayBase<OwnedRepr<()>, Dim<[usize; 1]>>> = mock_output.next()?;
    let expected: DatasetBase<ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>, ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>> = DatasetBase::new(expected_data.clone(), expected_data.clone());

    Ok(assert!(expected.records == actual.records))
}