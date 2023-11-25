use flowrs::{node::{Node, UpdateError, ChangeObserver}, connection::{Input, Output}};
use flowrs::RuntimeConnectable;

use linfa::prelude::*;
use ndarray::{prelude::*, OwnedRepr};
use linfa::traits::{Fit, Transformer};
use linfa_preprocessing::linear_scaling::LinearScaler;
use serde::{Deserialize, Serialize};


// Definition eines Structs
#[derive(RuntimeConnectable, Deserialize, Serialize)]
pub struct StandardscaleNode { // <--- Wenn man eine neue Node anlegt, einfach alles kopieren und hier den Namen ändern
    #[output]
    pub output: Output<DatasetBase<ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>, ArrayBase<OwnedRepr<()>, Dim<[usize; 1]>>>>, // <--- Wir haben in diesem Fall eine Output-Variable vom Typ Array2<u8>

    #[input]
    pub input: Input<DatasetBase<ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>, ArrayBase<OwnedRepr<()>, Dim<[usize; 1]>>>>, // <--- Wir haben in diesem Fall eine Input-Variable vom Typ Array2<u8>
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
            println!("JW-Debug: StandardscaleNode has received: {}.", node_data.records);

            // Learn scaling parameters
            let scaler = LinearScaler::standard().fit(&node_data).unwrap();
            // scale dataset according to parameters
            let standard_scaled_data = scaler.transform(node_data);
            println!("Data:\n{:?}\n", standard_scaled_data);

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

    let mut test_node: StandardscaleNode<> = StandardscaleNode::new(Some(&change_observer));
    let mock_output = flowrs::connection::Edge::new();
    flowrs::connection::connect(test_node.output.clone(), mock_output.clone());
    test_node.input.send(dataset)?;
    test_node.on_update()?;

    let expected_data: Array2<f64> = array![[-1.4143377359247322, -1.3343815168527249, -1.3919452566240385, -0.5892455532801077, -0.6668399873113844, -0.5645401987746118],
    [0.16764270325652061, 0.031143554253294955, 0.1019653254426467, 0.7596298096502591, 0.7046748802084846, 0.6979336634639715],
    [1.3954484172479409, 1.4206252055541575, 1.406172976453245, -1.4411668351308655, -1.4235378452533811, -1.4458898761864531],
    [-0.7295999338910557, -0.7354670119816634, -0.727984997927734, -0.021298032046269122, -0.004729361612137379, -0.01667418308617004],
    [0.6870989668682753, 0.7019277997088839, 0.6947869849929187, 1.3985707710383273, 1.4140791220291065, 1.412541510014113],
    [-1.4379493842707212, -1.454164417826937, -1.4393709893880604, -0.7312324335885673, -0.7141336034327592, -0.7312820296363116],
    [-0.021250483511390143, -0.016769606136389788, -0.01659900646740768, 0.688636369496029, 0.7046748802084846, 0.6979336634639715],
    [1.3954484172479409, 1.4206252055541575, 1.406172976453245, -1.4411668351308655, -1.4235378452533811, -1.4458898761864531],
    [-0.7295999338910557, -0.7354670119816634, -0.727984997927734, -0.021298032046269122, -0.004729361612137379, -0.01667418308617004],
    [0.6870989668682753, 0.7019277997088839, 0.6947869849929187, 1.3985707710383273, 1.4140791220291065, 1.412541510014113]];
    let actual: DatasetBase<ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>, ArrayBase<OwnedRepr<()>, Dim<[usize; 1]>>> = mock_output.next()?;
    let expected: DatasetBase<ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>, ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>> = DatasetBase::new(expected_data.clone(), expected_data.clone());

    Ok(assert!(expected.records == actual.records))
}