use flowrs::{node::{Node, UpdateError, ChangeObserver}, connection::{Input, Output}};
use flowrs::RuntimeConnectable;

use ndarray::{Array2, arr2};
use serde::{Deserialize, Serialize};


#[derive(RuntimeConnectable, Deserialize, Serialize)]
pub struct ScaleNode {
    #[output]
    pub output: Output<Array2<u8>>,

    #[input]
    pub input: Input<Array2<u8>>,
}

impl ScaleNode {
    pub fn new(change_observer: Option<&ChangeObserver>) -> Self {
        Self {
            output: Output::new(change_observer),
            input: Input::new()
        }
    }
}

impl Node for ScaleNode {
    fn on_update(&mut self) -> Result<(), UpdateError> {

        if let Ok(data) = self.input.next() {
            println!("JW-Debug: ScaleNode has received: {}.", data);

            self.output.send(data).map_err(|e| UpdateError::Other(e.into()))?;
        }
        Ok(())
    }
}

#[test]
fn input_output_test() -> Result<(), UpdateError> {
    let change_observer = ChangeObserver::new();
    let test_input: Array2<u8> = arr2(&[[1, 2, 3], [4, 5, 6], [7, 8, 9]]);

    let mut and: ScaleNode<> = ScaleNode::new(Some(&change_observer));
    let mock_output = flowrs::connection::Edge::new();
    flowrs::connection::connect(and.output.clone(), mock_output.clone());
    and.input.send(test_input)?;
    and.on_update()?;

    let expected = arr2(&[[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
    let actual = mock_output.next()?;
    Ok(assert!(expected == actual))
}