use flowrs::RuntimeConnectable;
use flowrs::{
    connection::{Input, Output},
    node::{Node, UpdateError, ChangeObserver},
};
use ndarray::{Array1};

#[derive(RuntimeConnectable)]
pub struct MaxOutputNode {
    #[input]
    pub output_tensor: Input<Array1<f32>>,
    #[input]
    pub input_classes: Input<Vec<String>>,
    pub classes: Option<Vec<String>>,
    #[output]
    pub output_class: Output<String>,
}

impl MaxOutputNode 
{
    pub fn new(change_observer:Option<&ChangeObserver>) -> Self {
        Self {
            output_tensor: Input::new(),
            input_classes: Input::new(),
            classes: None,
            output_class: Output::new(change_observer),
        }
    }

    fn get_max_output(&mut self, tensor: Array1<f32>) -> Option<String> {
        if let Some(classes) = &self.classes {
            if tensor.len() != classes.len() {
                eprintln!("Input tensor and classes need to have the same size!");
                return None;
            }
            let max_index = tensor.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                .map(|(index, _)| index);
            match max_index {
                Some(max_index) => {
                    let classes = &self.classes.as_mut().expect("no class vector provided");
                    let class = &classes[max_index];
                    Some(class.clone())
                }
                None => {
                    panic!("No max index found")
                }
            }
        } else {
            None
        }  
    }
}

impl Node for MaxOutputNode
{
    fn on_update(&mut self) -> anyhow::Result<(), UpdateError> {
        if let Ok(input_classes) = self.input_classes.next() {
            self.classes = Some(input_classes);
        }
        if let Ok(output_tensor) = self.output_tensor.next() {
            let result = self.get_max_output(output_tensor);
            match result {
                Some(result) => {
                    let _ = self.output_class.send(result);
                }
                None => {
                    UpdateError::SendError { message: "Could not find result".to_string() };
                }
            }
            
        }
        Ok(())
    }
}
