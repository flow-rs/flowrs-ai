use flowrs::{node::{Node, UpdateError, ChangeObserver}, connection::{Input, Output}};
use flowrs::RuntimeConnectable;

use linfa::DatasetBase;
use ndarray::{Array2, Array1};

use crate::csv2dataset::CSVToDatasetNode;


#[derive(RuntimeConnectable)]
pub struct TestNode<T,U>
where
    T: Clone,
    U: Clone
{
    #[output]
    pub output: Output<DatasetBase<Array2<T>, Array1<()>>>,

    #[input]
    pub data_input: Input<DatasetBase<Array2<T>, Array1<U>>>,

    #[input]
    pub config_input: Input<DatasetBase<Array2<T>, Array1<()>>>,

    config: Option<DatasetBase<Array2<T>, Array1<()>>>
}


impl<T,U> TestNode<T,U> 
where
    T: Clone,
    U: Clone
{
    pub fn new(change_observer: Option<&ChangeObserver>) -> Self {
        Self {
            output: Output::new(change_observer),
            data_input: Input::new(),
            config_input: Input::new(),
            config: None
        }
    }
}


impl<T,U> Node for TestNode<T,U>
where
    T: Clone + Send,
    U: Clone + Send,
{
    fn on_update(&mut self) -> Result<(), UpdateError> {

        // receiving config
        if let Ok(config) = self.config_input.next() {
            self.config = Some(config);
        }

        match self.config.clone() {
            Some(config) => self.output.send(config).map_err(|e| UpdateError::Other(e.into()))?,
            None => {},
        }
        
        Ok(())
    }
}

