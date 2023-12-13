use flowrs::{node::{Node, UpdateError, ChangeObserver}, connection::{Input, Output}};
use flowrs::RuntimeConnectable;

use ndarray::{prelude::*, OwnedRepr};
use linfa::{traits::Transformer, DatasetBase, Dataset};
use linfa_kernel::{Kernel, KernelType, KernelMethod};
use linfa_reduction::DiffusionMap;
use serde::{Deserialize, Serialize};

#[derive(Clone, Deserialize, Serialize)]
pub struct DiffusionMapConfig {
   pub embedding_size: usize,
   pub steps: usize
}

impl DiffusionMapConfig {
    pub fn new(embedding_size: usize, steps: usize) -> Self {
        DiffusionMapConfig {
            embedding_size,
            steps,
        }
    }
}

#[derive(RuntimeConnectable, Deserialize, Serialize)]
pub struct DiffusionMapNode {
    #[output]
    pub output: Output<DatasetBase<Array2<f64>, Array1<()>>>,

    #[input]
    pub input: Input<DatasetBase<Array2<f64>, Array1<()>>>,

    #[input]
    pub config_input: Input<DiffusionMapConfig>,

    config: DiffusionMapConfig
}

impl DiffusionMapNode {
    pub fn new(change_observer: Option<&ChangeObserver>) -> Self {
        Self {
            input: Input::new(),
            config_input: Input::new(),
            output: Output::new(change_observer),
            config: DiffusionMapConfig::new(2, 1)
        }
    }
}

impl Node for DiffusionMapNode {
    fn on_update(&mut self) -> Result<(), UpdateError> {
        println!("JW-Debug: DiffusionMapNode has received an update!");

        // Neue Config kommt an
        if let Ok(config) = self.config_input.next() {
            println!("JW-Debug: DbscanNode has received config: {}, {}", config.embedding_size, config.steps);

            self.config = config;
        }

        // Daten kommen an
        if let Ok(node_data) = self.input.next() {
            println!("JW-Debug: DiffusionMapNode has received an update!");     //println!("JW-Debug DiffusionMapNode has received: {}.", node_data.records);

            let kernel = Kernel::params()
            .kind(KernelType::Sparse(3))
            .method(KernelMethod::Gaussian(2.0))
            .transform(node_data.records.view());

            let mapped_kernel = DiffusionMap::<f64>::params(self.config.embedding_size)
            .steps(self.config.steps)
            .transform(&kernel)
            .unwrap();

            let embedding = mapped_kernel.embedding();
            let embedding_result = DatasetBase::from(embedding.clone());

            self.output.send(embedding_result).map_err(|e| UpdateError::Other(e.into()))?;
            println!("JW-Debug: DiffusionMapNode has sent an output!");
        }

        Ok(())
    }
}



#[test]
fn new_config_test() -> Result<(), UpdateError> {
    let change_observer = ChangeObserver::new();
    let test_config_input = DiffusionMapConfig{
        embedding_size: 2,
        steps: 1
    };
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
    let test_dataset = Dataset::from(test_input.clone());

    let mut test_node: DiffusionMapNode<> = DiffusionMapNode::new(Some(&change_observer));
    let mock_output = flowrs::connection::Edge::new();
    flowrs::connection::connect(test_node.output.clone(), mock_output.clone());
    test_node.input.send(test_dataset)?;
    test_node.config_input.send(test_config_input)?;
    test_node.on_update()?;


    let expected_data: Array2<f64> = array![[2.1218594193914674e-15, 5.3720002361451115e-17],
    [0.34922335766038987, -2.383630731487172e-13],
    [-4.964803802882625e-14, 0.24999992391726172],
    [1.9660992574078857e-12, -2.9521920719939894e-8],
    [2.4591022380216627e-10, 0.0001950420188612224],
    [1.8910998157060896e-15, 1.4485925063480052e-16],
    [0.3492233578927868, -3.076657427221702e-14],
    [-3.3125455168563386e-14, 0.24999992391720385],
    [1.942479529165688e-12, -2.952213544990568e-8],
    [2.4594727976304506e-10, 0.00019504201876301383]];
    let expected: DatasetBase<Array2<f64>, Array1<()>> = DatasetBase::from(expected_data.clone());

    let actual = mock_output.next()?;

    Ok(assert!(expected == actual))
}

#[test]
fn default_config_test() -> Result<(), UpdateError> {
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
    let test_dataset = Dataset::from(test_input.clone());

    let mut test_node: DiffusionMapNode<> = DiffusionMapNode::new(Some(&change_observer));
    let mock_output = flowrs::connection::Edge::new();
    flowrs::connection::connect(test_node.output.clone(), mock_output.clone());
    test_node.input.send(test_dataset)?;
    test_node.on_update()?;


    let expected_data: Array2<f64> = array![[2.1218594193914674e-15, 5.3720002361451115e-17],
    [0.34922335766038987, -2.383630731487172e-13],
    [-4.964803802882625e-14, 0.24999992391726172],
    [1.9660992574078857e-12, -2.9521920719939894e-8],
    [2.4591022380216627e-10, 0.0001950420188612224],
    [1.8910998157060896e-15, 1.4485925063480052e-16],
    [0.3492233578927868, -3.076657427221702e-14],
    [-3.3125455168563386e-14, 0.24999992391720385],
    [1.942479529165688e-12, -2.952213544990568e-8],
    [2.4594727976304506e-10, 0.00019504201876301383]];
    let expected: DatasetBase<Array2<f64>, Array1<()>> = DatasetBase::from(expected_data.clone());

    let actual = mock_output.next()?;

    Ok(assert!(expected == actual))
}