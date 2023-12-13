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
// Definition eines Structs
#[derive(RuntimeConnectable, Deserialize, Serialize)]
pub struct DiffusionMapNode {
    #[output]
    pub output: Output<DatasetBase<ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>, ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>>>,

    #[input]
    pub input: Input<DatasetBase<ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>, ArrayBase<OwnedRepr<()>, Dim<[usize; 1]>>>>,

    #[input]
    pub config_input: Input<DiffusionMapConfig>
}

impl DiffusionMapNode {
    // Hier will der Konstruktur als einzigen Parameter einen ChangeObserver
    pub fn new(change_observer: Option<&ChangeObserver>) -> Self {
        Self {
            output: Output::new(change_observer),
            input: Input::new(),
            config_input: Input::new()
        }
    }
}

// Hier befinden sich die Methoden von unserer Node. Wir verwenden erstmal nur die Methoden, welche wir implementieren müssen, da diese von "Node" vorgegeben werden.
impl Node for DiffusionMapNode {
    // on_update wird von der Pipeline automatisch getriggert, wenn diese Node einen Input bekommt.
    fn on_update(&mut self) -> Result<(), UpdateError> {

        // Hier überprüfen wir nur, ob ein input da ist und der passt
        if let Ok(node_data) = self.input.next() {
            println!("JW-Debug: DiffusionMapNode has received an update!");
            //println!("JW-Debug DiffusionMapNode has received: {}.", node_data.records);

            let kernel = Kernel::params()
                .kind(KernelType::Sparse(3))
                .method(KernelMethod::Gaussian(2.0))
                .transform(node_data.records.view());

            // Generate sparse gaussian kernel with eps = 2 and 15 neighbors
            if let Ok(config) = self.config_input.next() {
                println!("JW-Debug DiffusionMapNode has received config.");

                // Create embedding from kernel matrix using diffusion maps
                let mapped_kernel = DiffusionMap::<f64>::params(config.embedding_size)
                .steps(config.steps)
                .transform(&kernel)
                .unwrap();

                // Get embedding from the transformed kernel matrix
                let embedding = mapped_kernel.embedding();
                //println!("Embedding:\n{:?}\n", embedding);
    
                let myoutput = DatasetBase::new(node_data.records, embedding.clone());

                // Hier schicken wir node_data als output an die nächste node bzw. den output
                self.output.send(myoutput).map_err(|e| UpdateError::Other(e.into()))?;

            } else {
                //Err(UpdateError::Other(anyhow::Error::msg("No config received!")));
            }   
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
    let expected: DatasetBase<ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>, ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>> = DatasetBase::new(test_input.clone(), expected_data.clone());

    let actual = mock_output.next()?;

    Ok(assert!(expected == actual))
}