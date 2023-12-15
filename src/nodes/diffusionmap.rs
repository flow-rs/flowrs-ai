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
pub struct DiffusionMapNode<T> 
where
    T: Clone,
{
    #[output]
    pub output: Output<DatasetBase<Array2<T>, Array1<()>>>,

    #[input]
    pub input: Input<DatasetBase<Array2<T>, Array1<()>>>,

    #[input]
    pub config_input: Input<DiffusionMapConfig>
}

impl<T> DiffusionMapNode<T> 
where
    T: Clone,
{
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
impl Node for DiffusionMapNode<f32> {
    // on_update wird von der Pipeline automatisch getriggert, wenn diese Node einen Input bekommt.
    fn on_update(&mut self) -> Result<(), UpdateError> {

        // Hier überprüfen wir nur, ob ein input da ist und der passt
        if let Ok(node_data) = self.input.next() {
            println!("JW-Debug DiffusionMapNode has received: {}.", node_data.records);

            // #############################################################################
            // #############################################################################
            // Here begins the linfa_lib code
            // #############################################################################
            // #############################################################################

            // impl<'a, F: Float> Transformer<&'a KernelBase<ArrayBase<OwnedRepr<F>, Dim<[usize; 2]>>, CsMatBase<F, usize, Vec<usize, Global>, Vec<usize, Global>, Vec<F, Global>, usize>>, DiffusionMap<F>> for DiffusionMapValidParams
            // source
            // fn transform(&self, kernel: &'a Kernel<F>) -> DiffusionMap<F>
            // parameters: Kernel, embedding_size, steps
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
    
                let myoutput = DatasetBase::from(embedding.clone());

                // Hier schicken wir node_data als output an die nächste node bzw. den output
                self.output.send(myoutput).map_err(|e| UpdateError::Other(e.into()))?;

            } else {
                //Err(UpdateError::Other(anyhow::Error::msg("No config received!")));
            }   
        }
        Ok(())
    }
}

impl Node for DiffusionMapNode<f64> {
    // on_update wird von der Pipeline automatisch getriggert, wenn diese Node einen Input bekommt.
    fn on_update(&mut self) -> Result<(), UpdateError> {

        // Hier überprüfen wir nur, ob ein input da ist und der passt
        if let Ok(node_data) = self.input.next() {
            println!("JW-Debug DiffusionMapNode has received: {}.", node_data.records);

            let kernel = Kernel::params()
                .kind(KernelType::Sparse(3))
                .method(KernelMethod::Gaussian(2.0))
                .transform(node_data.records.view());

            if let Ok(config) = self.config_input.next() {
                println!("JW-Debug DiffusionMapNode has received config.");

                let mapped_kernel = DiffusionMap::<f64>::params(config.embedding_size)
                .steps(config.steps)
                .transform(&kernel)
                .unwrap();

                let embedding = mapped_kernel.embedding();

                let myoutput = DatasetBase::from(embedding.clone());

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
    let test_input: Array2<f32> = array![[1.1, 2.5, 3.2, 4.6, 5.2, 6.7], 
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

    let mut test_node: DiffusionMapNode<f32> = DiffusionMapNode::new(Some(&change_observer));
    let mock_output = flowrs::connection::Edge::new();
    flowrs::connection::connect(test_node.output.clone(), mock_output.clone());
    test_node.input.send(test_dataset)?;
    test_node.config_input.send(test_config_input)?;
    test_node.on_update()?;


    let expected: Array2<f32> = array![[0., 0.],
    [-0.34922323, -0.000000069743095],
    [0.00000018380972, -0.24999961],
    [0.000000015326316, -0.000015808142],
    [0.00000000012654844, -0.0005489692],
    [0.0000000000048550244, 0.000000000000068383836],
    [-0.3492234, -0.00000014807607],
    [0.00000013084646, -0.24999957],
    [0.000000007479324, -0.000015785135],
    [-0.00000000015578243, -0.00054900756]];


    let actual = mock_output.next()?.records;

    println!("{}", actual);
    
    Ok(assert!(expected == actual))
}