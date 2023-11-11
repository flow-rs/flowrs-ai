use flowrs::{node::{Node, UpdateError, ChangeObserver}, connection::{Input, Output}};
use flowrs::RuntimeConnectable;

use ndarray::{Array3, ArrayBase, OwnedRepr, Dim, arr2};
use anyhow::{anyhow};
use linfa::prelude::*;
use ndarray::Array2;
use ndarray::prelude::*;
use linfa::traits::{Fit, Predict, Transformer};
use linfa_preprocessing::linear_scaling::LinearScaler;
use linfa_preprocessing::norm_scaling::NormScaler;
use linfa_kernel::{Kernel, KernelType, KernelMethod};
use linfa_reduction::DiffusionMap;
use linfa_tsne::TSneParams;

use serde::{Deserialize, Serialize};


// Definition eines Structs
#[derive(RuntimeConnectable, Deserialize, Serialize)]
pub struct Linfa_Lib_Test_Node { // <--- Wenn man eine neue Node anlegt, einfach alles kopieren und hier den Namen ändern
    #[output]
    pub output: Output<u8>, // <--- Wir haben in diesem Fall eine Output-Variable vom Typ Array2<u8>

    #[input]
    pub input: Input<u8>, // <--- Wir haben in diesem Fall eine Input-Variable vom Typ Array2<u8>

    // Das bedeutet, unsere Node braucht als Input einen Array2<u8> und liefert als Output einen Array2<u8>
}

// Das ist einfach der Konstruktur
impl Linfa_Lib_Test_Node {
    // Hier will der Konstruktur als einzigen Parameter einen ChangeObserver
    pub fn new(change_observer: Option<&ChangeObserver>) -> Self {
        Self {
            output: Output::new(change_observer),
            input: Input::new()
        }
    }
}

// Hier befinden sich die Methoden von unserer Node. Wir verwenden erstmal nur die Methoden, welche wir implementieren müssen, da diese von "Node" vorgegeben werden.
impl Node for Linfa_Lib_Test_Node {
    // on_update wird von der Pipeline automatisch getriggert, wenn diese Node einen Input bekommt.
    fn on_update(&mut self) -> Result<(), UpdateError> {

        // Hier überprüfen wir nur, ob ein input da ist und der passt
        if let Ok(node_data) = self.input.next() {
            println!("JW-Debug: Linfa_Lib_Test_Node has received: {}.", node_data);

            // #############################################################################
            // #############################################################################
            // Here begins the linfa_lib code
            // #############################################################################
            // #############################################################################

            // impl<F: Float> Transformer<ArrayBase<OwnedRepr<F>, Dim<[usize; 2]>>, ArrayBase<OwnedRepr<F>, Dim<[usize; 2]>>> for NormScaler
            // fn transform(&self, x: Array2<F>) -> Array2<F>
            // Scales all samples in the array of shape (nsamples, nfeatures) to have unit norm.
            // parameters: l1 or l2 norm

            let data: ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>> = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
            let scaler = NormScaler::l2();
            let normalized_data = scaler.transform(data);
            println!("Data:\n{:?}\n", normalized_data);

            // impl<'a, F: Float> Transformer<&'a KernelBase<ArrayBase<OwnedRepr<F>, Dim<[usize; 2]>>, CsMatBase<F, usize, Vec<usize, Global>, Vec<usize, Global>, Vec<F, Global>, usize>>, DiffusionMap<F>> for DiffusionMapValidParams
            // source
            // fn transform(&self, kernel: &'a Kernel<F>) -> DiffusionMap<F>
            // parameters: Kernel, embedding_size, steps
            
            // Generate sparse gaussian kernel with eps = 2 and 15 neighbors
            let data: ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>> =
                array![[1.0, 2.0, 3.0, 4.0], [3.0, 4.0, 5.0, 6.0], [5.0, 6.0, 7.0, 8.0], [7.0, 4.0, 1.0, 9.0]];

            let kernel = Kernel::params()
                .kind(KernelType::Sparse(3))
                .method(KernelMethod::Gaussian(2.0))
                .transform(data.view());

            // Create embedding from kernel matrix using diffusion maps
            let mapped_kernel = DiffusionMap::<f64>::params(2)
                .steps(1)
                .transform(&kernel)
                .unwrap();

            // Get embedding from the transformed kernel matrix
            let embedding = mapped_kernel.embedding();
            println!("Embedding:\n{:?}\n", embedding);

            // impl<F: Float, R: Rng + Clone> Transformer<ArrayBase<OwnedRepr<F>, Dim<[usize; 2]>>, Result<ArrayBase<OwnedRepr<F>, Dim<[usize; 2]>>, TSneError>> for TSneParams<F, R>
            // source
            // fn transform(&self, x: Array2<F>) -> Result<Array2<F>>
            // parameters: embedding_size, perplexitiy, threshold

            let data: ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>> =
                array![[1.0, 2.0, 3.0, 4.0], [3.0, 4.0, 5.0, 6.0], [5.0, 6.0, 7.0, 8.0], [7.0, 4.0, 1.0, 9.0]];
            let ds = TSneParams::embedding_size(2)
                .perplexity(1.0)
                .approx_threshold(0.1)
                .transform(data)
                .unwrap();
            println!("t-SNE:\n{:?}\n", ds);





            // #############################################################################
            // #############################################################################
            // Here ends the linfa_libe code
            // #############################################################################
            // #############################################################################

            // Hier schicken wir node_data als output an die nächste node bzw. den output
            self.output.send(node_data).map_err(|e| UpdateError::Other(e.into()))?;
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
    let test_input: u8 = 1;
    let mut my_test_node: Linfa_Lib_Test_Node<> = Linfa_Lib_Test_Node::new(Some(&change_observer));

    let mock_output = flowrs::connection::Edge::new();
    flowrs::connection::connect(my_test_node.output.clone(), mock_output.clone());
    my_test_node.input.send(test_input)?;
    my_test_node.on_update()?;

    let expected = 1;
    let actual = mock_output.next()?;
    Ok(assert!(expected == actual))
}