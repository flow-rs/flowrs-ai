use flowrs::{node::{Node, UpdateError, ChangeObserver}, connection::{Input, Output}};
use flowrs::RuntimeConnectable;

use ndarray::prelude::*;
use linfa::traits::Transformer;
use linfa_kernel::{Kernel, KernelType, KernelMethod};
use linfa_reduction::DiffusionMap;
use serde::{Deserialize, Serialize};


// Definition eines Structs
#[derive(RuntimeConnectable, Deserialize, Serialize)]
pub struct DiffusionMapTestNode { // <--- Wenn man eine neue Node anlegt, einfach alles kopieren und hier den Namen ändern
    #[output]
    pub output: Output<Array2<f64>>, // <--- Wir haben in diesem Fall eine Output-Variable vom Typ Array2<u8>

    #[input]
    pub input: Input<Array2<f64>>, // <--- Wir haben in diesem Fall eine Input-Variable vom Typ Array2<u8>

    // Das bedeutet, unsere Node braucht als Input einen Array2<u8> und liefert als Output einen Array2<u8>
}

// Das ist einfach der Konstruktur
impl DiffusionMapTestNode {
    // Hier will der Konstruktur als einzigen Parameter einen ChangeObserver
    pub fn new(change_observer: Option<&ChangeObserver>) -> Self {
        Self {
            output: Output::new(change_observer),
            input: Input::new()
        }
    }
}

// Hier befinden sich die Methoden von unserer Node. Wir verwenden erstmal nur die Methoden, welche wir implementieren müssen, da diese von "Node" vorgegeben werden.
impl Node for DiffusionMapTestNode {
    // on_update wird von der Pipeline automatisch getriggert, wenn diese Node einen Input bekommt.
    fn on_update(&mut self) -> Result<(), UpdateError> {

        // Hier überprüfen wir nur, ob ein input da ist und der passt
        if let Ok(node_data) = self.input.next() {
            println!("JW-Debug: DiffusionMapTestNode has received: {}.", node_data);

            // #############################################################################
            // #############################################################################
            // Here begins the linfa_lib code
            // #############################################################################
            // #############################################################################

            // impl<'a, F: Float> Transformer<&'a KernelBase<ArrayBase<OwnedRepr<F>, Dim<[usize; 2]>>, CsMatBase<F, usize, Vec<usize, Global>, Vec<usize, Global>, Vec<F, Global>, usize>>, DiffusionMap<F>> for DiffusionMapValidParams
            // source
            // fn transform(&self, kernel: &'a Kernel<F>) -> DiffusionMap<F>
            // parameters: Kernel, embedding_size, steps
            
            // Generate sparse gaussian kernel with eps = 2 and 15 neighbors

            let kernel = Kernel::params()
                .kind(KernelType::Sparse(3))
                .method(KernelMethod::Gaussian(2.0))
                .transform(node_data.view());

            // Create embedding from kernel matrix using diffusion maps
            let mapped_kernel = DiffusionMap::<f64>::params(2)
                .steps(1)
                .transform(&kernel)
                .unwrap();

            // Get embedding from the transformed kernel matrix
            let embedding = mapped_kernel.embedding();
            println!("Embedding:\n{:?}\n", embedding);

            // #############################################################################
            // #############################################################################
            // Here ends the linfa_libe code
            // #############################################################################
            // #############################################################################

            // Hier schicken wir node_data als output an die nächste node bzw. den output
            self.output.send(embedding.clone()).map_err(|e| UpdateError::Other(e.into()))?;
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
    let test_input: Array2<f64> = array![[1.0, 2.0, 3.0, 4.0], [3.0, 4.0, 5.0, 6.0], [5.0, 6.0, 7.0, 8.0], [7.0, 4.0, 1.0, 9.0]];

    let mut and: DiffusionMapTestNode<> = DiffusionMapTestNode::new(Some(&change_observer));
    let mock_output = flowrs::connection::Edge::new();
    flowrs::connection::connect(and.output.clone(), mock_output.clone());
    and.input.send(test_input)?;
    and.on_update()?;

    let expected: Array2<f64> = array![[-0.6277633197008474, 0.7062768225572112],
                                        [-0.45945581589841195, -3.562607586318781e-7],
                                        [-0.6273416245953961, -0.7067513162800128],
                                        [1.5990652143251083e-6, 1.7822446319792776e-7]];
    let actual: Array2<f64> = mock_output.next()?;

    Ok(assert!(expected == actual))
}