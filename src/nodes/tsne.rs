use flowrs::{node::{Node, UpdateError, ChangeObserver}, connection::{Input, Output}};
use flowrs::RuntimeConnectable;

use ndarray::{Array2, OwnedRepr};
use ndarray::prelude::*;
use linfa::{traits::Transformer, DatasetBase, Dataset};
use linfa_tsne::TSneParams;
use serde::{Deserialize, Serialize};

#[derive(Clone, Deserialize, Serialize)]
pub struct TsneConfig {
   pub embedding_size: usize,
   pub perplexity: f64,
   pub approx_threshold: f64,
}  

// Definition eines Structs
#[derive(RuntimeConnectable, Deserialize, Serialize)]
pub struct TsneNode {
    #[output]
    pub output: Output<DatasetBase<ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>, ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>>>,

    #[input]
    pub input: Input<DatasetBase<ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>, ArrayBase<OwnedRepr<()>, Dim<[usize; 1]>>>>,

    #[input]
    pub config_input: Input<TsneConfig>
}

// Das ist einfach der Konstruktur
impl TsneNode {
    pub fn new(change_observer: Option<&ChangeObserver>) -> Self {
        Self {
            output: Output::new(change_observer),
            input: Input::new(),
            config_input: Input::new()
        }
    }
}

// Hier befinden sich die Methoden von unserer Node.
impl Node for TsneNode {
    // on_update wird von der Pipeline automatisch getriggert, wenn diese Node einen Input bekommt.
    fn on_update(&mut self) -> Result<(), UpdateError> {
        if let Ok(config) = self.config_input.next() {
                
            if let Ok(node_data) = self.input.next() {
                println!("JW-Debug: TsneNode has received: {}.", node_data.records);
    
                let dataset = TSneParams::embedding_size(config.embedding_size)
                    .perplexity(config.perplexity)
                    .approx_threshold(config.approx_threshold)
                    .transform(node_data.clone())
                    .unwrap();
                println!("t-SNE:\n{:?}\n", dataset);
    
                let myoutput: DatasetBase<ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>, ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>> = DatasetBase::new(node_data.records, dataset.records);
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
    let test_config_input = TsneConfig{
        embedding_size: 2,
        perplexity: 1.0,
        approx_threshold: 0.1
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
    let dataset: DatasetBase<ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>, ArrayBase<OwnedRepr<()>, Dim<[usize; 1]>>> = Dataset::from(test_input.clone());
    let mut and: TsneNode<> = TsneNode::new(Some(&change_observer));
    let mock_output = flowrs::connection::Edge::new();
    flowrs::connection::connect(and.output.clone(), mock_output.clone());
    and.input.send(dataset)?;
    and.config_input.send(test_config_input)?;
    and.on_update()?;

    let expected_data: Array2<f64> = array![[1699.4869710195842, 28.635799050127282],
    [-2855.4029300031552, -1650.0023484842843],
    [-401.11703947700613, 1213.3522351242418],
    [-2318.953494427868, 3207.352709992378],
    [3698.4231386786414, -2017.4342429070707],
    [1581.1775664430165, -646.5528618675547],
    [-2181.4630513740067, -1801.7723410306903],
    [-315.13553376905844, 275.25601936580455],
    [-3002.8827027429775, 2507.5112437087837],
    [4095.867075652829, -1116.3462129517347]];
    let expected: DatasetBase<ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>, ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>> = DatasetBase::new(test_input.clone(), expected_data.clone());

    let actual = mock_output.next()?;


    println!("Actual\n");
    println!("Records:\n {}\n", actual.records.clone());
    println!("Targets:\n {:?}\n", actual.targets.clone());
    println!("Feature names:\n {:?}\n", actual.feature_names().clone());

    println!("Expected\n");
    println!("Records:\n {}\n", expected.records.clone());
    println!("Targets:\n {:?}\n", expected.targets.clone());
    println!("Feature names:\n {:?}\n", expected.feature_names().clone());


    Ok(assert!(true))
}