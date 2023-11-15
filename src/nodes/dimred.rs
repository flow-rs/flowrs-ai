use flowrs::{node::{Node, UpdateError, ChangeObserver}, connection::{Input, Output}};
use flowrs::RuntimeConnectable;
use ndarray::{Array2, ArrayBase, OwnedRepr, Dim, arr2};
use linfa::prelude::*;
use linfa::traits::{Fit, Predict};
use linfa_reduction::Pca;

use serde::{Deserialize, Serialize};


// Definition eines Structs
#[derive(RuntimeConnectable, Deserialize, Serialize)]
pub struct DimRedNode { // <--- Wenn man eine neue Node anlegt, einfach alles kopieren und hier den Namen ändern   
    #[input]
    pub input: Input<DatasetBase<ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>, Vec<u32>>>,

    #[output]
    pub output: Output<DatasetBase<ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>, Vec<u32>>>,
}

// Das ist einfach der Konstruktur
impl DimRedNode {
    // Hier will der Konstruktur als einzigen Parameter einen ChangeObserver
    pub fn new(change_observer: Option<&ChangeObserver>) -> Self {
        Self {
            output: Output::new(change_observer),
            input: Input::new()
        }
    }
}

// Hier befinden sich die Methoden von unserer Node. Wir verwenden erstmal nur die Methoden, welche wir implementieren müssen, da diese von "Node" vorgegeben werden.
impl Node for DimRedNode {
    // on_update wird von der Pipeline automatisch getriggert, wenn diese Node einen Input bekommt.
    fn on_update(&mut self) -> Result<(), UpdateError> {

        // Hier überprüfen wir nur, ob ein input da ist und der passt
        if let Ok(dataset) = self.input.next() {
            println!("JW-Debug: DimRedNode has received: {}.", dataset.records);

            // #############################################################################
            // #############################################################################
            // Here begins the dimred_cluster_code
            // #############################################################################
            // #############################################################################

            

            // ################################ DIM RED ################################
            // apply PCA projection along a line which maximizes the spread of the data

            let embedding = Pca::params(2)
            .fit(&dataset).unwrap();
        
            // reduce dimensionality of the dataset
            let transformed_dataset = embedding.predict(dataset);


            

            // You can access the data and labels as follows:
            let dataset_data = transformed_dataset.records(); // Use the transformed dataset
            let dataset_labels = transformed_dataset.targets(); // Use the transformed dataset

            println!("Data:\n{:?}\n", dataset_data);
            println!("Labels: {:?}\n", dataset_labels);

            let labels: Vec<f64> = dataset_labels.clone().into_raw_vec();
            let num_features = 2; // Change this to match the number of features in each point

            // Der transformierte Datensatz steht in den Labels, muss aber nochmal in DatasetBase umgewandelt werden
            let labels: Vec<Vec<f64>> = labels.chunks(num_features).map(|chunk| chunk.to_vec()).collect();
            let labels: Array2<f64> = Array2::from_shape_vec((labels.len(), num_features), labels.iter().flat_map(|v| v.iter().cloned()).collect()).unwrap();

            println!("Labels: {:?}\n", labels);
            let dataset = DatasetBase::from(labels);
            let data = dataset.records;
            
            let myVec: Vec<u32> = vec![0, 1, 2, 4, 5];
            let dataset = DatasetBase::new(data, myVec);

            // Hier schicken wir node_data als output an die nächste node bzw. den output
            self.output.send(dataset).map_err(|e| UpdateError::Other(e.into()))?;
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
    
    // Define your data as a 2D array
    let data: Array2<f64> = Array2::from_shape_vec((10, 6), vec![1.1, 2.5, 3.2, 4.6, 5.2, 6.7, 7.8, 8.2, 9.5, 10.3, 11.0, 12.0, 13.0, 14.0, 15.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0])
    .expect("Failed to create data array");

    // Define labels as a vector
    let labels: Vec<u32> = vec![0, 1, 2, 4, 5];

    // Define feature names as a vector of strings
    let feature_names: Vec<&str> = vec!["Feature1", "Feature2", "Feature3", "a", "b", "c"];

    // Create a DatasetBase with data, labels, and feature names
    let dataset = DatasetBase::new(data, labels);
    let dataset = dataset.with_feature_names(feature_names);

    // You can access the data and labels as follows:
    let dataset_data = dataset.records();
    let dataset_labels = dataset.targets();
    println!("Dataset:\n{:?}\n", dataset);
    println!("Data:\n{:?}\n", dataset_data);
    println!("Labels: {:?}\n", dataset_labels);


    let mut my_test_node: DimRedNode<> = DimRedNode::new(Some(&change_observer));

    let mock_output = flowrs::connection::Edge::new();
    flowrs::connection::connect(my_test_node.output.clone(), mock_output.clone());
    my_test_node.input.send(dataset)?;
    my_test_node.on_update()?;

    let actual = mock_output.next()?;

    let expected = "Labels: [[-3.076047733203457, -10.562293260063301],
    [-3.561730416569943, 3.951032231750752],
    [14.63575200500477, 1.1072539713398344],
    [-3.347031741680441, -4.147375003300382],
    [-4.622799446757189, 10.4931265494172],
    [-2.709147889142067, -11.467625779659173],
    [-3.984915594218815, 3.1728757730584096],
    [14.63575200500477, 1.1072539713398344],
    [-3.347031741680441, -4.147375003300382],
    [-4.622799446757189, 10.4931265494172]], shape=[10, 2], strides=[2, 1], layout=Cc (0x5), const ndim=2";

    //let dataset_strings = dataset.records();

 
    Ok(assert!(expected == expected))
}