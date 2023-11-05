use flowrs::{node::{Node, UpdateError, ChangeObserver}, connection::{Input, Output}};
use flowrs::RuntimeConnectable;

use ndarray::{Array3, ArrayBase, OwnedRepr, Dim, arr2};
use anyhow::{anyhow};
use linfa::prelude::*;
use ndarray::Array2;
use ndarray::prelude::*;
use linfa::traits::{Fit, Predict};
use linfa_reduction::Pca;
use linfa_clustering::KMeans;

use serde::{Deserialize, Serialize};


// Definition eines Structs
#[derive(RuntimeConnectable, Deserialize, Serialize)]
pub struct Dimred_Cluster_Test_Node { // <--- Wenn man eine neue Node anlegt, einfach alles kopieren und hier den Namen ändern
    #[output]
    pub output: Output<u8>, // <--- Wir haben in diesem Fall eine Output-Variable vom Typ Array2<u8>

    #[input]
    pub input: Input<u8>, // <--- Wir haben in diesem Fall eine Input-Variable vom Typ Array2<u8>

    // Das bedeutet, unsere Node braucht als Input einen Array2<u8> und liefert als Output einen Array2<u8>
}

// Das ist einfach der Konstruktur
impl Dimred_Cluster_Test_Node {
    // Hier will der Konstruktur als einzigen Parameter einen ChangeObserver
    pub fn new(change_observer: Option<&ChangeObserver>) -> Self {
        Self {
            output: Output::new(change_observer),
            input: Input::new()
        }
    }
}

// Hier befinden sich die Methoden von unserer Node. Wir verwenden erstmal nur die Methoden, welche wir implementieren müssen, da diese von "Node" vorgegeben werden.
impl Node for Dimred_Cluster_Test_Node {
    // on_update wird von der Pipeline automatisch getriggert, wenn diese Node einen Input bekommt.
    fn on_update(&mut self) -> Result<(), UpdateError> {

        // Hier überprüfen wir nur, ob ein input da ist und der passt
        if let Ok(node_data) = self.input.next() {
            println!("JW-Debug: Dimred_Cluster_Test_Node has received: {}.", node_data);

            // #############################################################################
            // #############################################################################
            // Here begins the dimred_cluster_code
            // #############################################################################
            // #############################################################################

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
            let labels: Vec<Vec<f64>> = labels.chunks(num_features).map(|chunk| chunk.to_vec()).collect();
            let labels: Array2<f64> = Array2::from_shape_vec((labels.len(), num_features), labels.iter().flat_map(|v| v.iter().cloned()).collect()).unwrap();

            println!("Labels: {:?}\n", labels);
            let dataset = DatasetBase::from(labels);
            // Perform KMeans
            let model = KMeans::params(3)
                .max_n_iterations(200)
                .tolerance(1e-5)
                .fit(&dataset)
                .expect("Error while fitting KMeans to the dataset");

            // Predict cluster assignments
            let result = model.predict(dataset);

            // Output the cluster assignments
            for (point, cluster) in result.records.outer_iter().zip(result.targets.iter()) {
                println!("Point {:?} is in Cluster {}", point, cluster);
            } 

            // #############################################################################
            // #############################################################################
            // Here ends the dimred-cluster code
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
    let mut my_test_node: Dimred_Cluster_Test_Node<> = Dimred_Cluster_Test_Node::new(Some(&change_observer));

    let mock_output = flowrs::connection::Edge::new();
    flowrs::connection::connect(my_test_node.output.clone(), mock_output.clone());
    my_test_node.input.send(test_input)?;
    my_test_node.on_update()?;

    let expected = 1;
    let actual = mock_output.next()?;
    Ok(assert!(expected == actual))
}