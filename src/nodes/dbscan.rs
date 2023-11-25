use flowrs::{node::{Node, UpdateError, ChangeObserver}, connection::{Input, Output}};
use flowrs::RuntimeConnectable;

use ndarray::{Array2, Array1};
use linfa::dataset::DatasetBase;
use linfa::traits::Transformer;
use ndarray::{ArrayBase, OwnedRepr, Dim, array};
use linfa_clustering::Dbscan;
use linfa::dataset::Labels;
use serde::{Deserialize, Serialize};


#[derive(RuntimeConnectable, Deserialize, Serialize)]
pub struct DbscanNode {
    #[output]
    pub output: Output<DatasetBase<ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>, ArrayBase<OwnedRepr<Option<usize>>, Dim<[usize; 1]>>>>,

    #[input]
    pub input: Input<DatasetBase<ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>, ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>>>, 
}

impl DbscanNode {
    pub fn new(change_observer: Option<&ChangeObserver>) -> Self {
        Self {
            output: Output::new(change_observer),
            input: Input::new()
        }
    }
}

impl Node for DbscanNode {
    fn on_update(&mut self) -> Result<(), UpdateError> {

        if let Ok(node_data) = self.input.next() {
            println!("JW-Debug: DbscanNode has received: \n Records: {} \n Targets: {}.", node_data.records, node_data.targets);

            // parameter
            let min_points = 2;
            let tolerance = 0.5;

            // dbscan
            let clusters = Dbscan::params(min_points)
                .tolerance(tolerance)
                .transform(node_data)
                .unwrap();
            
            // debug
            println!("Clusters:");
            let label_count = clusters.label_count().remove(0);
            for (label, count) in label_count {
                match label {
                    None => println!(" - {} noise points", count),
                    Some(i) => println!(" - {} points in cluster {}", count, i),
                }
            }

            self.output.send(clusters).map_err(|e| UpdateError::Other(e.into()))?;
        }
        Ok(())
    }
}


#[test]
fn input_output_test() -> Result<(), UpdateError> {
    let change_observer = ChangeObserver::new();
    let record_input: Array2<f64> = array![[1.1, 2.5, 3.2, 4.6, 5.2, 6.7], 
    [7.8, 8.2, 9.5, 10.3, 11.0, 12.0], 
    [13.0, 14.0, 15.0, 1.0, 2.0, 3.0], 
    [4.0, 5.0, 6.0, 7.0, 8.0, 9.0], 
    [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 
    [7.0, 8.0, 9.0, 10.0, 11.0, 12.0], 
    [13.0, 14.0, 15.0, 1.0, 2.0, 3.0], 
    [4.0, 5.0, 6.0, 7.0, 8.0, 9.0], 
    [10.0, 11.0, 12.0, 13.0, 14.0, 15.0]];
    let target_input: Array2<f64> = array![[-3.076047733203457, -10.562293260063301],
    [-3.561730416569943, 3.951032231750752],
    [14.63575200500477, 1.1072539713398344], 
    [-3.347031741680441, -4.147375003300382],
    [-4.622799446757189, 10.4931265494172],
    [-2.709147889142067, -11.467625779659173],
    [-3.984915594218815, 3.1728757730584096],
    [14.63575200500477, 1.1072539713398344],
    [-3.347031741680441, -4.147375003300382],
    [-4.622799446757189, 10.4931265494172]];
    let input_data: DatasetBase<ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>, ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>> = DatasetBase::new(record_input.clone(), target_input.clone());


    let mut and: DbscanNode<> = DbscanNode::new(Some(&change_observer));
    let mock_output = flowrs::connection::Edge::new();
    flowrs::connection::connect(and.output.clone(), mock_output.clone());
    and.input.send(input_data)?;
    and.on_update()?;

    let expected: Array1<Option<usize>> = array![None, None, Some(0), Some(1), Some(2), None, None, Some(0), Some(1), Some(2)];
    let actual: DatasetBase<Array2<f64>, Array1<Option<usize>>> = mock_output.next()?;

    Ok(assert!(expected == actual.targets))
}