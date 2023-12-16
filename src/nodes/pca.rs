use std::arch::x86_64::_MM_EXCEPT_DENORM;

use flowrs::{node::{Node, UpdateError, ChangeObserver}, connection::{Input, Output}};
use flowrs::RuntimeConnectable;

use ndarray::{Array2, Array1, array, ArrayBase, Dim, OwnedRepr};
use linfa::{Dataset, DatasetBase};
use linfa_reduction::Pca;
use linfa::traits::{Fit, Predict};
use serde::{Deserialize, Serialize};


#[derive(Clone, Deserialize, Serialize)]
pub struct PCAConfig {
   pub embedding_size: usize
}


impl PCAConfig {
    pub fn new(embedding_size: usize) -> Self {
        PCAConfig {
            embedding_size
        }
    }
}


#[derive(RuntimeConnectable, Deserialize, Serialize)]
pub struct PCANode<T> 
where
    T: Clone
{
    #[input]
    pub config_input: Input<PCAConfig>,

    #[output]
    pub output: Output<DatasetBase<Array2<T>, Array1<()>>>,

    #[input]
    pub data_input: Input<DatasetBase<Array2<T>, Array1<()>>>,

    config: PCAConfig
}


impl<T> PCANode<T> 
where
    T: Clone
{
    pub fn new(change_observer: Option<&ChangeObserver>) -> Self {
        Self {
            config_input: Input::new(),
            data_input: Input::new(),
            output: Output::new(change_observer),
            config: PCAConfig::new(2)

        }
    }
}


impl Node for PCANode<f64> {
    fn on_update(&mut self) -> Result<(), UpdateError> {
        println!("JW-Debug: PCANode has received an update!");

        // Neue Config kommt an
        if let Ok(config) = self.config_input.next() {
            println!("JW-Debug: PCANode has received config: {}", config.embedding_size);

            self.config = config;
        }

        // Daten kommen an
        if let Ok(data) = self.data_input.next() {
            println!("JW-Debug: PCANode has received data!");

            let embedding = Pca::params(self.config.embedding_size)
                .fit(&data)
                .unwrap();
            let red_dataset = embedding.predict(data);
            
            let myoutput= DatasetBase::from(red_dataset.targets.clone());

            self.output.send(myoutput).map_err(|e| UpdateError::Other(e.into()))?;
        }
        Ok(())
    }
}


impl Node for PCANode<f32> {
    fn on_update(&mut self) -> Result<(), UpdateError> {
        println!("JW-Debug: PCANode has received an update!");

        // Neue Config kommt an
        if let Ok(config) = self.config_input.next() {
            println!("JW-Debug: PCANode has received config: {}", config.embedding_size);

            self.config = config;
        }

        // Daten kommen an
        if let Ok(data) = self.data_input.next() {
            println!("JW-Debug: PCANode has received data!");

            let data_f64 = DatasetBase::from(data.records.mapv(|x| x as f64));

            let embedding = Pca::params(self.config.embedding_size)
                .fit(&data_f64)
                .unwrap();
            let red_dataset = embedding.predict(data_f64);
            
            let myoutput= DatasetBase::from(red_dataset.targets.mapv(|x| x as f32));

            self.output.send(myoutput).map_err(|e| UpdateError::Other(e.into()))?;
        }
        Ok(())
    }
}


#[test]
fn input_output_test() -> Result<(), UpdateError> {
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
    let dataset = Dataset::from(test_input.clone());
    let test_config_input = PCAConfig{
        embedding_size: 2,
    };
    let mut and: PCANode<f64> = PCANode::new(Some(&change_observer));
    let mock_output = flowrs::connection::Edge::new();
    flowrs::connection::connect(and.output.clone(), mock_output.clone());
    and.data_input.send(dataset)?;
    and.config_input.send(test_config_input)?;
    and.on_update()?;

    let expected: Array2<f64> = array![[-3.076047733203457, -10.562293260063301],
                                       [-3.561730416569943, 3.951032231750752],
                                       [14.63575200500477, 1.1072539713398344], 
                                       [-3.347031741680441, -4.147375003300382],
                                       [-4.622799446757189, 10.4931265494172],
                                       [-2.709147889142067, -11.467625779659173],
                                       [-3.984915594218815, 3.1728757730584096],
                                       [14.63575200500477, 1.1072539713398344],
                                       [-3.347031741680441, -4.147375003300382],
                                       [-4.622799446757189, 10.4931265494172]];

    let actual = mock_output.next()?;

    Ok(assert!(expected == actual.records))
}


#[test]
fn test_f32() -> Result<(), UpdateError> {
    let change_observer = ChangeObserver::new();
    let mut node: PCANode<f32> = PCANode::new(Some(&change_observer));
    let mock_output = flowrs::connection::Edge::new();
    flowrs::connection::connect(node.output.clone(), mock_output.clone());

    let test_data = array![[1.0, 2.0, 3.0, 4.0],
    [3.0, 4.0, 5.0, 6.0],
    [5.0, 6.0, 7.0, 8.0],
    [7.0, 4.0, 1.0, 9.0]];
    let test_data_input = DatasetBase::from(test_data);

    node.data_input.send(test_data_input)?;
    node.on_update()?;

    let expected = array![[-4.457307, -1.3018891],
    [-1.2156291, 1.0415113],
    [2.0260487, 3.3849118],
    [3.6468875, -3.124534]];

    let actual = mock_output.next()?.records;
    println!("{}", actual);
    Ok(assert!(expected == actual))
}


#[test]
fn test_f64() -> Result<(), UpdateError> {
    let change_observer: ChangeObserver = ChangeObserver::new();
    let mut node: PCANode<f64> = PCANode::new(Some(&change_observer));
    let mock_output = flowrs::connection::Edge::new();
    flowrs::connection::connect(node.output.clone(), mock_output.clone());

    let test_data = array![[1.0, 2.0, 3.0, 4.0],
    [3.0, 4.0, 5.0, 6.0],
    [5.0, 6.0, 7.0, 8.0],
    [7.0, 4.0, 1.0, 9.0]];
    let test_data_input = DatasetBase::from(test_data);

    node.data_input.send(test_data_input)?;
    node.on_update()?;

    let expected = array![[-4.457306893827564, -1.3018891098082346],
    [-1.215629152862062, 1.0415112878465922],
    [2.02604858810344, 3.384911685501419],
    [3.6468874585861863, -3.1245338635397766]];

    let actual = mock_output.next()?.records;
    
    Ok(assert!(expected == actual))
}