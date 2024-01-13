use flowrs::RuntimeConnectable;
use flowrs::{
    connection::{Input, Output},
    node::{ChangeObserver, Node, UpdateError},
};

use linfa::{
    traits::{Fit, Transformer},
    DatasetBase, Float,
};
use linfa_preprocessing::linear_scaling::LinearScaler;
use log::debug;
use ndarray::{array, Array1, Array2};
use serde::{Deserialize, Serialize};

/// A runtime connectable node that applies Standard Scaling (Z-score normalization) to input data.
///
/// The `StandardScalerNode` struct is designed to be used as a connectable node within a flowrs-flow.
/// It receives input data and applies Standard Scaling (Z-score normalization) to it,
/// then sends the scaled data to its output port.
///

#[derive(RuntimeConnectable, Deserialize, Serialize)]
pub struct StandardScalerNode<T>
where
    T: Clone,
{
    /// The output port for sending the scaled data.
    #[output]
    pub output: Output<DatasetBase<Array2<T>, Array1<()>>>,

    /// The input port for receiving data to be scaled.
    #[input]
    pub data_input: Input<DatasetBase<Array2<T>, Array1<()>>>,
}

impl<T> StandardScalerNode<T>
where
    T: Clone,
{
    /// Create a new instance of `StandardScalerNode`.
    ///
    /// # Parameters
    ///
    /// - `change_observer`: An optional reference to a `ChangeObserver` for tracking changes.
    ///
    /// # Returns
    ///
    /// A new instance of `StandardScalerNode`.
    pub fn new(change_observer: Option<&ChangeObserver>) -> Self {
        Self {
            output: Output::new(change_observer),
            data_input: Input::new(),
        }
    }
}

impl<T> Node for StandardScalerNode<T>
where
    T: Clone + Send + Float,
{
    /// Process and scale input data using Standard Scaling (Z-score normalization).
    ///
    /// This method is called when the node is updated. It receives input data, applies Standard
    /// Scaling (Z-score normalization) to it, and sends the scaled data to the output port.
    ///
    /// # Returns
    ///
    /// - `Ok(())`: If the update is successful.
    /// - `Err(UpdateError)`: If an error occurs during the update.
    fn on_update(&mut self) -> Result<(), UpdateError> {
        // Receiving data
        if let Ok(data) = self.data_input.next() {
            debug!("StandardScalerNode has received an update!");

            // Apply Standard Scaling (Z-score normalization) to input data
            let scaler = LinearScaler::standard().fit(&data).unwrap();
            let scaled_data = scaler.transform(data);

            // Send the scaled data to the output port
            self.output
                .send(scaled_data)
                .map_err(|e| UpdateError::Other(e.into()))?;
            debug!("StandardScalerNode has sent an output!");
        }
        Ok(())
    }
}

#[test]
fn input_output_test() -> Result<(), UpdateError> {
    let change_observer = ChangeObserver::new();
    let test_input: Array2<f64> = array![
        [1.1, 2.5, 3.2, 4.6, 5.2, 6.7],
        [7.8, 8.2, 9.5, 10.3, 11.0, 12.0],
        [13.0, 14.0, 15.0, 1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        [13.0, 14.0, 15.0, 1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        [10.0, 11.0, 12.0, 13.0, 14.0, 15.0]
    ];
    let dataset = DatasetBase::from(test_input.clone());

    let mut test_node: StandardScalerNode<f64> = StandardScalerNode::new(Some(&change_observer));
    let mock_output = flowrs::connection::Edge::new();
    flowrs::connection::connect(test_node.output.clone(), mock_output.clone());
    test_node.data_input.send(dataset)?;
    test_node.on_update()?;

    let expected_data: Array2<f64> = array![
        [
            -1.4143377359247322,
            -1.3343815168527249,
            -1.3919452566240385,
            -0.5892455532801077,
            -0.6668399873113844,
            -0.5645401987746118
        ],
        [
            0.16764270325652061,
            0.031143554253294955,
            0.1019653254426467,
            0.7596298096502591,
            0.7046748802084846,
            0.6979336634639715
        ],
        [
            1.3954484172479409,
            1.4206252055541575,
            1.406172976453245,
            -1.4411668351308655,
            -1.4235378452533811,
            -1.4458898761864531
        ],
        [
            -0.7295999338910557,
            -0.7354670119816634,
            -0.727984997927734,
            -0.021298032046269122,
            -0.004729361612137379,
            -0.01667418308617004
        ],
        [
            0.6870989668682753,
            0.7019277997088839,
            0.6947869849929187,
            1.3985707710383273,
            1.4140791220291065,
            1.412541510014113
        ],
        [
            -1.4379493842707212,
            -1.454164417826937,
            -1.4393709893880604,
            -0.7312324335885673,
            -0.7141336034327592,
            -0.7312820296363116
        ],
        [
            -0.021250483511390143,
            -0.016769606136389788,
            -0.01659900646740768,
            0.688636369496029,
            0.7046748802084846,
            0.6979336634639715
        ],
        [
            1.3954484172479409,
            1.4206252055541575,
            1.406172976453245,
            -1.4411668351308655,
            -1.4235378452533811,
            -1.4458898761864531
        ],
        [
            -0.7295999338910557,
            -0.7354670119816634,
            -0.727984997927734,
            -0.021298032046269122,
            -0.004729361612137379,
            -0.01667418308617004
        ],
        [
            0.6870989668682753,
            0.7019277997088839,
            0.6947869849929187,
            1.3985707710383273,
            1.4140791220291065,
            1.412541510014113
        ]
    ];
    let actual = mock_output.next()?;
    let expected = DatasetBase::new(expected_data.clone(), expected_data.clone());

    Ok(assert!(expected.records == actual.records))
}

#[test]
fn test_f32() -> Result<(), UpdateError> {
    let change_observer = ChangeObserver::new();
    let mut node: StandardScalerNode<f32> = StandardScalerNode::new(Some(&change_observer));
    let mock_output = flowrs::connection::Edge::new();
    flowrs::connection::connect(node.output.clone(), mock_output.clone());

    let test_data = array![
        [1.0, 2.0, 3.0, 4.0],
        [3.0, 4.0, 5.0, 6.0],
        [5.0, 6.0, 7.0, 8.0],
        [7.0, 4.0, 1.0, 9.0]
    ];
    let test_data_input = DatasetBase::from(test_data);

    node.data_input.send(test_data_input)?;
    node.on_update()?;

    let expected = array![
        [-1.3416407, -1.4142135, -0.4472136, -1.432078],
        [-0.4472136, 0., 0.4472136, -0.39056674],
        [0.4472136, 1.4142135, 1.3416407, 0.6509446],
        [1.3416407, 0., -1.3416407, 1.1717002]
    ];

    let actual = mock_output.next()?.records;

    Ok(assert!(expected == actual))
}

#[test]
fn test_f64() -> Result<(), UpdateError> {
    let change_observer = ChangeObserver::new();
    let mut node: StandardScalerNode<f64> = StandardScalerNode::new(Some(&change_observer));
    let mock_output = flowrs::connection::Edge::new();
    flowrs::connection::connect(node.output.clone(), mock_output.clone());

    let test_data = array![
        [1.0, 2.0, 3.0, 4.0],
        [3.0, 4.0, 5.0, 6.0],
        [5.0, 6.0, 7.0, 8.0],
        [7.0, 4.0, 1.0, 9.0]
    ];
    let test_data_input = DatasetBase::from(test_data);

    node.data_input.send(test_data_input)?;
    node.on_update()?;

    let expected = array![
        [
            -1.3416407864998738,
            -1.414213562373095,
            -0.4472135954999579,
            -1.4320780207890627
        ],
        [
            -0.4472135954999579,
            0.,
            0.4472135954999579,
            -0.3905667329424717
        ],
        [
            0.4472135954999579,
            1.414213562373095,
            1.3416407864998738,
            0.6509445549041194
        ],
        [
            1.3416407864998738,
            0.,
            -1.3416407864998738,
            1.171700198827415
        ]
    ];

    let actual = mock_output.next()?.records;

    Ok(assert!(expected == actual))
}
