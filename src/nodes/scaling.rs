//! # Scaling Nodes
//!
//! This module contains Rust structs for runtime connectable nodes related to data scaling techniques.
//!
//! ## Modules
//!
//! - [l1normscaler](l1normscaler/index.html): Contains a node for performing L1-norm (Manhattan) scaling on input data.
//! - [l2normscaler](l2normscaler/index.html): Contains a node for performing L2-norm (Euclidean) scaling on input data.
//! - [maxabsscaler](maxabsscaler/index.html): Contains a node for scaling input data to have maximum absolute values within a specified range.
//! - [maxnormscaler](maxnormscaler/index.html): Contains a node for scaling input data to have a maximum L2-norm (Euclidean norm) within a specified range.
//! - [minmaxrangescaler](minmaxrangescaler/index.html): Contains a node for scaling input data to a specified minimum and maximum range.
//! - [minmaxscaler](minmaxscaler/index.html): Contains a node for performing min-max scaling on input data.
//! - [standardscaler](standardscaler/index.html): Contains a node for performing standardization (z-score) scaling on input data.
//!
//! ## Usage
//!
//! You can use these nodes to build flowrs-flows for data scaling tasks. Each node is designed to be connected
//! to other nodes within a flowrs-flow, allowing you to apply various scaling techniques to input data and send
//! the scaled datasets to other parts of your application.

pub mod l1normscaler;
pub mod l2normscaler;
pub mod maxabsscaler;
pub mod maxnormscaler;
pub mod minmaxrangescaler;
pub mod minmaxscaler;
pub mod standardscaler;
