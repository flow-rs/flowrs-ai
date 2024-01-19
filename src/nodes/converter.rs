//! # Converter Nodes
//!
//! This module contains Rust structs for runtime connectable nodes related to data conversion.
//!
//! ## Modules
//!
//! - [csv2dataset](csv2dataset/index.html): Contains a node for converting CSV data into a dataset.
//! - [csv2dataset_encoding](csv2dataset_encoding/index.html): Contains a node for encoding CSV data and converting it into a dataset.
//! - [csv2ndarray](csv2ndarray/index.html): Contains a node for converting CSV data into an ndarray.
//! - [ndarray2dataset](ndarray2dataset/index.html): Contains a node for converting an ndarray into a dataset.
//!
//! ## Usage
//!
//! You can use these nodes to build flowrs-flows for data conversion tasks. Each node is designed to be connected
//! to other nodes within a flowrs-flow, allowing you to perform various data conversion operations on input data
//! and send the resulting datasets or ndarrays to other parts of your application.

pub mod csv2dataset;
pub mod csv2dataset_encoding;
pub mod csv2ndarray;
pub mod ndarray2dataset;
