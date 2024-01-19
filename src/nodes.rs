//! # Nodes
//!
//! The `nodes` module contains various submodules for different types of nodes used in building data processing pipelines. These nodes can be connected together to create complex data analysis and machine learning workflows.
//!
//! ## Submodules
//!
//! - [clustering](clustering/index.html): Contains nodes related to clustering algorithms for grouping data points into clusters.
//! - [converter](converter/index.html): Contains nodes for converting data between different formats or representations.
//! - [dimension_reduction](dimension_reduction/index.html): Contains nodes for reducing the dimensionality of data while preserving essential information.
//! - [scaling](scaling/index.html): Contains nodes for scaling data to a desired range or distribution.
//!
//! ## Usage
//!
//! To use these nodes, you can import them from their respective submodules and create instances of the nodes to build your data processing pipelines. Each submodule provides nodes with specific functionality, allowing you to choose the nodes that best suit your data preprocessing and analysis needs.
//!
//! Explore the submodules to find the nodes that best suit your data preprocessing and analysis tasks.

pub mod clustering;
pub mod converter;
pub mod dimension_reduction;
pub mod scaling;
