//! # Dimension Reduction Nodes
//!
//! This module contains Rust structs for runtime connectable nodes related to dimension reduction techniques.
//!
//! ## Modules
//!
//! - [diffusionmap](diffusionmap/index.html): Contains a node for performing diffusion map dimension reduction.
//! - [pca](pca/index.html): Contains a node for performing Principal Component Analysis (PCA) dimension reduction.
//! - [tsne](tsne/index.html): Contains a node for performing t-Distributed Stochastic Neighbor Embedding (t-SNE) dimension reduction.
//!
//! ## Usage
//!
//! You can use these nodes to build flowrs-flows for dimension reduction tasks. Each node is designed to be connected
//! to other nodes within a flowrs-flow, allowing you to perform dimension reduction on input data and send the resulting
//! reduced datasets to other parts of your application.

pub mod diffusionmap;
pub mod pca;
pub mod tsne;
