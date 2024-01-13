//! # Clustering Nodes
//!
//! This module contains Rust structs for runtime connectable nodes related to clustering algorithms.
//!
//! ## Modules
//!
//! - [dbscan](dbscan/index.html): Contains a DBSCAN (Density-Based Spatial Clustering of Applications with Noise) clustering node.
//! - [kmeans](kmeans/index.html): Contains a K-Means clustering node.
//!
//! ## Usage
//!
//! You can use these nodes to build flowrs-flows for clustering tasks. Each node is designed to be connected
//! to other nodes within a flowrs-flow, allowing you to perform clustering operations on input data and send
//! the resulting clustered datasets to other parts of your application.

pub mod dbscan;
pub mod kmeans;
