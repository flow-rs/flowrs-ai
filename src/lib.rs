//! # Flowrs AI
//!
//! `flowrs-ai` is a Rust library for building data processing pipelines using runtime connectable nodes, focusing on data preprocessing, dimension reduction, scaling, and clustering for machine learning and data analysis tasks.
//!
//! ## Modules
//!
//! - [nodes](nodes/index.html): Contains various submodules for different types of nodes, including converters, scalers, dimension reduction, and clustering.
//!
//! ## Converter Nodes
//!
//! Converter nodes are responsible for converting data between different formats or representations. These nodes are used to prepare and preprocess data before applying other operations.
//!
//! - [csv2dataset](nodes/converter/csv2dataset/index.html): Contains a node for converting CSV data to a dataset format.
//! - [csv2dataset_encoding](nodes/converter/csv2dataset_encoding/index.html): Contains a node for encoding CSV data into a dataset format with specified encoding configurations.
//! - [csv2ndarray](nodes/converter/csv2ndarray/index.html): Contains a node for converting CSV data to an ndarray format.
//! - [ndarray2dataset](nodes/converter/ndarray2dataset/index.html): Contains a node for converting ndarray data to a dataset format.
//!
//! ## Scaling Nodes
//!
//! Scaling nodes are responsible for scaling data to a desired range or distribution. These nodes are used to normalize or standardize data before further analysis.
//!
//! - [l1normscaler](nodes/scaling/l1normscaler/index.html): Contains a node for performing L1-norm (Manhattan) scaling on input data.
//! - [l2normscaler](nodes/scaling/l2normscaler/index.html): Contains a node for performing L2-norm (Euclidean) scaling on input data.
//! - [maxabsscaler](nodes/scaling/maxabsscaler/index.html): Contains a node for scaling input data to have maximum absolute values within a specified range.
//! - [maxnormscaler](nodes/scaling/maxnormscaler/index.html): Contains a node for scaling input data to have a maximum L2-norm (Euclidean norm) within a specified range.
//! - [minmaxrangescaler](nodes/scaling/minmaxrangescaler/index.html): Contains a node for scaling input data to a specified minimum and maximum range.
//! - [minmaxscaler](nodes/scaling/minmaxscaler/index.html): Contains a node for performing min-max scaling on input data.
//! - [standardscaler](nodes/scaling/standardscaler/index.html): Contains a node for performing standardization (z-score) scaling on input data.
//!
//! ## Dimension Reduction Nodes
//!
//! Dimension reduction nodes are responsible for reducing the number of features or dimensions in the data while preserving essential information.
//!
//! - [diffusionmap](nodes/dimension_reduction/diffusionmap/index.html): Contains a node for applying the diffusion map dimension reduction technique.
//! - [pca](nodes/dimension_reduction/pca/index.html): Contains a node for performing Principal Component Analysis (PCA) dimension reduction.
//! - [tsne](nodes/dimension_reduction/tsne/index.html): Contains a node for applying the t-Distributed Stochastic Neighbor Embedding (t-SNE) dimension reduction technique.
//!
//! ## Clustering Nodes
//!
//! Clustering nodes are responsible for grouping data points into clusters based on similarity or other criteria.
//!
//! - [dbscan](nodes/clustering/dbscan/index.html): Contains a node for applying the Density-Based Spatial Clustering of Applications with Noise (DBSCAN) clustering algorithm.
//! - [kmeans](nodes/clustering/kmeans/index.html): Contains a node for applying the K-Means clustering algorithm.
//!
//! ## Usage
//!
//! You can use the provided nodes to build data processing pipelines by connecting them together. Each node represents a specific data processing step, and you can create complex workflows by combining multiple nodes to achieve your data analysis and machine learning goals.
//!
//!
//! ## Example
//!
//! See integration tests.

pub mod nodes;

pub use self::nodes::converter::csv2dataset;
pub use self::nodes::converter::csv2dataset_encoding;
pub use self::nodes::converter::csv2ndarray;
pub use self::nodes::converter::ndarray2dataset;

pub use self::nodes::scaling::l1normscaler;
pub use self::nodes::scaling::l2normscaler;
pub use self::nodes::scaling::maxabsscaler;
pub use self::nodes::scaling::maxnormscaler;
pub use self::nodes::scaling::minmaxrangescaler;
pub use self::nodes::scaling::minmaxscaler;
pub use self::nodes::scaling::standardscaler;

pub use self::nodes::dimension_reduction::diffusionmap;
pub use self::nodes::dimension_reduction::pca;
pub use self::nodes::dimension_reduction::tsne;

pub use self::nodes::clustering::dbscan;
pub use self::nodes::clustering::kmeans;
