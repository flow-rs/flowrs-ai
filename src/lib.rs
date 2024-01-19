pub mod nodes;

use wasm_bindgen::prelude::wasm_bindgen;

pub use self::nodes::converter::csv2ndarray;
pub use self::nodes::converter::csv2dataset;
pub use self::nodes::converter::csv2dataset_encoding;
pub use self::nodes::converter::ndarray2dataset;

pub use self::nodes::scaling::standardscaler;
pub use self::nodes::scaling::maxabsscaler;
pub use self::nodes::scaling::minmaxscaler;
pub use self::nodes::scaling::minmaxrangescaler;
pub use self::nodes::scaling::l1normscaler;
pub use self::nodes::scaling::l2normscaler;
pub use self::nodes::scaling::maxnormscaler;

pub use self::nodes::dimension_reduction::pca;
pub use self::nodes::dimension_reduction::tsne;
pub use self::nodes::dimension_reduction::diffusionmap;

pub use self::nodes::clustering::dbscan;
pub use self::nodes::clustering::kmeans;

pub use self::nodes::test_node;

// Required for debug node
#[wasm_bindgen]
extern "C" {
    fn alert(s: &str);
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}