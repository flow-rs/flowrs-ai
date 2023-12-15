pub mod nodes;

use wasm_bindgen::prelude::wasm_bindgen;

pub use self::nodes::clustering;
pub use self::nodes::model;
pub use self::nodes::max_output_value;


// Required for debug node
#[wasm_bindgen]
extern "C" {
    fn alert(s: &str);
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}