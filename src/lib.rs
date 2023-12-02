pub mod nodes;

use js_sys::Promise;
use wasm_bindgen::JsValue;
use wasm_bindgen::prelude::wasm_bindgen;



pub use self::nodes::clustering;
pub use self::nodes::model;


// Required for debug node
#[wasm_bindgen]
extern "C" {
    fn alert(s: &str);
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}