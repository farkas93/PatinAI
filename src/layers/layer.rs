use na::DMatrix;
use crate::optimizers::backprop_cache::BackpropCache;

pub trait Layer {

    fn forward(&mut self, x: &DMatrix<f64>) -> &DMatrix<f64>;
    fn backward(&mut self, cache: &mut BackpropCache);

    fn update(&mut self, _learning_rate: f64) {
        // Activation functions do not have anything to update
        return;
    }
}