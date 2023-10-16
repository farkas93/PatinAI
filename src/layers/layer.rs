use na::DMatrix;
use crate::optimizers::backprop_cache::BackpropCache;

pub trait Layer {

    fn forward(&mut self, x: DMatrix<f64>) -> DMatrix<f64>;
    fn backward(&mut self, cache: &mut BackpropCache);


    fn set_training_mode(&mut self, on: bool) {
        // Most layers don't need a training mode.
    }

    fn register_backprop_index(&mut self, bpc: &mut BackpropCache, init_fn: fn(&mut BackpropCache, usize, usize) -> usize) {
        // Some layers, for instance the activation layers, don't need to 
        // register themselves
    }
}