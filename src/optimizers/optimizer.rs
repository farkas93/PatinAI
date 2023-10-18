
use crate::layers::layer::Layer;
use na::DMatrix;

pub trait Optimizer {
    fn init(&mut self, layers: &mut Vec<Box<dyn Layer>>);
    fn loss(&mut self, model_result: &DMatrix<f64>, ground_truth: &DMatrix<f64>) -> f64;
    fn optimize(&mut self, layers: &mut Vec<Box<dyn Layer>>);
    fn get_lr(&self) -> f64;
    fn get_num_iters(&self) -> usize;
}