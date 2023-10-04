use na::DMatrix;
use crate::optimizers::optimizer::Optimizer;

pub trait Model {

    fn set_optimizer(&mut self, o: Box<dyn Optimizer>);
    fn train(&mut self);
    fn predict(&mut self, x: &DMatrix<f64>) -> DMatrix<f64>;
}