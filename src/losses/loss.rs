use na::DMatrix;

pub trait Loss {

    fn cost(&mut self, prediction: &DMatrix<f64>, ground_truth: &DMatrix<f64>) -> f64;

    fn get_derivate(&self) -> DMatrix<f64>;
}