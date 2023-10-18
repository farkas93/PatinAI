use na::DMatrix;
use super::loss::Loss;
struct CrossEntropyLoss {}

impl Loss for CrossEntropyLoss {

    fn cost(&mut self, prediction: &DMatrix<f64>, ground_truth: &DMatrix<f64>) -> f64{
        todo![];
    }

    fn get_derivate(&self) -> DMatrix<f64>{
        todo![];
    }
}