
use nalgebra::DMatrix;
use crate::optimizers::optimizer::Optimizer;
use crate::layers::layer::Layer;
use super::backprop_cache::BackpropCache;

pub struct RMSProp{
    num_iters: usize,
    learning_rate: f64,
    cost_derivate: DMatrix<f64>
}

impl Optimizer for RMSProp{
    
    fn loss(&mut self, model_result: &DMatrix<f64>, ground_truth: &DMatrix<f64>) -> f64{
        let batch_size = ground_truth.ncols() as f64;
        let log_res = model_result.map(|x| x.ln()); // Log(A)
        let counter_res = model_result.map(|x| (1.0 - x)); 
        let counter_log_res = counter_res.map(|x| x.ln()); // Log(1 - A)
        let counter_gt = ground_truth.map(|x| (1.0 - x)); // 1 - Y
        let logprobs = ground_truth.dot(&log_res) + counter_gt.dot(&counter_log_res);
        let cost = - logprobs / batch_size;

        self.cost_derivate = - (ground_truth.component_div(model_result) - counter_gt.component_div(&counter_res));
        return cost
    }

    fn optimize(&self, layers: &mut Vec<Box<dyn Layer>>){
        let d_a = self.cost_derivate.clone();
        let mut cache = BackpropCache::new(DMatrix::zeros(1, 1), d_a);
        for layer in layers.iter_mut().rev() {
            layer.backward(&mut cache);
            layer.update(self.learning_rate);
        }
    }

    fn get_num_iters(&self) -> usize {
        self.num_iters
    }

    fn get_lr(&self) -> f64 {
        self.learning_rate
    }

}

impl RMSProp {
    
    pub fn new(iterations: usize, learning_rate: f64) -> RMSProp {
        return RMSProp {
            num_iters: iterations,
            learning_rate: learning_rate,
            cost_derivate: DMatrix::zeros(1,1),
        }
    }
}