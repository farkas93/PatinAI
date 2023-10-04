
use nalgebra::DMatrix;
use crate::optimizers::optimizer::Optimizer;
use crate::layers::layer::Layer;
use super::backprop_cache::BackpropCache;

pub struct GradientDescent{
    num_iters: usize,
    learning_rate: f64,
    cost_derivate: DMatrix<f64>
}

impl Optimizer for GradientDescent{
    
    fn loss(&mut self, model_result: &DMatrix<f64>, ground_truth: &DMatrix<f64>) -> f64{
        let batch_size = ground_truth.ncols() as f64;
        let log_res = model_result.transpose().map(|x| x.ln());
        let counter_res = model_result.transpose().map(|x| (1.0 - x));
        let counter_log_res = counter_res.transpose().map(|x| x.ln());
        let counter_gt = ground_truth.map(|x| (1.0 - x));
        let logprobs = ground_truth.dot(&log_res) + counter_gt.dot(&counter_log_res);
        let cost = -logprobs / batch_size;        

        self.cost_derivate = - (ground_truth.component_div(model_result) - counter_gt.component_div(&counter_res));
        return cost
    }

    fn optimize(&self, layers: &mut Vec<Box<dyn Layer>>){
        let d_z = self.cost_derivate.clone();
        let mut cache = BackpropCache::new(d_z, DMatrix::zeros(1, 1));
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

impl GradientDescent {
    
    pub fn new(iterations: usize, learning_rate: f64) -> GradientDescent {
        return GradientDescent {
            num_iters: iterations,
            learning_rate: learning_rate,
            cost_derivate: DMatrix::zeros(1,1),
        }
    }
}