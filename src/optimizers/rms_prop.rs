
use nalgebra::DMatrix;
use crate::optimizers::optimizer::Optimizer;
use crate::layers::layer::Layer;
use super::backprop_cache::BackpropCache;

pub struct RMSProp{
    num_iters: usize,
    cost_derivate: DMatrix<f64>,
    cache: BackpropCache
}

impl Optimizer for RMSProp{
    
    fn init(&mut self, layers: &mut Vec<Box<dyn Layer>>) {
        for layer in layers.iter_mut().rev() {
            layer.register_backprop_index(&mut self.cache, Self::init_additional_cache);
        }
    }

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

    fn optimize(&mut self, layers: &mut Vec<Box<dyn Layer>>){
        self.cache.curr_iter = self.cache.curr_iter + 1;
        self.cache.d_a = self.cost_derivate.clone();
        for layer in layers.iter_mut().rev() {
            //TODO: merge backward and update and pass the update function of the Optimizer to backward.
            layer.backward(&mut self.cache);
        }
    }

    fn get_num_iters(&self) -> usize {
        self.num_iters
    }

    fn get_lr(&self) -> f64 {
        self.cache.learning_rate
    }

}

impl RMSProp {
    
    pub fn new(iterations: usize, learning_rate: f64) -> RMSProp {
        return RMSProp {
            num_iters: iterations,
            cost_derivate: DMatrix::zeros(1,1),
            cache: BackpropCache::new(learning_rate, Self::update_function)
        }
    }

    pub fn update_function(bpc: &mut BackpropCache, derivate :&DMatrix<f64>, index: usize) -> DMatrix<f64> {
        // s = beta2 * s + (1-beta2)*derivate^2
        bpc.set_s((bpc.beta_2 * bpc.get_s(index) + (1.0-bpc.beta_2) * derivate.component_mul(&derivate)), index);
        // return lr * derivate / (sqrt(d_s/(1-beta^t)) + epsilon)
        let bias_correction = 1.0/(1.0-bpc.beta_2.powi(bpc.curr_iter as i32));
        bpc.learning_rate * derivate.component_div(&bpc.get_s(index).map(|x| (x*bias_correction).sqrt() + bpc.epsilon))
    }
    
    pub fn init_additional_cache(bpc: &mut BackpropCache, nrows: usize, ncols: usize) -> usize {
        bpc.append_to_vec_s(DMatrix::zeros(nrows, ncols))
    }
}