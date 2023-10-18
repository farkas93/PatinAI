
use na::DMatrix;
use crate::losses::loss::Loss;
use crate::optimizers::optimizer::Optimizer;
use crate::layers::layer::Layer;
use super::backprop_cache::BackpropCache;

pub struct RMSProp{
    num_iters: usize,
    loss: Box<dyn Loss>,
    cache: BackpropCache
}

impl Optimizer for RMSProp{
    
    fn init(&mut self, layers: &mut Vec<Box<dyn Layer>>) {
        for layer in layers.iter_mut().rev() {
            layer.register_backprop_index(&mut self.cache, Self::init_additional_cache);
        }
    }

    fn compute_loss(&mut self, prediction: &DMatrix<f64>, ground_truth: &DMatrix<f64>) -> f64{
        self.loss.cost(prediction, ground_truth)
    }

    fn optimize(&mut self, layers: &mut Vec<Box<dyn Layer>>){
        self.cache.curr_iter = self.cache.curr_iter + 1;
        self.cache.d_a = self.loss.get_derivate();
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
    
    pub fn new(iterations: usize, learning_rate: f64, loss: Box<dyn Loss>) -> RMSProp {
        return RMSProp {
            num_iters: iterations,
            loss: loss,
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