
use na::DMatrix;
use crate::{optimizers::optimizer::Optimizer, losses::loss::Loss};
use crate::layers::layer::Layer;
use super::backprop_cache::{BackpropCache, self};

pub struct GradientMomentum{
    num_iters: usize,
    loss: Box<dyn Loss>,
    cache: BackpropCache,
}

impl Optimizer for GradientMomentum{
    
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

impl GradientMomentum {
    
    pub fn new(iterations: usize, learning_rate: f64, loss: Box<dyn Loss>) -> GradientMomentum {
        return GradientMomentum {
            num_iters: iterations,
            loss: loss,
            cache: BackpropCache::new(learning_rate, Self::update_function)
        }
    }

    pub fn update_function(bpc: &mut BackpropCache, derivate :&DMatrix<f64>, index: usize) -> DMatrix<f64> {
        // v__d = beta1 * v_d + (1-beta1)*derivate
        bpc.set_v((bpc.beta_1 * bpc.get_v(index) + (1.0-bpc.beta_1) * derivate), index);
        // return lr * v_d/(1-beta^t)
        let bias_correction = 1.0/(1.0-bpc.beta_1.powi(bpc.curr_iter as i32));
        bpc.learning_rate*bpc.get_v(index) * bias_correction
    }
    
    pub fn init_additional_cache(bpc: &mut BackpropCache, nrows: usize, ncols: usize) -> usize {
        bpc.append_to_vec_v(DMatrix::zeros(nrows, ncols))
    }
}