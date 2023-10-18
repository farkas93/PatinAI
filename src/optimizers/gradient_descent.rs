
use na::DMatrix;
use crate::losses::loss::Loss;
use crate::optimizers::optimizer::Optimizer;
use crate::layers::layer::Layer;
use super::backprop_cache::BackpropCache;

pub struct GradientDescent{
    num_iters: usize,
    loss: Box<dyn Loss>,
    cache: BackpropCache
}

impl Optimizer for GradientDescent{

    fn init(&mut self, layers: &mut Vec<Box<dyn Layer>>) {
        for layer in layers.iter_mut().rev() {
            layer.register_backprop_index(&mut self.cache, Self::dummy_init);
        }
    }

    fn compute_loss(&mut self, prediction: &DMatrix<f64>, ground_truth: &DMatrix<f64>) -> f64{
        self.loss.cost(prediction, ground_truth)
    }

    fn optimize(&mut self, layers: &mut Vec<Box<dyn Layer>>){
        self.cache.d_a = self.loss.get_derivate();
        for layer in layers.iter_mut().rev() {
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

impl GradientDescent {
    
    pub fn new(iterations: usize, learning_rate: f64, loss: Box<dyn Loss>) -> GradientDescent {
        return GradientDescent {
            num_iters: iterations,
            loss: loss,
            cache: BackpropCache::new(learning_rate, Self::update_function)
        }
    }

    pub fn update_function(bpc: &mut BackpropCache, derivate :&DMatrix<f64>, _index: usize) -> DMatrix<f64> {
        bpc.learning_rate*derivate.clone()
    }

    pub fn dummy_init(_bpc: &mut BackpropCache, _nrows: usize, _ncols: usize) -> usize {
        // In normal gradient descent there is no need for additional cache
        return 0;
    }
}