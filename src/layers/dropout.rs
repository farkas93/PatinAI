use nalgebra::DMatrix;

use super::layer::Layer;

pub struct DropoutLayer {
    keep_prob: f64,
    training_mode: bool,
    dropout: Option<DMatrix<f64>>
}

impl Layer for DropoutLayer {
    fn forward(&mut self, x: DMatrix<f64>) -> DMatrix<f64> {
        if self.training_mode {
            self.dropout = Some((DMatrix::new_random(x.nrows(), x.ncols()) as DMatrix<f64>)
                                                                        .map(|val| if val > self.keep_prob {0.0} else {1.0}));
            x.component_mul(self.dropout.as_ref().unwrap())
        }
        else { x }
    }

    fn backward(&mut self, cache: &mut crate::optimizers::backprop_cache::BackpropCache) {
        let divisior = 1.0 / self.keep_prob;
        cache.d_a = cache.d_a.component_mul(self.dropout.as_ref().unwrap()).map(|val| val * divisior);
    }

    fn set_training_mode(&mut self, on: bool) {
        self.training_mode = on;
    }

}

impl DropoutLayer {
    pub fn new(keep_prob: f64) -> DropoutLayer {
        return DropoutLayer {
            keep_prob: keep_prob,
            training_mode: false,
            dropout: None
        };
    }
}



#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DMatrix;

    #[test]
    fn test_dropout_forward_inference() {
        let mut layer = DropoutLayer::new(0.5);
        let input = DMatrix::from_vec(2, 2, vec![0.5, 1.0, 1.5, 2.0]);

        let output = layer.forward(input.clone());

        assert_eq!(input, output);
    }

    #[test]
    fn test_dropout_forward_training() {
        let mut layer = DropoutLayer::new(0.5);
        layer.set_training_mode(true);
        let input = DMatrix::from_vec(2, 2, vec![0.5, 1.0, 1.5, 2.0]);

        let output = layer.forward(input.clone());

        for i in 0..input.nrows() {
            for j in 0..input.ncols() {
                let value = output[(i, j)];
                assert!(value == 0.0 || value == input[(i, j)]);
            }
        }
    }

    #[test]
    fn test_dropout_backward() {
        let mut layer = DropoutLayer::new(0.5);
        layer.set_training_mode(true);
        let mut cache = crate::optimizers::backprop_cache::BackpropCache {
            d_a: DMatrix::from_vec(2, 2, vec![1.0, 1.0, 1.0, 1.0]),
            d_z: DMatrix::zeros(1, 1),
            beta_1: 0.0,
            beta_2: 0.0,
            learning_rate: 0.0,
            epsilon: 0.0,
            curr_iter: 0,
            grad_step_fn: crate::optimizers::gradient_descent::GradientDescent::update_function,
            vec_v: None,
            vec_s: None,
        };
        layer.forward(DMatrix::from_vec(2, 2, vec![1.0, 1.0, 1.0, 1.0]));
        layer.backward(&mut cache);

        for i in 0..cache.d_a.nrows() {
            for j in 0..cache.d_a.ncols() {
                let value = cache.d_a[(i, j)];
                assert!(value == 0.0 || value == 2.0);
            }
        }
    }
}
