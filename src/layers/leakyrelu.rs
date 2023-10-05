extern crate nalgebra as na;
use na::DMatrix;
use crate::{layers::layer::Layer, optimizers::backprop_cache::BackpropCache};


pub struct LeakyReLULayer {
    out: Option<DMatrix<f64>>,
    grad : f64
}

impl Layer for LeakyReLULayer {


    fn forward(&mut self, x: &DMatrix<f64>) -> &DMatrix<f64>{
        self.out = Some(x.map(|val| f64::max(self.grad*val, val)));
        return &self.out.as_ref().unwrap();
    }

    fn backward(&mut self, cache: &mut BackpropCache){
        let d_g = self.out.as_ref().unwrap().map(|val| if val > 0.0 {1.0} else {-self.grad});
        cache.d_z = cache.d_a.component_mul(&d_g);
    }
        
}

impl LeakyReLULayer {    

    pub fn new() -> LeakyReLULayer {
        // Since we will transpose the weights for mutliplication with
        // the input, the nr cols has to match with number of biases.
        return LeakyReLULayer { 
            out: None,
            grad: 0.01
        };
    }

    pub fn set(&mut self, grad: f64) {
        self.grad = grad;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_leaky_relu() {
        let input = DMatrix::from_vec(5, 1, vec![0.0, 1.0, -1.0, 2.0, -2.0]);
        let mut relu = LeakyReLULayer::new();
       

        let output = relu.forward(&input);
        // Expected results calculated using the sigmoid formula
        let expected = DMatrix::from_vec(5, 1, vec![
            0.0, // relu(0)
            1.0, // relu(1)
            -0.01, // relu(-1)
            2.0, // relu(2)
            -0.02  // relu(-2)
        ]);

        // Use a small epsilon value for floating-point comparisons
        let epsilon = 1e-15;

        for i in 0..input.len() {
            assert_abs_diff_eq!(output[i], expected[i], epsilon = epsilon);
        }

        let grad = 0.2;

        relu.set(grad);
        let output = relu.forward(&input);

        // Expected results calculated using the sigmoid formula
        let expected = DMatrix::from_vec(5, 1, vec![
            0.0, // relu(0)
            1.0, // relu(1)
            -0.2, // relu(-1)
            2.0, // relu(2)
            -0.4  // relu(-2)
        ]);

        // Use a small epsilon value for floating-point comparisons
        let epsilon = 1e-15;

        for i in 0..input.len() {
            assert_abs_diff_eq!(output[i], expected[i], epsilon = epsilon);
        }

    }
}