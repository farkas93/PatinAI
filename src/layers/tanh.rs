extern crate nalgebra as na;
use crate::{layers::layer::Layer, optimizers::backprop_cache::BackpropCache};
use na::DMatrix;

pub struct TanHLayer {
    out: DMatrix<f64>,
}

impl Layer for TanHLayer {


    fn forward(&mut self, x: &DMatrix<f64>) -> &DMatrix<f64>{
        self.out = x.map(|val| ((val).exp() - (-val).exp()) / ((val).exp() + (-val).exp()));
        return &self.out;
    }    
    
    fn backward(&mut self, cache: &mut BackpropCache) {
        let d_g = self.out.map(|val| 1.0 - (val*val));
        cache.d_z = cache.d_a.component_mul(&d_g);
    }
    
}

impl TanHLayer {
    
    pub fn new(channel_size: usize, batch_size: usize) -> TanHLayer {
        // Since we will transpose the weights for mutliplication with
        // the input, the nr cols has to match with number of biases.
        return TanHLayer { 
            out: DMatrix::zeros(channel_size, batch_size),
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use na::DMatrix;

    #[test]
    fn test_tanh() {
        let input = DMatrix::from_vec(5,1,vec![0.0, 1.0, -1.0, 2.0, -2.0]);
        let mut tanh = TanHLayer::new(1, 1);
       
        let output = tanh.forward(&input);

        // Expected results calculated using the sigmoid formula
        let expected = DMatrix::from_vec(5,1,vec![
            0.0,                 // tanh(0)
            0.7615941559557648,  // tanh(1)
            -0.7615941559557648, // tanh(-1)
            0.9640275800758168,  // tanh(2)
            -0.9640275800758168  // tanh(-2)
        ]);

        // Use a small epsilon value for floating-point comparisons
        let epsilon = 1e-15;

        for i in 0..input.len() {
            assert_abs_diff_eq!(output[(i,0)], expected[(i,0)], epsilon = epsilon);
        }
    }
}