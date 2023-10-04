extern crate nalgebra as na;
use na::{DVector, DMatrix};
use crate::optimizers::backprop_cache::BackpropCache;

use super::layer::Layer;

pub struct LinearLayer {
    weights: DMatrix<f64>,
    bias: DMatrix<f64>,
    d_weights: DMatrix<f64>,
    d_bias: DMatrix<f64>,
    out: DMatrix<f64>,
    d_out: DMatrix<f64>,
}

impl Layer for LinearLayer {
    
    fn forward(&mut self, x: &DMatrix<f64>) -> &DMatrix<f64> {
        assert_eq!(x.nrows(), self.weights.ncols());
        self.out = &self.weights * x + &self.bias;
        return &self.out;
    }
    

    fn backward(&mut self, cache: &mut BackpropCache) {
        let batch_size = self.bias.ncols() as f64;
        self.d_out = self.weights.transpose() * &cache.d_z;
        self.d_weights =  &cache.d_z * &cache.d_a.transpose() / batch_size;
        self.d_bias = self.sum_and_broadcast(&cache.d_z) / batch_size;
        
        cache.d_z = self.d_out.clone();
    }
    
    fn update(&mut self, learning_rate: f64) {
        // Activation functions do not have anything to update
        self.weights = &self.weights + learning_rate*&self.d_weights;
        self.bias = &self.bias + learning_rate*&self.bias;
        return;
    }
}

impl LinearLayer {
    
    pub fn new(ch_in: usize, ch_out: usize, batch_size: usize) -> LinearLayer {
        // Since we will transpose the weights for mutliplication with
        // the input, the nr cols has to match with number of biases.        
        return LinearLayer { 
            weights: na::DMatrix::new_random(ch_out, ch_in),            
            d_weights: na::DMatrix::new_random(ch_out, ch_in),
            bias: na::DMatrix::zeros(ch_out, batch_size),
            d_bias: na::DMatrix::zeros(ch_out, batch_size),
            out: na::DMatrix::zeros(ch_out, batch_size),
            d_out: na::DMatrix::zeros(ch_out, batch_size)
        };
    }

    pub fn set(&mut self, weights: DMatrix<f64>, bias: DMatrix<f64>) {
        self.weights = weights;
        self.bias = bias;
    }

    // TODO: do the broadcasting in the forward cycle to spare memory.
    fn sum_and_broadcast(&self, dZ: &DMatrix<f64>) -> DMatrix<f64>{
        let batch_size = self.bias.ncols();
        let sums_data:  Vec<f64> = (0..dZ.nrows())
        .map(|col_idx| dZ.row(col_idx).sum())
        .collect();
        let sums = DVector::from_vec(sums_data);
        let ones_vector = DVector::<f64>::from_element(batch_size, 1.0);
        sums * ones_vector.transpose()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_forward() {

        let ch_in = 2;
        let batch_size = 2;
        let ch_out = 3;
        // Create a 2x3 matrix of weights and a 3x1 vector of biases
        let mut layer = LinearLayer {
            weights: DMatrix::from_vec(ch_out, ch_in, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            bias: DMatrix::from_vec(ch_out, batch_size,vec![0.5, 0.5, 0.5, 0.5, 0.5, 0.5]),
            out: DMatrix::zeros(ch_out, batch_size),
            d_out: DMatrix::zeros(1, 1),
            d_bias: DMatrix::zeros(3, 2),
            d_weights: DMatrix::zeros(3, 2),
        };

        // Example input vector
        let input = DMatrix::from_vec(2, 2, vec![1.0, 1.0, 1.0, 1.0]);

        // Forward pass
        layer.forward(&input);
        let output = layer.out;
        println!("Output vector: {:?}", output);

        // Expected result: 
        // [1*1 + 2*1, 3*1] + [4*1, 5*1 + 6*1] + [0.5, 0.5, 0.5]
        // [5.5, 7.5, 9.5]
        let expected = DMatrix::from_vec(3,2, vec![5.5, 7.5, 9.5, 5.5, 7.5, 9.5]);

        assert_abs_diff_eq!(output, expected, epsilon = 1.0e-15);
    }

    #[test]
    fn test_create() {
        let rows = 4;
        let cols = 5;
        let bs = 2;
        let layer = LinearLayer::new(cols, rows, bs);

        // Check dimensions of weights and biases
        assert_eq!(layer.weights.nrows(), rows);
        assert_eq!(layer.weights.ncols(), cols);
        assert_eq!(layer.bias.nrows(), rows);
        assert_eq!(layer.bias.ncols(), bs);
    }
}
