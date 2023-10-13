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
    input: Option<DMatrix<f64>>,
}

impl Layer for LinearLayer {
    
    fn forward(&mut self, x: &DMatrix<f64>) -> &DMatrix<f64> {
        assert_eq!(x.nrows(), self.weights.ncols());
        // Save the input for backpropagation
        self.input = Some(x.clone());
        // Execute forward pass.
        self.out = &self.weights * x + &self.broadcast_bias(x.ncols());
        return &self.out;
    }    

    fn backward(&mut self, cache: &mut BackpropCache) {
        let batch_size = self.bias.ncols() as f64;
        let x = self.input.as_ref().unwrap();
        // Do backward pass
        cache.d_a =  self.weights.transpose() * &cache.d_z;  
        self.d_weights =  &cache.d_z * x.transpose() / batch_size;
        self.d_bias = self.sum_each_row(&cache.d_z) / batch_size;        
    }
    
    fn update(&mut self, learning_rate: f64, update_fn: fn(&DMatrix<f64>) -> DMatrix<f64>) {
        // Activation functions do not have anything to update
        assert_eq!(self.weights.nrows(), self.d_weights.nrows());
        assert_eq!(self.weights.ncols(), self.d_weights.ncols());
        let update_term = update_fn(&self.d_weights); 
        self.weights = &self.weights - learning_rate*&self.d_weights;

        assert_eq!(self.bias.nrows(), self.d_bias.nrows());
        assert_eq!(self.bias.ncols(), self.d_bias.ncols());

        let update_term = update_fn(&self.d_bias);
        self.bias = &self.bias - learning_rate*&self.d_bias;
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
            bias: na::DMatrix::zeros(ch_out, 1),
            d_bias: na::DMatrix::zeros(ch_out, 1),
            out: na::DMatrix::zeros(ch_out, batch_size),
            input: None,
        };
    }

    pub fn set(&mut self, weights: DMatrix<f64>, bias: DMatrix<f64>) {
        self.weights = weights;
        self.bias = bias;
    }
    
    fn broadcast_bias(&self, batch_size: usize) ->DMatrix<f64> {
        let ones_vector = DMatrix::<f64>::from_element(1, batch_size, 1.0);
        &self.bias * ones_vector
    }

    fn sum_each_row(&self, d_z: &DMatrix<f64>) -> DMatrix<f64>{
        let sums_data:  Vec<f64> = (0..d_z.nrows())
        .map(|idx| d_z.row(idx).sum())
        .collect();
        let nrows = self.bias.nrows();
        let ncols = self.bias.ncols();
        DMatrix::from_vec(nrows, ncols, sums_data)
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
            bias: DMatrix::from_vec(ch_out, 1, vec![0.5, 0.5, 0.5]),
            out: DMatrix::zeros(ch_out, batch_size),
            input: None,
            d_bias: DMatrix::zeros(ch_out, 1),
            d_weights: DMatrix::zeros(ch_out, ch_in),
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
        assert_eq!(layer.d_weights.nrows(), rows);
        assert_eq!(layer.d_weights.ncols(), cols);
        assert_eq!(layer.bias.len(), rows);
        assert_eq!(layer.d_bias.len(), rows);
        assert_eq!(layer.out.nrows(), rows);
        assert_eq!(layer.out.ncols(), bs);
    }
}
