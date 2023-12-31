use na::{DVector, DMatrix};
use crate::optimizers::backprop_cache::BackpropCache;
use crate::utils::matrix_manipulations as mm;
use super::layer::Layer;

pub struct LinearLayer {
    weights: DMatrix<f64>,
    bias: DMatrix<f64>,
    d_weights: DMatrix<f64>,
    d_bias: DMatrix<f64>,
    input: Option<DMatrix<f64>>,
    bp_ind_w: Option<usize>,
    bp_ind_b: Option<usize>
}

impl Layer for LinearLayer {
    
    fn forward(&mut self, x: DMatrix<f64>) -> DMatrix<f64> {
        assert_eq!(x.nrows(), self.weights.ncols());
        // Save the input for backpropagation
        self.input = Some(x.clone());
        // Execute forward pass.
        let batch_size = self.input.as_ref().unwrap().ncols();
        self.weights.clone() * x + mm::broadcast(&self.bias, batch_size)
    }    

    fn backward(&mut self, cache: &mut BackpropCache) {
        let batch_size = self.bias.ncols() as f64;
        let x = self.input.as_ref().unwrap();
        // Do backward pass
        cache.d_a =  self.weights.transpose() * &cache.d_z;  
        self.d_weights =  &cache.d_z * x.transpose() / batch_size;
        self.d_bias = mm::sum_each_row(&cache.d_z) / batch_size;        

        // Activation functions do not have anything to update
        assert_eq!(self.weights.nrows(), self.d_weights.nrows());
        assert_eq!(self.weights.ncols(), self.d_weights.ncols());
        self.weights = &self.weights - cache.grad_step(&self.d_weights, self.bp_ind_w.unwrap()); 

        assert_eq!(self.bias.nrows(), self.d_bias.nrows());
        assert_eq!(self.bias.ncols(), self.d_bias.ncols());
        self.bias = &self.bias - cache.grad_step(&self.d_bias, self.bp_ind_b.unwrap());
        return;
    }


    fn register_backprop_index(&mut self, bpc: &mut BackpropCache, init_fn: fn(&mut BackpropCache, usize, usize) -> usize) {
        self.bp_ind_w = Some(init_fn(bpc, self.weights.nrows(), self.weights.ncols()));
        self.bp_ind_b = Some(init_fn(bpc, self.bias.nrows(), self.bias.ncols()));
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
            input: None,
            bp_ind_w: None,
            bp_ind_b: None
        };
    }

    pub fn set_w_and_b(&mut self, weights: DMatrix<f64>, bias: DMatrix<f64>) {
        self.weights = weights;
        self.bias = bias;
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
            d_bias: DMatrix::zeros(ch_out, 1),
            d_weights: DMatrix::zeros(ch_out, ch_in),
            input: None,
            bp_ind_w: None,
            bp_ind_b: None
        };

        // Example input vector
        let input = DMatrix::from_vec(ch_in, batch_size, vec![1.0, 1.0, 1.0, 1.0]);

        // Forward pass        
        let output = layer.forward(input);
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
    }
}
