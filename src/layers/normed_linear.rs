extern crate nalgebra as na;
use na::{DVector, DMatrix};
use crate::optimizers::backprop_cache::BackpropCache;

use super::layer::Layer;

pub struct NormedLinearLayer {
    weights: DMatrix<f64>,
    bias: DMatrix<f64>,
    gamma: DMatrix<f64>,
    z: DMatrix<f64>,
    d_weights: DMatrix<f64>,
    d_bias: DMatrix<f64>,
    d_gamma: DMatrix<f64>,
    input: Option<DMatrix<f64>>,
    bp_ind_w: Option<usize>,
    bp_ind_b: Option<usize>,
    bp_ind_g: Option<usize>
}

//TODO: Implement batch normalization.
impl Layer for NormedLinearLayer {
    
    fn forward(&mut self, x: DMatrix<f64>) -> DMatrix<f64> {
        assert_eq!(x.nrows(), self.weights.ncols());
        // Save the input for backpropagation
        self.input = Some(x.clone());
        // Execute forward pass.
        let batch_size = self.input.as_ref().unwrap().ncols();
        self.z = self.weights.clone() * x;
        self.broadcast(&self.gamma, batch_size).component_mul(&self.norm(&self.z)) + self.broadcast(&self.bias, batch_size)
    }    

    fn backward(&mut self, cache: &mut BackpropCache) {
        let batch_size = self.bias.ncols() as f64;
        let x = self.input.as_ref().unwrap();
        // Do backward pass
        let d_z_tilde = &cache.d_z;
        let div_bs = 1.0/(batch_size as f64);
        let mu = self.sum_each_row(&self.z).map(|val| val*div_bs);
        let d_z = (batch_size-1.0) * self.gamma.component_div(&self.variance(&self.z, &mu));
        cache.d_a =  self.weights.transpose() * d_z.clone(); 
        self.d_weights = d_z * x.transpose();
        self.d_bias = self.sum_each_row(&d_z_tilde);
        self.d_gamma = d_z_tilde.component_mul(&self.norm(&self.z));  

        // Activation functions do not have anything to update
        assert_eq!(self.weights.nrows(), self.d_weights.nrows());
        assert_eq!(self.weights.ncols(), self.d_weights.ncols());
        self.weights = &self.weights - cache.grad_step(&self.d_weights, self.bp_ind_w.unwrap()); 

        assert_eq!(self.bias.nrows(), self.d_bias.nrows());
        assert_eq!(self.bias.ncols(), self.d_bias.ncols());
        self.bias = &self.bias - cache.grad_step(&self.d_bias, self.bp_ind_b.unwrap());


        assert_eq!(self.gamma.nrows(), self.d_gamma.nrows());
        assert_eq!(self.gamma.ncols(), self.d_gamma.ncols());
        self.gamma = &self.gamma - cache.grad_step(&self.d_gamma, self.bp_ind_g.unwrap());
    }


    fn register_backprop_index(&mut self, bpc: &mut BackpropCache, init_fn: fn(&mut BackpropCache, usize, usize) -> usize) {
        self.bp_ind_w = Some(init_fn(bpc, self.weights.nrows(), self.weights.ncols()));
        self.bp_ind_b = Some(init_fn(bpc, self.bias.nrows(), self.bias.ncols()));
        self.bp_ind_g = Some(init_fn(bpc, self.gamma.nrows(), self.gamma.ncols()));
    }
    
}

impl NormedLinearLayer {
    
    pub fn new(ch_in: usize, ch_out: usize, batch_size: usize) -> NormedLinearLayer {
        // Since we will transpose the weights for mutliplication with
        // the input, the nr cols has to match with number of biases.    
        return NormedLinearLayer { 
            weights: na::DMatrix::new_random(ch_out, ch_in),            
            d_weights: na::DMatrix::new_random(ch_out, ch_in),
            bias: na::DMatrix::zeros(ch_out, 1),
            d_bias: na::DMatrix::zeros(ch_out, 1),
            gamma: na::DMatrix::zeros(ch_out, 1),
            d_gamma: na::DMatrix::zeros(ch_out, 1),
            z: na::DMatrix::zeros(ch_out, 1),
            input: None,
            bp_ind_w: None,
            bp_ind_b: None,
            bp_ind_g: None
        };
    }

    pub fn set(&mut self, weights: DMatrix<f64>, bias: DMatrix<f64>) {
        self.weights = weights;
        self.bias = bias;
    }
    
    fn broadcast(&self, mat: &DMatrix<f64>, batch_size: usize) ->DMatrix<f64> {
        let ones_vector = DMatrix::<f64>::from_element(1, batch_size, 1.0);
        &mat.clone() * ones_vector
    }

    fn sum_each_row(&self, mat: &DMatrix<f64>) -> DMatrix<f64>{
        // Input from (n x m) shaped matrix the sum along the columns. 
        // Returns (n, 1)
        let sums_data:  Vec<f64> = (0..mat.nrows())
        .map(|idx| mat.row(idx).sum())
        .collect();
        let nrows = mat.nrows();
        let ncols = 1;
        DMatrix::from_vec(nrows, ncols, sums_data)
    }

    fn variance(&self, mat: &DMatrix<f64>, mu: &DMatrix<f64>) -> DMatrix<f64>{
        let batch_size = mat.ncols();
        let div_bs = 1.0/(batch_size as f64);
        self.sum_each_row(&(mat-mu).map(|val| val.powi(2)*div_bs))
    }

    fn norm(&self, mat: &DMatrix<f64>) -> DMatrix<f64>{
        let batch_size = mat.ncols();
        let div_bs = 1.0/(batch_size as f64);
        let mu = self.sum_each_row(&mat).map(|val| val*div_bs);
        let broad_mu = self.broadcast(&mu, batch_size);
        let var = self.variance(mat, &broad_mu);
        (mat-broad_mu).component_div(&var.map(|val| f64::sqrt(val + f64::EPSILON)))
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
        let mut layer = NormedLinearLayer {
            weights: DMatrix::from_vec(ch_out, ch_in, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            bias: DMatrix::from_vec(ch_out, 1, vec![0.5, 0.5, 0.5]),
            gamma: DMatrix::from_vec(ch_out, 1, vec![1.0, 1.0, 1.0]),
            z: DMatrix::zeros(ch_out, 1),
            input: None,
            d_bias: DMatrix::zeros(ch_out, 1),
            d_gamma: DMatrix::zeros(ch_out, 1),
            d_weights: DMatrix::zeros(ch_out, ch_in),
            bp_ind_w: None,
            bp_ind_b: None,
            bp_ind_g: None
        };

        // Example input vector
        let input = DMatrix::from_vec(2, 2, vec![1.0, 1.0, 1.0, 1.0]);

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
        let layer = NormedLinearLayer::new(cols, rows, bs);

        // Check dimensions of weights and biases
        assert_eq!(layer.weights.nrows(), rows);
        assert_eq!(layer.weights.ncols(), cols);
        assert_eq!(layer.d_weights.nrows(), rows);
        assert_eq!(layer.d_weights.ncols(), cols);
        assert_eq!(layer.bias.len(), rows);
        assert_eq!(layer.d_bias.len(), rows);
    }
}
