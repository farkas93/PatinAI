use na::{DVector, DMatrix};
use crate::optimizers::backprop_cache::BackpropCache;
use super::layer::Layer;
use crate::utils::matrix_manipulations as mm;
use crate::utils::statistics as stat;

pub struct BatchNormLayer {
    weights: DMatrix<f64>, // Corresponds to gamma in the lecture notes of Andrew Ng
    bias: DMatrix<f64>, // Corresponds to beta in the lecture notes of Andrew Ng
    mu: DMatrix<f64>,
    var: DMatrix<f64>,
    beta: f64, // The beta rate of the exponentially weighted moving average
    d_weights: DMatrix<f64>,
    d_bias: DMatrix<f64>,
    training_mode: bool,
    input: Option<DMatrix<f64>>,
    bp_ind_w: Option<usize>,
    bp_ind_b: Option<usize>
}

impl Layer for BatchNormLayer {
    
    fn forward(&mut self, x: DMatrix<f64>) -> DMatrix<f64> {
        // Execute forward pass.
        let batch_size = x.ncols();
        let broad_weights =  mm::broadcast(&self.weights, batch_size);
        let broad_bias = mm::broadcast(&self.bias, batch_size);

        if self.training_mode {
            // Save the input for backpropagation
            self.input = Some(x.clone());

            // Calculate Mean and Variance                    
            let curr_mu = stat::calc_mean(&x, batch_size);
            let curr_var = stat::calc_variance(&x, batch_size);

            // Update the mean and the variance
            self.mu = self.update_mu(&curr_mu);
            self.var = self.update_variance(&curr_var);

            let normalized_input = &self.normalize(&x, &curr_mu, &curr_var, batch_size);
            return broad_weights.component_mul(normalized_input) + broad_bias;
        }

        let normalized_input = &self.normalize(&x, &self.mu, &self.var, batch_size);
        broad_weights.component_mul(normalized_input) + broad_bias
    }

    fn backward(&mut self, cache: &mut BackpropCache) {
        let batch_size = self.bias.ncols() as f64;
        let x = self.input.as_ref().unwrap();
        // Do backward pass
        cache.d_a =  self.weights.transpose() * &cache.d_z;  
        self.d_weights =  self.sum_each_row(&cache.d_z.component_mul(x)) / batch_size;
        self.d_bias = self.sum_each_row(&cache.d_z) / batch_size;        

        // Activation functions do not have anything to update
        assert_eq!(self.weights.nrows(), self.d_weights.nrows());
        assert_eq!(self.weights.ncols(), self.d_weights.ncols());
        self.weights = &self.weights - cache.grad_step(&self.d_weights, self.bp_ind_w.unwrap()); 

        assert_eq!(self.bias.nrows(), self.d_bias.nrows());
        assert_eq!(self.bias.ncols(), self.d_bias.ncols());
        self.bias = &self.bias - cache.grad_step(&self.d_bias, self.bp_ind_b.unwrap());
        return;
    }


    fn set_training_mode(&mut self, is_on: bool) {
        // Most layers don't need a training mode.
        self.training_mode = is_on;
    }

    fn register_backprop_index(&mut self, bpc: &mut BackpropCache, init_fn: fn(&mut BackpropCache, usize, usize) -> usize) {
        self.bp_ind_w = Some(init_fn(bpc, self.weights.nrows(), self.weights.ncols()));
        self.bp_ind_b = Some(init_fn(bpc, self.bias.nrows(), self.bias.ncols()));
    }
    
}

impl BatchNormLayer {
    
    pub fn new(channels: usize) -> BatchNormLayer {
        // Since we will transpose the weights for mutliplication with
        // the input, the nr cols has to match with number of biases.    
        return BatchNormLayer { 
            weights: na::DMatrix::new_random(channels, 1),            
            d_weights: na::DMatrix::new_random(channels, 1),
            bias: na::DMatrix::zeros(channels, 1),
            d_bias: na::DMatrix::zeros(channels, 1),
            mu: na::DMatrix::zeros(channels, 1),
            var: na::DMatrix::zeros(channels, 1),
            beta: 0.9,
            training_mode: false,
            input: None,
            bp_ind_w: None,
            bp_ind_b: None,
        };
    }

    pub fn set_w_and_b(&mut self, weights: DMatrix<f64>, bias: DMatrix<f64>) {
        self.weights = weights;
        self.bias = bias;
    }
    
    pub fn set_beta(&mut self, beta: f64) {
        self.beta = beta;
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

    fn update_mu(&self, curr_mu: &DMatrix<f64>) -> DMatrix<f64>{
        self.beta * &self.mu + (1.0-self.beta)*curr_mu
    }

    fn update_variance(&self, curr_var: &DMatrix<f64>) -> DMatrix<f64>{
        self.beta * &self.var + (1.0-self.beta)*curr_var
    }

    fn normalize(&self, mat: &DMatrix<f64>, mu: &DMatrix<f64>, var: &DMatrix<f64>, batch_size: usize) -> DMatrix<f64>{
        let broad_mu = mm::broadcast(&mu, batch_size);
        let broad_var = mm::broadcast(&var, batch_size);
        let dividend = mat-&broad_mu;
        let divisor = &broad_var.map(|val| f64::sqrt(val + f64::EPSILON));
        dividend.component_div(divisor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_forward() {

        let channels = 3;
        // Create a 2x3 matrix of weights and a 3x1 vector of biases
        let mut layer = BatchNormLayer::new(channels);
        layer.set_w_and_b(DMatrix::from_vec(channels, 1, vec![1.0, 1.0, 1.0]),
                    DMatrix::from_vec(channels, 1, vec![0.5, 0.5, 0.5]));
        layer.set_training_mode(true);
        
        // Example input vector
        let batch_size = 2;
        let input = DMatrix::from_vec(channels, batch_size, vec![1.0, 1.0, 
                                                        1.0, 1.0, 
                                                        1.0, 1.0]);

        // Forward pass        
        let output = layer.forward(input);

        // Expected result: 
        let expected = DMatrix::from_vec(channels, batch_size, vec![0.5, 0.5, 0.5, 0.5, 0.5, 0.5]);
        assert_abs_diff_eq!(output, expected, epsilon = 1.0e-15);
    }

    #[test]
    fn test_new() {
        let rows = 4;
        let layer = BatchNormLayer::new(rows);

        // Check dimensions of weights and biases
        assert_eq!(layer.weights.nrows(), rows);
        assert_eq!(layer.weights.ncols(), 1);
        assert_eq!(layer.d_weights.nrows(), rows);
        assert_eq!(layer.d_weights.ncols(), 1);
        assert_eq!(layer.bias.nrows(), rows);
        assert_eq!(layer.d_bias.ncols(), 1);
        assert_eq!(layer.mu.nrows(), rows);
        assert_eq!(layer.var.ncols(), 1);
    }
}
