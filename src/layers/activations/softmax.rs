use na::{DVector, DMatrix};
use crate::optimizers::backprop_cache::BackpropCache;
use crate::layers::layer::Layer;
use crate::utils::matrix_manipulations as mm;

pub struct SoftmaxLayer {
    out: Option<DMatrix<f64>>,
    
}

impl Layer for SoftmaxLayer {

    fn forward(&mut self, x: DMatrix<f64>) -> DMatrix<f64>{

        let exps = x.map(|val| val.exp());
        let inv_sums = mm::vertical_broadcast(&mm::sum_each_column(&exps).map(|val| 1.0/val), exps.nrows());
        self.out= Some(exps.component_mul(&inv_sums));
        return self.out.as_ref().unwrap().clone();
    }

    fn backward(&mut self, cache: &mut BackpropCache) {
        let d_g = self.out.as_ref().unwrap().map(|val| val * (1.0 - val));
        cache.d_z = cache.d_a.component_mul(&d_g);
    }
    
}

impl SoftmaxLayer {
    
    pub fn new() -> SoftmaxLayer {
        // Since we will transpose the weights for mutliplication with
        // the input, the nr cols has to match with number of biases.
        return SoftmaxLayer { 
            out: None,
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use na::DMatrix;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_forward_single() {
        let channels = 3;
        let batch_size = 1; 
        let mut layer = SoftmaxLayer::new();
        let input = DMatrix::from_row_slice(channels, batch_size, &[1.0, 2.0, 3.0]);
        let output = layer.forward(input);

        let sum: f64 = output.column(0).iter().sum();

        let expected = DMatrix::from_row_slice(channels, batch_size, &[0.09003057, 0.24472847, 0.66524096]);
        assert_eq!(output.nrows(), channels);
        assert_eq!(output.ncols(), batch_size);
        assert_abs_diff_eq!(output, expected, epsilon = 1e-8);
        assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-8);
    }

    #[test]
    fn test_forward_multiple() {
        let mut layer = SoftmaxLayer::new();
        let channels = 3;
        let batch_size = 3; 
        let input = DMatrix::from_row_slice(channels, batch_size, &[
            1.0, 3.0, 2.0,
            1.0, 2.0, 3.0,
            1.0, 1.0, 1.0           
        ]);
        let output = layer.forward(input);

        let sum1: f64 = output.column(0).iter().sum();
        let sum2: f64 = output.column(1).iter().sum();

        let expected = DMatrix::from_row_slice(batch_size, channels, &[
            0.33333333333, 0.66524096, 0.24472847,  
            0.33333333333, 0.24472847, 0.66524096,
            0.33333333333, 0.09003057, 0.09003057
        ]);

        assert_eq!(output.nrows(), channels);
        assert_eq!(output.ncols(), batch_size);

        // First row
        assert_abs_diff_eq!(output, expected, epsilon = 1e-8);

        assert_abs_diff_eq!(sum1, 1.0, epsilon = 1e-8);
        assert_abs_diff_eq!(sum2, 1.0, epsilon = 1e-8);
    }
}
