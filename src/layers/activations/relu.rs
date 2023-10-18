use crate::{layers::layer::Layer, optimizers::backprop_cache::BackpropCache};
use na::DMatrix;

pub struct ReLULayer {    
    out: Option<DMatrix<f64>>,
}

impl Layer for ReLULayer {

    fn forward(&mut self, x: DMatrix<f64>) -> DMatrix<f64>{
        self.out = Some(x.map(|val| f64::max(0.0, val)));
        return self.out.as_ref().unwrap().clone();
    }

    
    fn backward(&mut self, cache: &mut BackpropCache) {
        let d_g = self.out.as_ref().unwrap().map(|val| if val > 0.0 { 1.0 } else { 0.0 });
        cache.d_z = cache.d_a.component_mul(&d_g);
    }
    
}

impl ReLULayer {
    
    pub fn new() -> ReLULayer {
        // Since we will transpose the weights for mutliplication with
        // the input, the nr cols has to match with number of biases.
        return ReLULayer { 
            out: None,
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use na::DVector;

    #[test]
    fn test_relu() {
        let input = DMatrix::from_vec(5,1,vec![0.0, 1.0, -1.0, 2.0, -2.0]);
        let mut relu = ReLULayer::new();
        
        let output = relu.forward(input.clone());

        // Expected results calculated using the sigmoid formula
        let expected = DVector::from_vec(vec![
            0.0, // relu(0)
            1.0, // relu(1)
            0.0, // relu(-1)
            2.0, // relu(2)
            0.0  // relu(-2)
        ]);

        // Use a small epsilon value for floating-point comparisons
        let epsilon = 1e-15;

        for i in 0..input.len() {
            assert_abs_diff_eq!(output[i], expected[i], epsilon = epsilon);
        }
    }
}