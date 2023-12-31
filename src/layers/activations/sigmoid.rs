use na::{DVector, DMatrix};
use crate::optimizers::backprop_cache::BackpropCache;
use crate::layers::layer::Layer;

pub struct SigmoidLayer {
    out: Option<DMatrix<f64>>,
}

impl Layer for SigmoidLayer {

    fn forward(&mut self, x: DMatrix<f64>) -> DMatrix<f64>{
        self.out= Some(x.map(|val| 1.0 / (1.0 + (-val).exp())));
        return self.out.as_ref().unwrap().clone();
    }

    fn backward(&mut self, cache: &mut BackpropCache) {
        let d_g = self.out.as_ref().unwrap().map(|val| val * (1.0 - val));
        cache.d_z = cache.d_a.component_mul(&d_g);
    }
    
}

impl SigmoidLayer {
    
    pub fn new() -> SigmoidLayer {
        // Since we will transpose the weights for mutliplication with
        // the input, the nr cols has to match with number of biases.
        return SigmoidLayer { 
            out: None,
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_sigmoid() {
        let input = DMatrix::from_vec(5,1,vec![0.0, 1.0, -1.0, 2.0, -2.0]);
        let mut sigmoid = SigmoidLayer::new();
        
        let output = sigmoid.forward(input.clone());

        // Expected results calculated using the sigmoid formula
        let expected = DVector::from_vec(vec![
            0.5,                 // sigmoid(0)
            0.7310585786300049,  // sigmoid(1)
            0.2689414213699951,  // sigmoid(-1)
            0.8807970779778823,  // sigmoid(2)
            0.11920292202211755  // sigmoid(-2)
        ]);

        // Use a small epsilon value for floating-point comparisons
        let epsilon = 1e-15;

        for i in 0..input.len() {
            assert_abs_diff_eq!(output[(i,0)], expected[(i,0)], epsilon = epsilon);
        }
    }
}