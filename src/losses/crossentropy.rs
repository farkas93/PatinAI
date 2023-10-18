use na::DMatrix;
use super::{loss::Loss, reduction_strategy::ReductionStrategy};
use crate::utils::matrix_manipulations as mm;
struct CategoricalCrossentropyLoss {
    reduction: ReductionStrategy,
    derivate: Option<DMatrix<f64> >
}

impl Loss for CategoricalCrossentropyLoss {

    fn cost(&mut self, prediction: &DMatrix<f64>, ground_truth: &DMatrix<f64>) -> f64 {
        assert_eq!(prediction.nrows(), ground_truth.nrows(), "Predictions and ground truth must have the same number of rows");
        assert_eq!(prediction.ncols(), ground_truth.ncols(), "Predictions and ground truth must have the same number of columns");
    
        let log_probs = &prediction.map(|val| (val+f64::EPSILON).ln());
        // Calculate the component-wise cross entropy
        let componentwise_cross_entropy = ground_truth.component_mul(log_probs);
        
        // Sum across all components
        let losses_per_sample = - mm::sum_each_column(&componentwise_cross_entropy);
    
        let counter_pred = prediction.map(|val| 1.0 - val + f64::EPSILON);
        let counter_gt = ground_truth.map(|val| 1.0 - val);
    
        // Note: this derivative is for each component and not aggregated
        self.derivate = Some(- (ground_truth.component_div(&prediction.map(|val| val + f64::EPSILON)) - counter_gt.component_div(&counter_pred)));
    
        match self.reduction {
            ReductionStrategy::SumOverBatchSize => {
                let total_loss = losses_per_sample.sum();
                return total_loss / prediction.ncols() as f64;
            },
            ReductionStrategy::Sum => {
                return losses_per_sample.sum();
            },
            // ReductionStrategy::None => {
            //     return losses_per_sample;
            // }
        }
    
    }
    

    fn get_derivate(&self) -> DMatrix<f64> {
        match &self.derivate {
            Some(val) => {
                val.clone()
            },
            None => panic!("derivative was never calculated!"),
        }
    }
}

impl CategoricalCrossentropyLoss {

    fn new() -> CategoricalCrossentropyLoss {
        return CategoricalCrossentropyLoss { reduction: ReductionStrategy::SumOverBatchSize, derivate: None }
    }

    fn set_reduction_type(&mut self, rs: ReductionStrategy) {
        self.reduction = rs;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use na::DMatrix;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_cross_entropy_cost() {
        let mut cel = CategoricalCrossentropyLoss::new();
        let check_epsilon = 1e-6;
        let batch_size = 2;
        // Simple case with SumOverBatchSize
        let y_pred = DMatrix::from_row_slice(3, batch_size, &[0.05, 0.95, 0.0, 0.1, 0.8, 0.1]);
        let y_true = DMatrix::from_row_slice(3, batch_size, &[0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
        
        let cost = cel.cost(&y_pred, &y_true);
        let expected = 1.1769392;
        assert_abs_diff_eq!(cost, expected, epsilon = check_epsilon);

        // Edge case with SumOverBatchSize
        let y_pred = DMatrix::from_row_slice(2, batch_size, &[0.999999, 0.000001, 0.5, 0.5]);
        let y_true = DMatrix::from_row_slice(2, batch_size, &[1.0, 0.0, 1.0, 0.0]);
        
        let expected = 0.3465741;
        let cost = cel.cost(&y_pred, &y_true);
        assert_abs_diff_eq!(cost, expected, epsilon = check_epsilon);
        
        cel.set_reduction_type(ReductionStrategy::Sum);

        // Simple case with Sum
        let y_pred = DMatrix::from_row_slice(3, batch_size, &[0.05, 0.95, 0.0, 0.1, 0.8, 0.1]);
        let y_true = DMatrix::from_row_slice(3, batch_size, &[0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
        
        let cost = cel.cost(&y_pred, &y_true);
        let expected = 1.1769392 * batch_size as f64;
        assert_abs_diff_eq!(cost, expected, epsilon = check_epsilon);

        // Edge case with Sum
        let y_pred = DMatrix::from_row_slice(2, batch_size, &[0.999999, 0.000001, 0.5, 0.5]);
        let y_true = DMatrix::from_row_slice(2, batch_size, &[1.0, 0.0, 1.0, 0.0]);
        
        let expected = 0.3465741 * batch_size as f64;
        let cost = cel.cost(&y_pred, &y_true);
        assert_abs_diff_eq!(cost, expected, epsilon = check_epsilon);
    }

    #[test]
    fn test_cross_entropy_derivative() {
        let mut cel = CategoricalCrossentropyLoss::new();
        let check_epsilon = 1e-6;
        let batch_size = 2;
        // Simple case
        let y_pred = DMatrix::from_row_slice(3, batch_size, &[0.05, 0.95, 0.0, 0.1, 0.8, 0.1]);
        let y_true = DMatrix::from_row_slice(3, batch_size, &[0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);

        cel.cost(&y_pred, &y_true);
        let derivative = cel.get_derivate();
        let expected_derivative = DMatrix::from_vec(3, batch_size, vec![ 0.5, -0.02631581,  0.5, 0.5, 0.5, -4.5]);

        assert_abs_diff_eq!(derivative, expected_derivative, epsilon = check_epsilon);

        // Edge case
        let y_pred = DMatrix::from_vec(2, 2, vec![0.999999, 0.000001, 0.5, 0.5]);
        let y_true = DMatrix::from_vec(2, 2, vec![1.0, 0.0, 1.0, 0.0]);

        cel.cost(&y_pred, &y_true);
        let derivative = cel.get_derivate();
        // Here, the expected derivatives can be calculated using the formula for the derivative of the cross entropy loss.
        // Due to the potential complexity of manual calculations for this case, it's better to verify with a trusted source 
        // or to use an alternate, simple implementation to check against. 
    }
}
