use na::DMatrix;
use super::{loss::Loss, reduction_strategy::ReductionStrategy};
use crate::utils::matrix_manipulations as mm;
struct CategoricalCrossentropyLoss {
    reduction: ReductionStrategy,
    derivate: Option<DMatrix<f64> >,
    normalize: bool
}

impl Loss for CategoricalCrossentropyLoss {

    fn cost(&mut self, prediction: &DMatrix<f64>, ground_truth: &DMatrix<f64>) -> f64 {
        assert_eq!(prediction.nrows(), ground_truth.nrows(), "Predictions and ground truth must have the same number of rows");
        assert_eq!(prediction.ncols(), ground_truth.ncols(), "Predictions and ground truth must have the same number of columns");
        let log_probs;
        if self.normalize {
            // Normalize the predictions before calculating the logprobs 
            let pred_batch_sum_divisor = mm::vertical_broadcast(&mm::sum_each_column(prediction).map(|val| 1.0/val), prediction.nrows());
            let normalized_preds = prediction.component_mul(&pred_batch_sum_divisor);
            log_probs = normalized_preds.map(|val| mm::clip(val, 0.0, 1.0).ln());
            // already define the additionally occuring term for the derivate 1.0/sum(sample)
            match self.reduction {
                ReductionStrategy::SumOverBatchSize => {
                    let batch_size = prediction.ncols() as f64;
                    self.derivate = Some(pred_batch_sum_divisor.map(|val| val/batch_size) )
                },
                ReductionStrategy::Sum => {
                    self.derivate = Some(pred_batch_sum_divisor)
                },
                // ReductionStrategy::None => {
                //     TODO: Reimplement the cost function to return a DMatrix
                //     losses_per_sample
                // }
            }
        }
        else {            
            log_probs = prediction.map(|val| mm::clip(val, 0.0, 1.0).ln()); 
            self.derivate = Some(DMatrix::zeros(prediction.nrows(), prediction.ncols()))
        }
        // Calculate the component-wise cross entropy
        let componentwise_cross_entropy = ground_truth.component_mul(&log_probs);
        
        // Sum across all components
        let losses_per_sample = - mm::sum_each_column(&componentwise_cross_entropy);
    
        // Note: this derivative is for each component and not aggregated
    
        match self.reduction {
            ReductionStrategy::SumOverBatchSize => {
                let batch_size = prediction.ncols() as f64;
                self.derivate = Some(self.get_derivate() - ground_truth.component_div(&prediction.map(|val| mm::clip(val, 0.0, 1.0)*batch_size)) );
                losses_per_sample.sum() / batch_size
            },
            ReductionStrategy::Sum => {
                self.derivate = Some(self.get_derivate() - ground_truth.component_div(&prediction.map(|val| mm::clip(val, 0.0, 1.0))) );
                losses_per_sample.sum()
            },
            // ReductionStrategy::None => {
            //     TODO: Reimplement the cost function to return a DMatrix
            //     losses_per_sample
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
        return CategoricalCrossentropyLoss { reduction: ReductionStrategy::SumOverBatchSize, derivate: None, normalize: false}
    }

    fn set_reduction_type(&mut self, rs: ReductionStrategy) {
        self.reduction = rs;
    }

    fn set_normalize(&mut self, is_on: bool) {
        self.normalize = is_on;
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

        // Same with normalization
        // ==============================
        
        cel.set_normalize(true);

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
        let y_pred = DMatrix::from_vec(3, batch_size, vec![0.05, 0.95, 0.0, 0.1, 0.8, 0.1]);
        let y_true = DMatrix::from_vec(3, batch_size, vec![0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);

        cel.cost(&y_pred, &y_true);
        let derivative = cel.get_derivate();
        let expected_derivative = DMatrix::from_vec(3, batch_size, vec![0.0, -1.05263158, 0.0, 0.0, 0.0, -10.0])/batch_size as f64;

        assert_abs_diff_eq!(derivative, expected_derivative, epsilon = check_epsilon);

        // Edge case with SumOverBatchSize
        let y_pred = DMatrix::from_vec(2, batch_size, vec![0.999999, 0.000001, 0.5, 0.5]);
        let y_true = DMatrix::from_vec(2, batch_size, vec![1.0, 0.0, 1.0, 0.0]);
        
        let expected_derivative = DMatrix::from_vec(2, batch_size, vec![-1.000001, 0.0, -2.0, 0.0 ])/ batch_size as f64;

        cel.cost(&y_pred, &y_true);
        let derivative = cel.get_derivate();
        assert_abs_diff_eq!(derivative, expected_derivative, epsilon = check_epsilon);
        
        //Reduction SUM
        cel.set_reduction_type(ReductionStrategy::Sum);
        // Simple case
        let y_pred = DMatrix::from_vec(3, batch_size, vec![0.05, 0.95, 0.0, 0.1, 0.8, 0.1]);
        let y_true = DMatrix::from_vec(3, batch_size, vec![0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);

        cel.cost(&y_pred, &y_true);
        let derivative = cel.get_derivate();
        let expected_derivative = DMatrix::from_vec(3, batch_size, vec![0.0, -1.05263158, 0.0, 0.0, 0.0, -10.0]);

        assert_abs_diff_eq!(derivative, expected_derivative, epsilon = check_epsilon);

        // Edge case with SumOverBatchSize
        let y_pred = DMatrix::from_vec(2, batch_size, vec![0.999999, 0.000001, 0.5, 0.5]);
        let y_true = DMatrix::from_vec(2, batch_size, vec![1.0, 0.0, 1.0, 0.0]);
        
        let expected_derivative = DMatrix::from_vec(2, batch_size, vec![-1.000001, 0.0, -2.0, 0.0 ]);

        cel.cost(&y_pred, &y_true);
        let derivative = cel.get_derivate();
        assert_abs_diff_eq!(derivative, expected_derivative, epsilon = check_epsilon);


        // Adjusted derivative with normalization
        // ==============================
        cel.set_reduction_type(ReductionStrategy::SumOverBatchSize);
        cel.set_normalize(true);

        // Simple case
        let y_pred = DMatrix::from_vec(3, batch_size, vec![0.05, 0.95, 0.0, 0.1, 0.8, 0.1]);
        let y_true = DMatrix::from_vec(3, batch_size, vec![0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);

        cel.cost(&y_pred, &y_true);
        let derivative = cel.get_derivate();
        let expected_derivative = DMatrix::from_vec(3, batch_size, vec![0.0, -1.05263158, 0.0, 0.0, 0.0, -10.0])/batch_size as f64;
        let new_expected_derivative = expected_derivative.map(|val| val + 1.0/batch_size as f64);

        assert_abs_diff_eq!(derivative, new_expected_derivative, epsilon = check_epsilon);

        // Edge case with SumOverBatchSize
        let y_pred = DMatrix::from_vec(2, batch_size, vec![0.999999, 0.000001, 0.5, 0.5]);
        let y_true = DMatrix::from_vec(2, batch_size, vec![1.0, 0.0, 1.0, 0.0]);
        
        let expected_derivative = DMatrix::from_vec(2, batch_size, vec![-1.000001, 0.0, -2.0, 0.0 ])/ batch_size as f64;
        let new_expected_derivative = expected_derivative.map(|val| val + 1.0/batch_size as f64);

        cel.cost(&y_pred, &y_true);
        let derivative = cel.get_derivate();
        assert_abs_diff_eq!(derivative, new_expected_derivative, epsilon = check_epsilon);
        
        //Reduction SUM
        cel.set_reduction_type(ReductionStrategy::Sum);
        // Simple case
        let y_pred = DMatrix::from_vec(3, batch_size, vec![0.05, 0.95, 0.0, 0.1, 0.8, 0.1]);
        let y_true = DMatrix::from_vec(3, batch_size, vec![0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);

        cel.cost(&y_pred, &y_true);
        let derivative = cel.get_derivate();
        let expected_derivative = DMatrix::from_vec(3, batch_size, vec![0.0, -1.05263158, 0.0, 0.0, 0.0, -10.0]);
        let new_expected_derivative = expected_derivative.map(|val| val + 1.0 as f64);

        assert_abs_diff_eq!(derivative, new_expected_derivative, epsilon = check_epsilon);

        // Edge case with SumOverBatchSize
        let y_pred = DMatrix::from_vec(2, batch_size, vec![0.999999, 0.000001, 0.5, 0.5]);
        let y_true = DMatrix::from_vec(2, batch_size, vec![1.0, 0.0, 1.0, 0.0]);
        
        let expected_derivative = DMatrix::from_vec(2, batch_size, vec![-1.000001, 0.0, -2.0, 0.0 ]);
        let new_expected_derivative = expected_derivative.map(|val| val + 1.0 as f64);

        cel.cost(&y_pred, &y_true);
        let derivative = cel.get_derivate();
        assert_abs_diff_eq!(derivative, new_expected_derivative, epsilon = check_epsilon);
    }
}
