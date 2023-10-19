use na::DMatrix;
use super::{loss::Loss, reduction_strategy::ReductionStrategy};
use crate::utils::matrix_manipulations as mm;
struct MeanSquaredErrorLoss {
    reduction: ReductionStrategy,
    derivate: Option<DMatrix<f64> >
}

impl Loss for MeanSquaredErrorLoss {

    fn cost(&mut self, prediction: &DMatrix<f64>, ground_truth: &DMatrix<f64>) -> f64 {
        assert_eq!(prediction.nrows(), ground_truth.nrows(), "Predictions and ground truth must have the same number of rows");
        assert_eq!(prediction.ncols(), ground_truth.ncols(), "Predictions and ground truth must have the same number of columns");
        
        let batch_size = prediction.ncols() as f64;
        let output_dim = prediction.nrows() as f64;
        let error = ground_truth - prediction;
        let mse = error.map(|val| val.powi(2));
        let mses_per_sample = mm::sum_each_column(&mse);
        
    
        // Note: this derivative is for each component and not aggregated
        match self.reduction {
            ReductionStrategy::SumOverBatchSize => {
                let m = batch_size*output_dim;
                let rmses_per_sample = mses_per_sample.map(|val| (m*val).sqrt());
                self.derivate = Some((prediction - ground_truth).component_div(
                    &mm::vertical_broadcast(&rmses_per_sample, prediction.nrows())));
                (mses_per_sample.sum() / m).sqrt()
            },
            ReductionStrategy::Sum => {
                let m = output_dim;
                let rmses_per_sample = mses_per_sample.map(|val| (m*val).sqrt());
                self.derivate = Some((prediction - ground_truth).component_div(
                    &mm::vertical_broadcast(&rmses_per_sample, prediction.nrows())));
                (mses_per_sample.sum() / m).sqrt()
            },
            // ReductionStrategy::None => {
            //     TODO: Reimplement the cost function to return a DMatrix
            //     mses_per_sample
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

impl MeanSquaredErrorLoss {

    fn new() -> MeanSquaredErrorLoss {
        return MeanSquaredErrorLoss { reduction: ReductionStrategy::SumOverBatchSize, derivate: None }
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
    fn test_rmse_cost() {
        let mut cel = MeanSquaredErrorLoss::new();
        let check_epsilon = 1e-6;
        let batch_size = 2;
        // Simple case with SumOverBatchSize
        let y_pred = DMatrix::from_row_slice(3, batch_size, &[0.05, 0.95, 0.0, 0.1, 0.8, 0.1]);
        let y_true = DMatrix::from_row_slice(3, batch_size, &[0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
        
        let cost = cel.cost(&y_pred, &y_true);
        let expected = (0.24416667 as f64).sqrt();
        assert_abs_diff_eq!(cost, expected, epsilon = check_epsilon);

        // Edge case with SumOverBatchSize
        let y_pred = DMatrix::from_row_slice(2, batch_size, &[0.999999, 0.000001, 0.5, 0.5]);
        let y_true = DMatrix::from_row_slice(2, batch_size, &[1.0, 0.0, 1.0, 0.0]);
        
        let expected :f64 = 0.125;
        let cost = cel.cost(&y_pred, &y_true);
        assert_abs_diff_eq!(cost, expected.sqrt(), epsilon = check_epsilon);
        
        cel.set_reduction_type(ReductionStrategy::Sum);

        // Simple case with Sum
        let y_pred = DMatrix::from_row_slice(3, batch_size, &[0.05, 0.95, 0.0, 0.1, 0.8, 0.1]);
        let y_true = DMatrix::from_row_slice(3, batch_size, &[0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
        
        let cost = cel.cost(&y_pred, &y_true);
        let expected :f64= 0.48833334;
        assert_abs_diff_eq!(cost, expected.sqrt(), epsilon = check_epsilon);

        // Edge case with Sum
        let y_pred = DMatrix::from_row_slice(2, batch_size, &[0.999999, 0.000001, 0.5, 0.5]);
        let y_true = DMatrix::from_row_slice(2, batch_size, &[1.0, 0.0, 1.0, 0.0]);
        
        let expected :f64= 0.25;
        let cost = cel.cost(&y_pred, &y_true);
        assert_abs_diff_eq!(cost, expected.sqrt(), epsilon = check_epsilon);
    }

    #[test]
    fn test_rmse_derivative() {
        let mut cel = MeanSquaredErrorLoss::new();
        let check_epsilon = 1e-6;
        let batch_size = 2;

        // Simple case
        let y_pred = DMatrix::from_vec(3, batch_size, vec![0.05, 0.95, 0.0, 0.1, 0.8, 0.1]);
        let y_true = DMatrix::from_vec(3, batch_size, vec![0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);

        cel.cost(&y_pred, &y_true);
        let derivative = cel.get_derivate();
        let expected_derivative = DMatrix::from_vec(3, batch_size, vec![0.28867513459481275, -0.288675134594813, 0.0, 0.033786868919974296, 0.27029495135979437, -0.30408182027976866]);

        assert_abs_diff_eq!(derivative, expected_derivative, epsilon = check_epsilon);

        // Edge case with SumOverBatchSize
        let y_pred = DMatrix::from_vec(2, batch_size, vec![0.999999, 0.000001, 0.5, 0.5]);
        let y_true = DMatrix::from_vec(2, batch_size, vec![1.0, 0.0, 1.0, 0.0]);
        
        let expected_derivative = DMatrix::from_vec(2, batch_size, vec![-0.3535533905983571, 0.3535533905881904, -0.35355339059327373, 0.35355339059327373]);

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
        let expected_derivative = DMatrix::from_vec(3, batch_size, vec![0.40824829046386285, -0.40824829046386324, 0.0, 0.04778184825674965, 0.3822547860539972, -0.43003663431074685]);

        assert_abs_diff_eq!(derivative, expected_derivative, epsilon = check_epsilon);

        // Edge case with SumOverBatchSize
        let y_pred = DMatrix::from_vec(2, batch_size, vec![0.999999, 0.000001, 0.5, 0.5]);
        let y_true = DMatrix::from_vec(2, batch_size, vec![1.0, 0.0, 1.0, 0.0]);
        
        let expected_derivative = DMatrix::from_vec(2, batch_size, vec![-0.500000000007189, 0.49999999999281114, -0.5, 0.5]);

        cel.cost(&y_pred, &y_true);
        let derivative = cel.get_derivate();
        assert_abs_diff_eq!(derivative, expected_derivative, epsilon = check_epsilon);
    }
}
