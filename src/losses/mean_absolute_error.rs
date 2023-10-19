use na::DMatrix;
use super::{loss::Loss, reduction_strategy::ReductionStrategy};
use crate::utils::matrix_manipulations as mm;
struct MeanAbsoluteErrorLoss {
    reduction: ReductionStrategy,
    derivate: Option<DMatrix<f64> >
}

impl Loss for MeanAbsoluteErrorLoss {

    fn cost(&mut self, prediction: &DMatrix<f64>, ground_truth: &DMatrix<f64>) -> f64 {
        assert_eq!(prediction.nrows(), ground_truth.nrows(), "Predictions and ground truth must have the same number of rows");
        assert_eq!(prediction.ncols(), ground_truth.ncols(), "Predictions and ground truth must have the same number of columns");
        
        let batch_size = prediction.ncols() as f64;
        let output_dim = prediction.nrows() as f64;
        let error = ground_truth - prediction;
        let mses_per_sample = mm::sum_each_column(&&error.abs());
    
        // Note: this derivative is for each component and not aggregated
        match self.reduction {
            ReductionStrategy::SumOverBatchSize => {
                let one_over_m = 1.0 / (batch_size*output_dim);
                self.derivate = Some(-error.map(|val| if val > 0.0 {one_over_m} else if val < 0.0 {-one_over_m} else {0.0}));
                mses_per_sample.sum() * one_over_m
            },
            ReductionStrategy::Sum => {
                let one_over_m = 1.0 / (output_dim);
                self.derivate = Some(-error.map(|val| if val > 0.0 {one_over_m} else if val < 0.0 {-one_over_m} else {0.0}));
                mses_per_sample.sum() * one_over_m
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

impl MeanAbsoluteErrorLoss {

    fn new() -> MeanAbsoluteErrorLoss {
        return MeanAbsoluteErrorLoss { reduction: ReductionStrategy::SumOverBatchSize, derivate: None }
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
    fn test_mae_cost() {
        let mut cel = MeanAbsoluteErrorLoss::new();
        let check_epsilon = 1e-6;
        let batch_size = 2;
        // Simple case with SumOverBatchSize
        let y_pred = DMatrix::from_row_slice(3, batch_size, &[0.05, 0.95, 0.0, 0.1, 0.8, 0.1]);
        let y_true = DMatrix::from_row_slice(3, batch_size, &[0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
        
        let cost = cel.cost(&y_pred, &y_true);
        let expected = 0.31666666;
        assert_abs_diff_eq!(cost, expected, epsilon = check_epsilon);

        // Edge case with SumOverBatchSize
        let y_pred = DMatrix::from_row_slice(2, batch_size, &[0.999999, 0.000001, 0.5, 0.5]);
        let y_true = DMatrix::from_row_slice(2, batch_size, &[1.0, 0.0, 1.0, 0.0]);
        
        let expected = 0.25000027;
        let cost = cel.cost(&y_pred, &y_true);
        assert_abs_diff_eq!(cost, expected, epsilon = check_epsilon);
        
        cel.set_reduction_type(ReductionStrategy::Sum);

        // Simple case with Sum
        let y_pred = DMatrix::from_row_slice(3, batch_size, &[0.05, 0.95, 0.0, 0.1, 0.8, 0.1]);
        let y_true = DMatrix::from_row_slice(3, batch_size, &[0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
        
        let cost = cel.cost(&y_pred, &y_true);
        let expected = 0.6333333;
        assert_abs_diff_eq!(cost, expected, epsilon = check_epsilon);

        // Edge case with Sum
        let y_pred = DMatrix::from_row_slice(2, batch_size, &[0.999999, 0.000001, 0.5, 0.5]);
        let y_true = DMatrix::from_row_slice(2, batch_size, &[1.0, 0.0, 1.0, 0.0]);
        
        let expected = 0.50000054;
        let cost = cel.cost(&y_pred, &y_true);
        assert_abs_diff_eq!(cost, expected, epsilon = check_epsilon);
    }

    #[test]
    fn test_mae_derivative() {
        let mut cel = MeanAbsoluteErrorLoss::new();
        let check_epsilon = 1e-6;
        let batch_size = 2;

        // Simple case
        let y_pred = DMatrix::from_vec(3, batch_size, vec![0.05, 0.95, 0.0, 0.1, 0.8, 0.1]);
        let y_true = DMatrix::from_vec(3, batch_size, vec![0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);

        cel.cost(&y_pred, &y_true);
        let derivative = cel.get_derivate();
        let expected_derivative = DMatrix::from_vec(3, batch_size, vec![ 0.16666667, -0.16666667, 0.0, 0.16666667,  0.16666667, -0.16666667]);

        assert_abs_diff_eq!(derivative, expected_derivative, epsilon = check_epsilon);

        // Edge case with SumOverBatchSize
        let y_pred = DMatrix::from_vec(2, batch_size, vec![0.999999, 0.000001, 0.5, 0.5]);
        let y_true = DMatrix::from_vec(2, batch_size, vec![1.0, 0.0, 1.0, 0.0]);
        
        let expected_derivative = DMatrix::from_vec(2, batch_size, vec![-0.25, 0.25, -0.25, 0.25]);

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
        let expected_derivative = DMatrix::from_vec(3, batch_size, vec![ 0.33333334, -0.33333334, 0.0, 0.33333334,  0.33333334, -0.33333334]);

        assert_abs_diff_eq!(derivative, expected_derivative, epsilon = check_epsilon);

        // Edge case with SumOverBatchSize
        let y_pred = DMatrix::from_vec(2, batch_size, vec![0.999999, 0.000001, 0.5, 0.5]);
        let y_true = DMatrix::from_vec(2, batch_size, vec![1.0, 0.0, 1.0, 0.0]);
        
        let expected_derivative = DMatrix::from_vec(2, batch_size, vec![-0.5, 0.5, -0.5,  0.5]);

        cel.cost(&y_pred, &y_true);
        let derivative = cel.get_derivate();
        assert_abs_diff_eq!(derivative, expected_derivative, epsilon = check_epsilon);
    }
}
