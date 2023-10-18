use na::DMatrix;
use super::matrix_manipulations;

pub fn calc_mean(batch: &DMatrix<f64>, batch_size: usize) -> DMatrix<f64> {
    assert_ne!(batch_size, 0);
    let div_bs = 1.0/(batch_size as f64);
    matrix_manipulations::sum_each_row(&batch).map(|val| val*div_bs)
}

pub fn calc_variance(batch: &DMatrix<f64>, batch_size: usize) -> DMatrix<f64> {
    assert_ne!(batch_size, 0);
    let mu = &calc_mean(batch, batch_size);
    let broad_mu = matrix_manipulations::broadcast(mu, batch_size);
    let div_bs = 1.0/(batch_size as f64);
    matrix_manipulations::sum_each_row(&(batch-broad_mu).map(|val| val.powi(2)*div_bs))
}

#[cfg(test)]
mod tests {
    use super::*;
    use na::DMatrix;
    use approx::assert_abs_diff_eq;  // For floating-point comparisons

    #[test]
    fn test_calc_mean() {
        let batch_size = 2;
        let channels = 3;
        let batch = DMatrix::from_row_slice(channels, batch_size, &[
            1.0, 2.0,
            3.0, 4.0,
            5.0, 6.0
        ]);
        let result = calc_mean(&batch, batch_size);
        let expected = DMatrix::from_row_slice(channels, 1, &[
            1.5,  // Mean of (1.0, 2.0)
            3.5,  // Mean of (3.0, 4.0)
            5.5   // Mean of (5.0, 6.0)
        ]);
        assert_abs_diff_eq!(result, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_calc_variance() {
        let batch_size = 2;
        let channels = 3;
        let batch = DMatrix::from_row_slice(channels, batch_size, &[
            1.0, 2.0,
            3.0, 4.0,
            5.0, 6.0
        ]);
        let result = calc_variance(&batch, batch_size);
        let expected = DMatrix::from_row_slice(channels, 1, &[
            0.25,  // Variance of (1.0, 2.0)
            0.25,  // Variance of (3.0, 4.0)
            0.25   // Variance of (5.0, 6.0)
        ]);
        assert_abs_diff_eq!(result, expected, epsilon = 1e-10);
    }
}
