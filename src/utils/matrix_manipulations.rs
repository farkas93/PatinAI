
use na::DMatrix;

pub fn sum_each_row(mat: &DMatrix<f64>) -> DMatrix<f64>{
    // Input from (n x m) shaped matrix the sum along the columns. 
    // Returns (n, 1)
    let sums_data:  Vec<f64> = (0..mat.nrows())
    .map(|idx| mat.row(idx).sum())
    .collect();
    let nrows = mat.nrows();
    let ncols = 1;
    DMatrix::from_vec(nrows, ncols, sums_data)
}

pub fn broadcast(mat: &DMatrix<f64>, batch_size: usize) -> DMatrix<f64> {
    let ones_vector = DMatrix::<f64>::from_element(1, batch_size, 1.0);
    &mat.clone() * ones_vector
}

#[cfg(test)]
mod tests {
    use super::*;
    use na::DMatrix;
    use approx::assert_abs_diff_eq;  // For floating-point comparisons

    #[test]
    fn test_sum_each_row() {
        let mat = DMatrix::from_row_slice(3, 3, &[
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0
        ]);
        let result = sum_each_row(&mat);
        let expected = DMatrix::from_row_slice(3, 1, &[
            6.0,
            15.0,
            24.0
        ]);
        assert_abs_diff_eq!(result, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_broadcast() {
        let mat =DMatrix::from_vec(2, 1, vec![
            1.0, 2.0
        ]);
        let result = broadcast(&mat, 2);
        let expected = DMatrix::from_vec(2, 2, vec![
            1.0, 2.0, 1.0, 2.0
        ]);
        assert_abs_diff_eq!(result, expected, epsilon = 1e-10);
    }
}
