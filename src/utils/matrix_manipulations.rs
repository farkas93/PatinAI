
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


pub fn sum_each_column(mat: &DMatrix<f64>) -> DMatrix<f64>{
    // Input from (n x m) shaped matrix the sum along the columns. 
    // Returns (n, 1)
    let sums_data:  Vec<f64> = (0..mat.ncols())
    .map(|idx| mat.column(idx).sum())
    .collect();
    let ncols = mat.ncols();
    let nrows = 1;
    DMatrix::from_vec(nrows, ncols, sums_data)
}

pub fn broadcast(mat: &DMatrix<f64>, batch_size: usize) -> DMatrix<f64> {
    let ones_vector = DMatrix::<f64>::from_element(1, batch_size, 1.0);
    &mat.clone() * ones_vector
}

pub fn vertical_broadcast(mat: &DMatrix<f64>, nrows: usize) -> DMatrix<f64> {
    let ones_vector = DMatrix::<f64>::from_element(nrows, 1, 1.0);
    ones_vector * &mat.clone() 
}

#[cfg(test)]
mod tests {
    use super::*;
    use na::DMatrix;
    use approx::assert_abs_diff_eq;  // For floating-point comparisons

    #[test]
    fn test_sum_each_row() {
        let nrows = 3;
        let ncols = 1;
        let mat = DMatrix::from_row_slice(nrows, nrows, &[
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0
        ]);
        let result = sum_each_row(&mat);
        let expected = DMatrix::from_row_slice(nrows, ncols, &[
            6.0,
            15.0,
            24.0
        ]);

        assert_eq!(result.nrows(), nrows);
        assert_eq!(result.ncols(), ncols);
        assert_abs_diff_eq!(result, expected, epsilon = 1e-10);
    }
    
    #[test]
    fn test_sum_each_column() {
        let nrows = 1;
        let ncols = 3;
        let mat = DMatrix::from_row_slice(ncols, ncols, &[
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0
        ]);
        let result = sum_each_column(&mat);
        let expected = DMatrix::from_row_slice(nrows, ncols, &[
            12.0,
            15.0,
            18.0
        ]);
        assert_eq!(result.nrows(), nrows);
        assert_eq!(result.ncols(), ncols);
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


    #[test]
    fn test_vertical_broadcast() {
        let mat =DMatrix::from_vec(1, 2, vec![
            1.0, 2.0
        ]);
        let result = vertical_broadcast(&mat, 2);
        let expected = DMatrix::from_vec(2, 2, vec![
            1.0, 1.0, 2.0, 2.0
        ]);
        assert_abs_diff_eq!(result, expected, epsilon = 1e-10);
    }
}
