use nalgebra::DMatrix;

pub struct BackpropCache {
    pub d_z: DMatrix<f64>,
    pub d_a: DMatrix<f64>
}

impl BackpropCache {
    pub fn new(d_z: DMatrix<f64>, d_a: DMatrix<f64>) -> BackpropCache{
        return BackpropCache {
            d_a: d_a,
            d_z: d_z
        };
    }
}