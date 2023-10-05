extern crate nalgebra as na;
use na::{DMatrix, DVector};

mod layers;
use layers::layer::Layer;
use layers::linear::LinearLayer;
use layers::sigmoid::SigmoidLayer;

mod models;
use models::logistic_regression;

mod optimizers;
use optimizers::gradient_descent;


fn main() {
    let rows = 3;
    let cols = 3;
    let batch_size = 2; // This will be m

    let input = DMatrix::from_vec(rows, batch_size, vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    let w =  DMatrix::from_vec(rows, cols, vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    let b =  DVector::from_vec(vec![-3.0, -2.0, -4.0]);
    let mut lin_layer = LinearLayer::new(rows, cols, batch_size);
    lin_layer.set(w, b);
    let mut sigmoid = SigmoidLayer::new(1, 1);
    
    
    let mut x = lin_layer.forward(&input);
    x = sigmoid.forward(x);
    println!("Output vector: {:?}", x);
    
}
