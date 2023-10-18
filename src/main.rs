extern crate nalgebra as na;
use na::{DMatrix};

mod layers;
use layers::layer::Layer;
use layers::linear::LinearLayer;
use layers::activations::sigmoid::SigmoidLayer;

mod models;
mod optimizers;
mod utils;
mod losses;


fn main() {
    let rows = 3;
    let cols = 3;
    let batch_size = 2; // This will be m

    let input = DMatrix::from_vec(rows, batch_size, vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    let w =  DMatrix::from_vec(rows, cols, vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    let b =  DMatrix::from_vec(cols, 1, vec![-3.0, -2.0, -4.0]);
    let mut lin_layer = LinearLayer::new(rows, cols, batch_size);
    lin_layer.set_w_and_b(w, b);
    let mut sigmoid = SigmoidLayer::new();
    
    
    let mut x = lin_layer.forward(input);
    x = sigmoid.forward(x);
    println!("Output vector: {:?}", x);
    
}
