use nalgebra::DMatrix;
use crate::layers::relu::ReLULayer;
use crate::models::model::Model;
use crate::optimizers::optimizer::Optimizer;
use crate::layers::layer::Layer;
use crate::layers::sigmoid::SigmoidLayer;
use crate::layers::linear::LinearLayer;

pub struct FullyConnectedNeuralNet {
    layers: Vec<Box<dyn Layer>>,
    data_train: Vec<DMatrix<f64>>,
    data_validate: Vec<DMatrix<f64>>,
    labels_train: Vec<DMatrix<f64>>,
    labels_validate: Vec<DMatrix<f64>>,
    optimizer: Box<dyn Optimizer>,
}

impl Model for FullyConnectedNeuralNet {
    fn set_optimizer(&mut self, o: Box<dyn Optimizer>){
        self.optimizer = o;
    }

    fn train(&mut self) {
        let training_on = true;
        let num_iters = self.optimizer.get_num_iters();
        let epoch_size = num_iters/10;
        for i in 0..num_iters{
            // Train for an epoch
            for (batch, labels) in self.data_train.iter().zip(self.labels_train.iter()) {
                let predictions = Self::forward(&mut self.layers, batch, training_on);
                self.optimizer.loss(&predictions, labels);
                self.optimizer.optimize(&mut self.layers);
            }
            if i%epoch_size == 0 {
                // validate
                let mut sum_loss = 0.0;
                for (val_batch, val_labels) in self.data_validate.iter().zip(self.labels_validate.iter()) {
                    let predictions = Self::forward(&mut self.layers, val_batch, training_on);
                    sum_loss = sum_loss + self.optimizer.loss(&predictions, val_labels);
                }
                let val_loss = sum_loss/ (self.data_validate.len() as f64);
                println!("Epoch: {}; Loss after iteration {}: {}", (i/epoch_size)+1, i, val_loss);
            }
        }   
    }

    fn predict(&mut self,  x: &DMatrix<f64>) -> DMatrix<f64> {
        return Self::binary_decision_boundary(&Self::forward(&mut self.layers, x, false));
    }

}

impl FullyConnectedNeuralNet {
    fn new(x_train: Vec<DMatrix<f64>>, x_val: Vec<DMatrix<f64>>, y_train: Vec<DMatrix<f64>>, y_val: Vec<DMatrix<f64>>, o: Box<dyn Optimizer>) -> FullyConnectedNeuralNet{
        let first_batch = x_train.first().unwrap();
        let batch_size = first_batch.ncols();
        let channels_in1 = first_batch.nrows();
        let channels_out1 = 8;
        let channels_in2 = channels_out1;
        let channels_out2 = 8;
        let channels_in3 = channels_out2;
        let channels_out3 = 1;

        let linear1 = LinearLayer::new(channels_in1, channels_out1, batch_size);
        let relu1 = ReLULayer::new();
        let linear2 = LinearLayer::new(channels_in2, channels_out2, batch_size);
        let relu2 = ReLULayer::new();
        let linear3 = LinearLayer::new(channels_in3, channels_out3, batch_size);
        let sigmoid = SigmoidLayer::new();
        let mut layers: Vec<Box<dyn Layer>> = Vec::new();
    
        // Add instances of LinearLayer and SigmoidLayer
        layers.push(Box::new(linear1));
        layers.push(Box::new(relu1));
        layers.push(Box::new(linear2));
        layers.push(Box::new(relu2));
        layers.push(Box::new(linear3));
        layers.push(Box::new(sigmoid));

        return FullyConnectedNeuralNet {
            layers: layers,
            data_train: x_train,
            data_validate: x_val,
            labels_train: y_train,
            labels_validate: y_val,
            optimizer: o,
        };
    }
    
    fn binary_decision_boundary(x: &DMatrix<f64>) -> DMatrix<f64>{        
        x.map(|v| if v > 0.5 { 1.0 } else { 0.0 })
    }

    fn cap_outputs(x: &DMatrix<f64>) -> DMatrix<f64> {        
        x.map(|v| if v < f64::EPSILON { f64::EPSILON } else { if v > (1.0-f64::EPSILON) {1.0-f64::EPSILON} else { v } })
    }

    fn forward(layers: &mut Vec<Box<dyn Layer>>, x: &DMatrix<f64>, training: bool) -> DMatrix<f64> {
        
        let mut out = x;
        for layer in layers.iter_mut() {
            out = layer.forward(out);
        }
        if training {
            Self::cap_outputs(out)
        }
        else {            
            println!("Pre decision vector: {:?}", out);
            Self::binary_decision_boundary(out)
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use crate::optimizers::gradient_descent::GradientDescent;


    #[test]
    fn test_create() {
        let x =  DMatrix::from_vec(2, 11, vec![1.0, 1.0, 
                                                    1.0, 0.0, 
                                                    1.0, 1.0,
                                                    0.0, 1.0,
                                                    1.0, 1.0,
                                                    0.0, 1.0,
                                                    0.0, 0.0,
                                                    1.0, 1.0,
                                                    0.0, 0.0,
                                                    0.0, 1.0,
                                                    1.0, 1.0, ]);
        let y = DMatrix::from_vec(1, 11, vec![1.0, 
                                            0.0, 
                                            1.0,
                                            0.0,
                                            1.0, 
                                            0.0,
                                            0.0,
                                            1.0, 
                                            0.0,
                                            0.0,
                                            1.0]);

        let optimizer = Box::new(GradientDescent::new(1000, 0.01));
        let mut reg = FullyConnectedNeuralNet::new(vec![x.clone()], 
                                                                vec![x.clone()], 
                                                                vec![y.clone()], 
                                                                vec![y.clone()], 
                                                                optimizer);

        reg.train();
        let predictions = reg.predict(&x);
        
        println!("Output vector: {:?}", predictions);
        println!("Solution vector: {:?}", y);
        
        // Check dimensions of weights and biases
        assert_eq!(predictions.nrows(), y.nrows());
        assert_abs_diff_eq!(predictions, y, epsilon = 1.0e-15);

    }
}