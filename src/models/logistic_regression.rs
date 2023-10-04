use nalgebra::DMatrix;
use crate::models::model::Model;
use crate::optimizers::optimizer::Optimizer;
use crate::layers::layer::Layer;
use crate::layers::sigmoid::SigmoidLayer;
use crate::layers::linear::LinearLayer;

pub struct LogisticRegression {
    layers: Vec<Box<dyn Layer>>,
    data_train: Vec<DMatrix<f64>>,
    data_validate: Vec<DMatrix<f64>>,
    labels_train: Vec<DMatrix<f64>>,
    labels_validate: Vec<DMatrix<f64>>,
    optimizer: Box<dyn Optimizer>,
}

impl Model for LogisticRegression {
    fn set_optimizer(&mut self, o: Box<dyn Optimizer>){
        self.optimizer = o;
    }

    fn train(&mut self) {
        for i in 0..self.optimizer.get_num_iters(){
            // Train for an epoch
            for (batch, labels) in self.data_train.iter().zip(self.labels_train.iter()) {
                let predictions = Self::forward(&mut self.layers, batch);
                let cost = self.optimizer.loss(&predictions, labels);
                self.optimizer.optimize(&mut self.layers);
            }
            // validate
            for (val_batch, val_labels) in self.data_validate.iter().zip(self.labels_validate.iter()) {
                let predictions = Self::forward(&mut self.layers, val_batch);
                let val_cost = self.optimizer.loss(&predictions, val_labels);
                println!("Loss after iteration {}: {}", i, val_cost);
            }
        }   
    }

    fn predict(&mut self,  x: &DMatrix<f64>) -> DMatrix<f64> {
        return Self::forward(&mut self.layers, x);
    }

}

impl LogisticRegression {
    fn new(x_train: Vec<DMatrix<f64>>, x_val: Vec<DMatrix<f64>>, y_train: Vec<DMatrix<f64>>, y_val: Vec<DMatrix<f64>>, o: Box<dyn Optimizer>) -> LogisticRegression{
        let first_batch = x_train.first().unwrap();
        let batch_size = first_batch.nrows();
        let channels_in = first_batch.nrows();
        let channels_out = 1;
        let linear = LinearLayer::new(channels_in, channels_out, batch_size);
        let sigmoid = SigmoidLayer::new(1, batch_size);
        let mut layers: Vec<Box<dyn Layer>> = Vec::new();
    
        // Add instances of LinearLayer and SigmoidLayer
        layers.push(Box::new(linear));
        layers.push(Box::new(sigmoid));

        return LogisticRegression {
            layers: layers,
            data_train: x_train,
            data_validate: x_val,
            labels_train: y_train,
            labels_validate: y_val,
            optimizer: o,
        };
    }
    
    fn forward(layers: &mut Vec<Box<dyn Layer>>, x: &DMatrix<f64>) -> DMatrix<f64>{
        
        let mut out = x;
        for layer in layers.iter_mut() {
            out = layer.forward(out);
        }
        out.clone()
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use crate::optimizers::gradient_descent::GradientDescent;


    #[test]
    fn test_create() {
        let x =  DMatrix::from_vec(2, 10, vec![1.0, 1.0, 
                                                    1.0, 0.0, 
                                                    1.0, 1.0, 
                                                    1.0, 1.0,
                                                    0.0, 0.0,
                                                    0.0, 1.0,
                                                    0.0, 0.2,
                                                    1.0, 0.3,
                                                    0.3, 0.3,
                                                    0.8, 0.1]);
        let y = DMatrix::from_vec(1, 10, vec![1.0, 
                                            0.0, 
                                            1.0, 
                                            1.0,
                                            0.0,
                                            0.0,
                                            0.0,
                                            0.0,
                                            0.0,
                                            0.0]);

        let optimizer = Box::new(GradientDescent::new(100, 0.1));
        let mut reg = LogisticRegression::new(vec![x.clone()], vec![x.clone()], vec![y.clone()], vec![y.clone()], optimizer);

        reg.train();
        let predictions = reg.predict(&x);

        println!("Output vector: {:?}", predictions);
        println!("Solution vector: {:?}", y);
        
        // Check dimensions of weights and biases
        assert_abs_diff_eq!(predictions, y, epsilon = 1.0e-15);
    }
}
