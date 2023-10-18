use na::DMatrix;
use super::loss::Loss;

pub struct BinaryCrossentropyLoss {
    derivate: Option<DMatrix<f64> >
}

impl Loss for BinaryCrossentropyLoss {

    fn cost(&mut self, prediction: &DMatrix<f64>, ground_truth: &DMatrix<f64>) -> f64 {
        let batch_size = ground_truth.ncols() as f64;
        let log_res = prediction.map(|x| x.ln()); // Log(A)
        let counter_pred = prediction.map(|x| (1.0 - x)); 
        let counter_log_res = counter_pred.map(|x| x.ln()); // Log(1 - A)
        let counter_gt = ground_truth.map(|x| (1.0 - x)); // 1 - Y
        let logprobs = ground_truth.dot(&log_res) + counter_gt.dot(&counter_log_res);
        let cost = - logprobs / batch_size;

        self.derivate = Some(- (ground_truth.component_div(prediction) - counter_gt.component_div(&counter_pred)));
        return cost;
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

impl BinaryCrossentropyLoss {

    pub fn new() -> BinaryCrossentropyLoss {
        return BinaryCrossentropyLoss { derivate: None }
    }
    
}