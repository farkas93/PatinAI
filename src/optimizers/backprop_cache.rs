use na::DMatrix;

pub struct BackpropCache {
    pub d_z: DMatrix<f64>,
    pub d_a: DMatrix<f64>,
    pub curr_iter: usize,
    pub learning_rate: f64,
    pub beta_1: f64,
    pub beta_2: f64,
    pub epsilon: f64,
    pub grad_step_fn: fn(&mut BackpropCache, &DMatrix<f64>, usize) -> DMatrix<f64>,
    pub vec_v: Option<Vec<DMatrix<f64>>>,
    pub vec_s: Option<Vec<DMatrix<f64>>>
}

impl BackpropCache {
    pub fn new(lr: f64, grad_step_fn: fn(&mut BackpropCache, &DMatrix<f64>, usize) -> DMatrix<f64>) -> BackpropCache{
        return BackpropCache {
            d_a: DMatrix::zeros(1, 1),
            d_z: DMatrix::zeros(1, 1),
            curr_iter: 0,
            learning_rate: lr,
            beta_1: 0.9,
            beta_2: 0.999,
            epsilon: 1e-8,
            grad_step_fn: grad_step_fn,
            vec_v: None,
            vec_s: None
        };
    }

    pub fn append_to_vec_v(&mut self, matrix: DMatrix<f64>) -> usize {
        if let Some(vec) = &mut self.vec_v {
            vec.push(matrix);
            vec.len()-1
        } else {
            self.vec_v = Some(vec![matrix]);
            0
        }
    }
    
    pub fn append_to_vec_s(&mut self, matrix: DMatrix<f64>) -> usize {
        if let Some(vec) = &mut self.vec_s {
            vec.push(matrix);
            vec.len()-1
        } else {
            self.vec_s = Some(vec![matrix]);
            0
        }
    }


    pub fn set_v(&mut self, matrix: DMatrix<f64>, ind: usize) {
        if let Some(vec) = &mut self.vec_v {
            if ind as usize >= vec.len() {
                // Handle out of bounds. Here, we do nothing, but you might want to handle differently.
                return;
            }
            vec[ind as usize] = matrix;
        }
        else{
            // vec_d_v is None 
            panic!("vec_d_v was not initialized!");
        }
    }
    
    pub fn set_s(&mut self, matrix: DMatrix<f64>, ind: usize) {
        if let Some(vec) = &mut self.vec_s {
            if ind as usize >= vec.len() {
                // Handle out of bounds. Here, we do nothing, but you might want to handle differently.
                return;
            }
            vec[ind as usize] = matrix;
        }
        else{
            // vec_d_s is None 
            panic!("vec_d_s was not initialized!");
        }
    }
    

    pub fn get_v(&self, ind: usize) -> &DMatrix<f64> {
        match &self.vec_v {
            Some(vec) => {
                if ind >= vec.len() {
                    panic!("Index out of bounds");
                }
                &vec[ind]
            },
            None => panic!("vec_d_v was not initialized!"),
        }
    }

    pub fn get_s(&self, ind: usize) -> &DMatrix<f64> {
        match &self.vec_s {
            Some(vec) => {
                if ind >= vec.len() {
                    panic!("Index out of bounds");
                }
                &vec[ind]
            },
            None => panic!("vec_d_s was not initialized!"),
        }
    }

    pub fn grad_step(&mut self, input: &DMatrix<f64>, bp_ind: usize) -> DMatrix<f64>{
        (self.grad_step_fn)(self, input, bp_ind)
    }
}