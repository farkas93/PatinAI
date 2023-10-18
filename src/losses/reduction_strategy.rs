pub enum ReductionStrategy {
    SumOverBatchSize, // divides the total loss by the batch size (default)
    Sum, // Simply sums up the loss of all samples of the batch
    //None // Returns for each sample in the batcht the loss individually
}