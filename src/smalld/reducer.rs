//! trait for dimension reduction
//!
use ndarray::Array1;

pub trait Reducer<T> {
    /// vector interface
    fn reduce(&self, data: &[&[T]]) -> Vec<Vec<T>>;
    /// array interface
    fn reduce_a(&self, data: &[&Array1<T>]) -> Vec<Array1<T>>;
}
