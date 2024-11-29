//! trait for dimension reduction
//!
use ndarray::{Array1, ArrayView1};

pub trait Reducer<T> {
    // vector interface
    fn reduce(&self, data: &[&Vec<T>]) -> Vec<Vec<T>>;
    // array interface
    fn reduce_a(&self, data: &[&Array1<T>]) -> Vec<Array1<T>>;
}
