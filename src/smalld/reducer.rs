//! trait for dimension reduction

pub trait Reducer<T> {
    fn reduce(&self, data: &[&Vec<T>]) -> Vec<Vec<T>>;
}
