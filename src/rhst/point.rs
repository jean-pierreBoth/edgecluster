//! defines data description

use num_traits::cast::*;
use num_traits::float::Float;

use std::fmt::Debug;

type NodeId = usize;

/// data to cluster identifier
pub type PointId = usize;

#[derive(Debug, Clone)]
pub struct Point<T> {
    // id to identify points as coming from external client
    id: PointId,
    /// data point
    p: Vec<T>,
    /// original label
    label: u32,
}

impl<T> Point<T>
where
    T: Float + Debug,
{
    pub fn new(id: PointId, p: Vec<T>, label: u32) -> Self {
        Point { id, p, label }
    }
    /// get the original label
    pub fn get_label(&self) -> u32 {
        self.label
    }

    /// gets the points coordinate
    pub fn get_position(&self) -> &[T] {
        &self.p
    }

    /// get minima and maxima of coordinates over all dimensions
    pub fn get_minmax(&self) -> (T, T) {
        self.p
            .iter()
            .fold((T::max_value(), T::min_value()), |acc, x| {
                (acc.0.min(*x), acc.0.max(*x))
            })
    }

    pub fn get_dimension(&self) -> usize {
        self.p.len()
    }
} // end of impl Point