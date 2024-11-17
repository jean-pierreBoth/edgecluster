//! dimension reduction using multiplication by full gaussion matrix
//! Used in cunjunction with rhst when we do not need sparsity

use num_traits::cast::*;
use num_traits::float::Float;

use ndarray::{Array1, Array2, ArrayView, ArrayView1};
use ndarray_rand::rand_distr::{Distribution, StandardNormal};
use ndarray_rand::RandomExt;

use rayon::prelude::*;

#[cfg_attr(doc, katexit::katexit)]
/// Dimension reduction using multiplication  by  a full  $N(0,1)$ matrix
pub struct GaussianMat<T: Float> {
    to_dim: usize,
    //
    from_dim: usize,
    //
    gauss: Array2<T>,
}

impl<T: Send + Sync + Float + 'static> GaussianMat<T>
where
    StandardNormal: Distribution<T>,
{
    pub fn new(from_dim: usize, to_dim: usize) -> Self {
        assert!(to_dim < from_dim);
        let gauss: Array2<T> = Array2::<T>::random((to_dim, from_dim), StandardNormal);
        GaussianMat {
            to_dim,
            from_dim,
            gauss,
        }
    }

    ///
    pub fn reduce(&self, data: &[&Vec<T>]) -> Vec<Vec<T>> {
        //
        let mut reduced_data = Vec::<Vec<T>>::with_capacity(data.len());
        //
        let reduce_item = |item: &Vec<T>| -> Vec<T> {
            let small_item = (0..self.to_dim)
                .map(|i| {
                    self.gauss
                        .row(i)
                        .dot(&ArrayView1::from_shape((item.len()), &item).unwrap())
                })
                .collect();
            small_item
        };
        //
        let reduced_data = data
            .par_iter()
            .map(|item| reduce_item(*item))
            .collect::<Vec<Vec<T>>>();
        //
        reduced_data
    }
} // end of impl
