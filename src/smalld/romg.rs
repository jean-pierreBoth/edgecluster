//! implement Random Orthogonal Matrix generation based on the following papers:
//!
//! - Johnson-Lindenstrauss Transforms with Best Confidence Skorski 2021
//!     See [skorski](https://proceedings.mlr.press/v134/skorski21a/skorski21a.pdf)
//! - How to generate random matrices from the classic compact groups Mezzadri 2007
//!        See [mezzadri](https://arxiv.org/pdf/math-ph/0609050)

use std::f32::consts::LN_10;

use lax::layout;
use num_traits::cast::*;
use num_traits::float::Float;

use ndarray::{Array1, Array2, ArrayView, ArrayView1};
use ndarray_rand::rand_distr::{Distribution, StandardNormal};
use ndarray_rand::RandomExt;

use rayon::prelude::*;

use lax::{layout::MatrixLayout, JobSvd, Lapack};

use super::reducer::Reducer;

#[cfg_attr(doc, katexit::katexit)]
/// Dimension reduction by orthogonal matrix multiplication.
/// The basic idea to generate a random $N(0,1)$ matrix and do a QR decomposition is flawed
/// as explained in [mezzadri](https://proceedings.mlr.press/v134/skorski21a/skorski21a.pdf).
/// Then the problem is to get an unbiaised sampling of a random orthogonal matrix with a good guarantee
/// of distance preservation
///
pub struct Romg<T: Float> {
    to_dim: usize,
    //
    from_dim: usize,
    //
    orthogonal: Array2<T>,
}

impl<T> Romg<T>
where
    T: 'static + Float + Send + Sync,
{
    pub fn new(from_dim: usize, to_dim: usize) -> Self {
        assert!(to_dim < from_dim);
        // generate a (from_dim, from_dim) random orthogonal matrix
        //
        panic!("not yet implemented");
    } // end of new
} // end of impl Romg

// here we implement Mezzadri algorithm
fn generate_romg<T>(n: usize) -> Array2<T>
where
    StandardNormal: Distribution<T>,
    T: Float + Lapack + ndarray::ScalarOperand,
{
    let mut gauss: Array2<T> = Array2::<T>::random((n, n), StandardNormal);
    // do a QR decomposition
    let layout = MatrixLayout::C {
        row: gauss.nrows() as i32,
        lda: gauss.nrows() as i32,
    };
    let l_res = T::qr(layout, gauss.as_slice_mut().unwrap());
    if l_res.is_err() {
        log::error!("generate_romg : a lapack error occurred in F::householder");
        panic!();
    }
    // now we have Qt in place of a and L in l_res, we get diagonal terms of L
    panic!("not yet implemented");
}
