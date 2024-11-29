//! implementation of Dimension Reduction based on the following papers:
//!
//! - Johnson-Lindenstrauss Transforms with Best Confidence Skorski 2021
//!     See [skorski](https://proceedings.mlr.press/v134/skorski21a/skorski21a.pdf)
//! - How to generate random matrices from the classic compact groups Mezzadri 2007
//!        See [mezzadri](https://arxiv.org/pdf/math-ph/0609050)

use std::f32::consts::LN_10;

use lax::layout;
use log::log_enabled;
use num_traits::cast::*;
use num_traits::float::Float;

use ndarray::{Array1, Array2, ArrayView, ArrayView1, Axis};
use ndarray_rand::rand_distr::{Distribution, StandardNormal};
use ndarray_rand::RandomExt;

use rayon::prelude::*;

use lax::{layout::MatrixLayout, JobSvd, Lapack};

use super::reducer::Reducer;

#[cfg_attr(doc, katexit::katexit)]
/// Dimension reduction by orthogonal matrix multiplication.
/// The basic idea to generate a random $N(0,1)$ matrix and do a QR decomposition is in fact flawed (!!)
/// as explained in [mezzadri](https://proceedings.mlr.press/v134/skorski21a/skorski21a.pdf).  
/// The problem is to get an unbiaised sampling of a random orthogonal matrix to be used in Skorski algorithm
/// to ensure distance preservation
///
pub struct Romg<T: Float> {
    to_dim: usize,
    //
    from_dim: usize,
    //
    mat_reducer: Array2<T>,
}

impl<T> Romg<T>
where
    T: 'static + Float + NumCast + Lapack + ndarray::ScalarOperand + Send + Sync,
    StandardNormal: Distribution<T>,
{
    pub fn new(from_dim: usize, to_dim: usize) -> Self {
        assert!(to_dim < from_dim);
        // generate a (from_dim, from_dim) random orthogonal matrix
        let v = generate_romg::<T>(from_dim);
        // generate a (to_dim, to_dim) random orthogonal matrix
        let u = generate_romg::<T>(to_dim);
        // estimate lambda (take  mean or median for Beta(to/2, (from - to)/2 )) with to to >=2 from > to
        let lambda: T = T::from((to_dim as f64 / from_dim as f64).sqrt()).unwrap();
        //
        let mut proj = Array2::<T>::zeros((to_dim, from_dim));
        for i in 0..to_dim {
            proj[[i, i]] = T::one();
        }
        //
        let tmp = proj.dot(&v.t());
        let mat_reducer = u.dot(&tmp) / lambda;
        //
        Romg {
            to_dim,
            from_dim,
            mat_reducer,
        }
    } // end of new
} // end of impl Romg

// here we implement Mezzadri algorithm
// As our matrices are Float (and not Complex) the diagonal matrix Λ of Mezzadri is made of values 1. andf -1. and so is its own inverse.
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
    //
    let lower_try = Array2::from_shape_vec((gauss.nrows(), gauss.nrows()), l_res.unwrap());
    if lower_try.is_err() {
        panic!("generate_romg : could not extract lower")
    }
    let lower: ndarray::ArrayBase<ndarray::OwnedRepr<T>, ndarray::Dim<[usize; 2]>> =
        lower_try.unwrap();

    // now we have Qt in place of a and L (Λ in paper) in l_res, we get sign of diagonal terms of L
    let mut signs = Array1::<T>::ones(gauss.nrows());
    let mut nb_flip = 0;
    for i in 0..gauss.nrows() {
        if lower[[i, i]] < T::zero() {
            signs[[i]] = -T::one();
            nb_flip += 1;
        }
    }
    log::info!("nb flip sign : {}", nb_flip);
    // Now Qt = Q * Λ, R =  Λ^(-1) * t(Λ). We just have to multiply Q by Λ i.e multiply columns of Q by signs
    for (i, mut row) in gauss.axis_iter_mut(Axis(0)).enumerate() {
        // Perform calculations and assign to `row`; this is a trivial example:
        row.iter_mut().enumerate().map(|(j, x)| *x *= signs[j]);
    }
    //
    gauss
}

//================

impl<T> Reducer<T> for Romg<T>
where
    T: 'static + Float + Send + Sync,
{
    //
    fn reduce(&self, data: &[&Vec<T>]) -> Vec<Vec<T>> {
        //
        let reduce_item = |v: &Vec<T>| -> Vec<T> {
            let v = self
                .mat_reducer
                .dot(&ArrayView1::from_shape((v.len()), v).unwrap());
            v.to_vec()
        };

        let reduced = data
            .par_iter()
            .map(|item| reduce_item(item))
            .collect::<Vec<Vec<T>>>();
        //
        reduced
    }

    fn reduce_a(&self, data: &[&Array1<T>]) -> Vec<Array1<T>> {
        //
        let reduce_item = |v: &Array1<T>| -> Array1<T> {
            let v = self.mat_reducer.dot(v);
            v
        };

        let reduced = data
            .par_iter()
            .map(|item| reduce_item(item))
            .collect::<Vec<Array1<T>>>();
        //
        reduced
    }
}

//====================

fn check_orthogonality<T>(mat: &Array2<T>)
where
    T: 'static + Float + NumCast + std::fmt::Debug + std::fmt::LowerExp,
{
    assert_eq!(mat.ncols(), mat.nrows());
    //
    let epsil: T = T::from(1.0e-5_f64).unwrap();
    let mut id = mat.t().into_owned();
    id = id.dot(mat);
    //
    log::debug!("input mat = {:?}", id);
    //
    for i in 0..mat.nrows() {
        for j in 0..mat.ncols() {
            if i == j {
                if (id[[i, j]] - T::one()).abs() > epsil {
                    log::error!(" i : {} , j : {} , val : {:?}", i, j, id[[i, j]]);
                }
            } else if (id[[i, j]]).abs() > epsil {
                log::error!(" i : {} , j : {} , val : {:?}", i, j, id[[i, j]]);
            }
        }
    }
}

//======================================

mod tests {

    use super::*;

    use rand::distributions::Uniform;
    use rand::prelude::*;
    use rand_xoshiro::Xoshiro256PlusPlus;

    use rand_distr::{Distribution, Exp};

    fn log_init_test() {
        let _ = env_logger::builder().is_test(true).try_init();
    }

    // compare norm ration
    fn check_isometry<T>(reduced: &Array1<T>, origin: &Array1<T>) -> f64 {
        panic!("not yet implemented");
    }

    #[test]
    fn chech_generate_romg() {
        log_init_test();
        //
        let q = generate_romg::<f64>(2);
        check_orthogonality(&q);
        //
        let q = generate_romg::<f64>(3);
        check_orthogonality(&q);

        let q = generate_romg::<f64>(4);
        check_orthogonality(&q);
    } // end of chech_generate_romg

    #[test]
    fn check_reducer() {
        //
        log_init_test();
        let n = 200;
        let mut gauss: Array1<f64> = Array1::<f64>::random((n), StandardNormal);
        let q = generate_romg::<f64>(5);

        panic!("not yet implemented");
    }
}
