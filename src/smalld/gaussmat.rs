//! dimension reduction using multiplication by full gaussion matrix
//! Used in cunjunction with rhst when we do not need sparsity

use num_traits::float::Float;

use ndarray::{Array1, Array2, ArrayView1};
use ndarray_rand::rand_distr::{Distribution, StandardNormal};
use ndarray_rand::RandomExt;

use rayon::prelude::*;

use super::reducer::Reducer;
#[cfg_attr(doc, katexit::katexit)]
/// Dimension reduction using multiplication  by  a full  $N(0,1)$ matrix
pub struct GaussianMat<T: Float> {
    to_dim: usize,
    //
    _from_dim: usize,
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
            _from_dim: from_dim,
            gauss,
        }
    }
} // end of impl

impl<T> Reducer<T> for GaussianMat<T>
where
    T: 'static + Send + Sync + Float,
{
    /// reduce dimension of data, returning reduced data
    fn reduce(&self, data: &[&[T]]) -> Vec<Vec<T>> {
        //
        let reduce_item = |item: &[T]| -> Vec<T> {
            let f: T = T::from(self.to_dim).unwrap().sqrt();
            let small_item = (0..self.to_dim)
                .map(|i| {
                    self.gauss
                        .row(i)
                        .dot(&ArrayView1::from_shape(item.len(), item).unwrap())
                        / f
                })
                .collect();
            small_item
        };
        //
        let reduced_data = data
            .par_iter()
            .map(|item| reduce_item(item))
            .collect::<Vec<Vec<T>>>();

        reduced_data
    }

    // array interface
    fn reduce_a(&self, data: &[&Array1<T>]) -> Vec<Array1<T>> {
        //
        let reduce_item = |item: &[T]| -> Array1<T> {
            let f: T = T::from(self.to_dim).unwrap().sqrt();
            let small_item = (0..self.to_dim)
                .map(|i| {
                    self.gauss
                        .row(i)
                        .dot(&ArrayView1::from_shape(item.len(), item).unwrap())
                        / f
                })
                .collect();
            small_item
        };
        //
        let reduced_data = data
            .par_iter()
            .map(|item| reduce_item(item.as_slice().unwrap()))
            .collect::<Vec<Array1<T>>>();
        //
        reduced_data
    }
} // end of impl Reduce

mod tests {
    #![allow(unused)]

    use super::*;

    use rand::distributions::Uniform;
    use rand::prelude::*;
    use rand_xoshiro::Xoshiro256PlusPlus;

    fn log_init_test() {
        let _ = env_logger::builder().is_test(true).try_init();
    }

    // compare norm ration
    fn norm_ratio<T: Float + From<T>>(reduced: &[T], origin: &[T]) -> f64 {
        let reduced_norm = reduced
            .iter()
            .fold(T::zero(), |acc, x| acc + *x * *x)
            .sqrt();
        let origin_norm = origin.iter().fold(T::zero(), |acc, x| acc + *x * *x).sqrt();
        //
        <f64 as num_traits::NumCast>::from(reduced_norm / origin_norm).unwrap()
    }

    #[test]
    fn check_gauss_reducer() {
        log_init_test();
        log::info!("in test_uniform_random");
        //
        let from_dim = 500;
        let to_dim = 50;
        let width: f64 = 10.;
        let unif_01 = Uniform::<f64>::new(0., 1.);
        let unif_range = Uniform::<f64>::new(0., width);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(234567_u64);
        let nb_test = 1000_000;

        let data_large: Vec<Vec<f64>> = (0..nb_test)
            .map(|_| {
                let m = unif_range.sample(&mut rng);
                let p: Vec<f64> = (0..from_dim)
                    .map(|_| m * unif_01.sample(&mut rng))
                    .collect();
                return p;
            })
            .collect();
        // reduce dim
        let to_reduce: Vec<&[f64]> = data_large.iter().map(|v| v.as_slice()).collect();
        let mat = GaussianMat::<f64>::new(from_dim, to_dim);
        let reduced = mat.reduce(&to_reduce);
        //
        // now compare norms
        let mut mean = 0.0_f64;
        let mut sample = Vec::<f64>::with_capacity(nb_test);
        for i in 0..nb_test {
            let ratio = norm_ratio(&reduced[i], &to_reduce[i]);
            sample.push(ratio);
            mean += ratio;
            log::debug!("ratio : {:.3e}", ratio);
        }
        mean /= nb_test as f64;
        let mut var = sample
            .iter()
            .fold(0., |acc, x| acc + (x - mean) * (x - mean));
        var /= nb_test as f64;
        //
        log::info!("mean ratio : {:.3e}, sigma : {:.3e}", mean, var.sqrt());
    } // end check_gauss_reducer

    // use RUST_LOG=info cargo test check_gauss_reducer_a -- --nocapture
    #[test]
    fn check_gauss_reducer_a() {
        //
        log_init_test();
        let from_dim = 500;
        let to_dim = 400;
        let width = 10.;
        //
        let mat = GaussianMat::<f64>::new(from_dim, to_dim);
        let nb_test: usize = 100_000;
        //
        let unif_01 = Uniform::<f64>::new(0., 1.);
        let unif_range = Uniform::<f64>::new(0., width);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(234567_u64);
        //
        let data: Vec<Array1<f64>> = (0..nb_test)
            .map(|_| unif_range.sample(&mut rng) * Array1::<f64>::random(from_dim, unif_01))
            .collect();
        let to_reduce: Vec<&Array1<f64>> = data.iter().map(|a| a).collect();
        let reduced = mat.reduce_a(&to_reduce);
        // now compare norms
        let mut mean = 0.0_f64;
        let mut sample = Vec::<f64>::with_capacity(nb_test);
        for i in 0..nb_test {
            let ratio = norm_ratio(
                &reduced[i].as_slice().unwrap(),
                &to_reduce[i].as_slice().unwrap(),
            );
            sample.push(ratio);
            mean += ratio;
            log::trace!("ratio : {:.3e}", ratio);
        }
        mean /= nb_test as f64;
        let mut var = sample
            .iter()
            .fold(0., |acc, x| acc + (x - mean) * (x - mean));
        var /= nb_test as f64;
        //
        log::info!("mean ratio : {:.3e}, sigma : {:.3e}", mean, var.sqrt());
    } // end of check_reducer_a
} // end of mod tests
