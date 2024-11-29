//! dimension reduction using multiplication by full gaussion matrix
//! Used in cunjunction with rhst when we do not need sparsity

use num_traits::cast::*;
use num_traits::float::Float;

use ndarray::{Array1, Array2, ArrayView, ArrayView1};
use ndarray_rand::rand_distr::{Distribution, StandardNormal};
use ndarray_rand::RandomExt;

use rayon::prelude::*;

use super::reducer::Reducer;
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
} // end of impl

impl<T> Reducer<T> for GaussianMat<T>
where
    T: 'static + Send + Sync + Float,
{
    /// reduce dimension of data, returning reduced data
    fn reduce(&self, data: &[&Vec<T>]) -> Vec<Vec<T>> {
        //
        let mut reduced_data = Vec::<Vec<T>>::with_capacity(data.len());
        //
        let reduce_item = |item: &Vec<T>| -> Vec<T> {
            let f: T = T::from(self.to_dim).unwrap().sqrt();
            let small_item = (0..self.to_dim)
                .map(|i| {
                    (self
                        .gauss
                        .row(i)
                        .dot(&ArrayView1::from_shape((item.len()), item).unwrap())
                        / f)
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

    fn reduce_a(&self, data: &[&Array1<T>]) -> Vec<Array1<T>> {
        panic!("not yet implemented");
    }
} // end of impl Reduce

mod tests {

    use super::*;

    use rand::distributions::Uniform;
    use rand::prelude::*;
    use rand_xoshiro::Xoshiro256PlusPlus;

    use rand_distr::{Distribution, Exp};

    fn log_init_test() {
        let _ = env_logger::builder().is_test(true).try_init();
    }

    #[test]
    fn check_norm() {
        log_init_test();
        log::info!("in test_uniform_random");
        //
        let nbvec = 10_000usize;
        let dim = 200;
        let width: f64 = 1000.;
        let unif_01 = Uniform::<f64>::new(0., 1.);
        let unif_range = Uniform::<f64>::new(0., width);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(234567_u64);
        let nb_vec = 1000;

        let data_large: Vec<Vec<f64>> = (0..nb_vec)
            .map(|_| {
                let m = unif_range.sample(&mut rng);
                let p: Vec<f64> = (0..dim).map(|_| m * unif_01.sample(&mut rng)).collect();
                return p;
            })
            .collect();
        // reduce dim
        let data_ref: Vec<&Vec<f64>> = data_large.iter().map(|v| v).collect();
        let mat = GaussianMat::<f64>::new(dim, 15);
        let data_small = mat.reduce(&data_ref);
        //
        let norm = |v: &[f64]| -> f64 { v.iter().fold(0., |acc, x| acc + (*x) * (*x)).sqrt() };
        //
        let mut ratio = 0.0;
        for i in 0..nb_vec {
            ratio += (norm(&data_large[i]) - norm(&data_small[i])) / norm(&data_large[i]);
        }
        log::info!("norm ratio : {:.e}", ratio / nb_vec as f64);
    } // end check_norm
} // end of mod tests
