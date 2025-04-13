//! construct agglomerative cluster from rhst2
//!
//!

use cpu_time::ProcessTime;
use rand_distr::{Distribution, StandardNormal};
use std::time::{Duration, SystemTime};

use lax::Lapack;
// use ndarray_rand::rand_distr::{Distribution, StandardNormal};

// points are assimilated to cells of layer 0. Most cells of layer 0 should have one or very few points.
// The algorithms is translated
use num_traits::Float;

use super::point::*;
use super::rhst2::*;
use crate::smalld::*;

//==========================

pub struct Hcluster<'a, T> {
    debug_level: usize,
    //
    points: Vec<&'a Point<T>>,
    //
    mindist: f64,
    //
    reducer: Option<&'a dyn reducer::Reducer<T>>,
    // if we (must) reduce points dimension,
    reduced_points: Option<Vec<Point<T>>>,
}

impl<'a, T> Hcluster<'a, T>
where
    T: Float + std::fmt::Debug + Sync + Send + Lapack + ndarray::ScalarOperand,
    StandardNormal: Distribution<T>,
{
    /// This algorithm requires the data to have a small dimension (<= ~10).  
    /// It is possible to specify an algorithm to reduce data dimension and a target dimension (See [smalld](crate::smalld)).  
    /// If not, the algorithm will try to choose one.
    pub fn new(points: Vec<&'a Point<T>>, reducer: Option<&'a dyn reducer::Reducer<T>>) -> Self {
        //
        log::info!("entering Hcluster::new");
        // construct space
        let (xmin, xmax) = points
            .iter()
            .map(|p| p.get_minmax())
            .fold((T::max_value(), T::min_value()), |acc, x| {
                (acc.0.min(x.0), acc.1.max(x.1))
            });
        //
        let xmin: f64 = xmin.to_f64().unwrap();
        let xmax: f64 = xmax.to_f64().unwrap();
        // construct spacemesh
        let dim = points[0].get_dimension();

        println!(
            "space dimension : {}, xmin : {:.3e}, xmax : {:.3e}",
            dim, xmin, xmax
        );
        log::info!(
            "space dimension : {}, xmin : {:.3e}, xmax : {:.3e}",
            dim,
            xmin,
            xmax
        );
        Hcluster {
            debug_level: 0,
            points,
            mindist: 0.,
            reducer,
            reduced_points: None,
        }
    } // end of new

    pub fn set_debug_level(&mut self, level: usize) {
        self.debug_level = level;
    }

    pub fn cluster(&mut self, mindist: f64, nb_cluster: usize) {
        // construct space
        self.mindist = mindist;
        let (xmin, xmax) = self
            .points
            .iter()
            .map(|p| p.get_minmax())
            .fold((T::max_value(), T::min_value()), |acc, x| {
                (acc.0.min(x.0), acc.0.max(x.1))
            });
        //
        let xmin: f64 = xmin.to_f64().unwrap();
        let xmax: f64 = xmax.to_f64().unwrap();
        let dim = self.points[0].get_dimension();
        log::debug!("dim : {} xmin : {:.3e}, xmax : {:.3e}", dim, xmin, xmax);
        // TODO: do we need to keep points in HCluster (we clone a vec of references)
        let points_to_cluster: Vec<&Point<T>>;
        if dim > 10 {
            let to_dim = 10;
            log::info!("reducing dimension from : {} to : {}", dim, to_dim);
            // we reduce dimension
            self.reduced_points = Some(self.reduce_points(to_dim));
            points_to_cluster = self
                .reduced_points
                .as_ref()
                .unwrap()
                .iter()
                .collect::<Vec<&Point<T>>>();
        } else {
            points_to_cluster = self.points.clone();
        }
        // we have points_to_cluster , we can construct space
        let (xmin, xmax) = points_to_cluster
            .iter()
            .map(|p| p.get_minmax())
            .fold((T::max_value(), T::min_value()), |acc, x| {
                (acc.0.min(x.0), acc.0.max(x.1))
            });
        //
        let xmin: f64 = xmin.to_f64().unwrap();
        let xmax: f64 = xmax.to_f64().unwrap();
        let dim = points_to_cluster[0].get_dimension();
        log::info!("dim : {}, xmin : {:.3e}, xmax : {:.3e}", dim, xmin, xmax);
        // construct spacemesh
        let mut space = Space::new(dim, xmin, xmax, mindist);
        let mut spacemesh = SpaceMesh::new(&mut space, points_to_cluster);
        spacemesh.embed();

        if self.debug_level > 0 {
            spacemesh.dump_layer(0, self.debug_level);
        }
        //
        spacemesh.summary();

        let filtered_benefits = spacemesh.compute_benefits();
        /*         // now we extract best subtrees from benefits in mesh
        let mut best_tree = BestTree::new(spacemesh.get_nb_layers(), &spacemesh);
        best_tree.get_benefits(&benefits);

        let filtered_benefits = best_tree.get_filtered_benefits(); */

        log::info!("dump of filtered_benefits");
        dump_benefits(&spacemesh, &filtered_benefits);
        check_partition(&spacemesh, &filtered_benefits);
        //
        // we have benefits, we can try to cluster
        //
        let _clusters = spacemesh.get_partition(nb_cluster, &filtered_benefits);
    }

    // return a vector of points with reduced data dimension, label and id preserved
    fn reduce_points(&self, to_dim: usize) -> Vec<Point<T>> {
        let from_dim = self.points[0].get_dimension();
        //
        let to_reduce = self
            .points
            .iter()
            .map(|p| p.get_position())
            .collect::<Vec<&[T]>>();
        //
        let reduced_data = match self.reducer {
            Some(reducer) => reducer.reduce(&to_reduce),
            _ => {
                let reducer = Romg::<T>::new(from_dim, to_dim);
                reducer.reduce(&to_reduce)
            }
        };
        let reduced_points: Vec<Point<T>> = self
            .points
            .iter()
            .enumerate()
            .map(|(i, p)| Point::new(p.get_id(), reduced_data[i].clone(), p.get_label()))
            .collect();
        //
        reduced_points
    }
} // end impl Hcluster

//========================================================

#[cfg(test)]
mod tests {

    use super::*;

    use rand::distr::Uniform;
    use rand::prelude::*;
    use rand_xoshiro::Xoshiro256PlusPlus;

    use rand_distr::{Distribution, Exp};

    fn log_init_test() {
        let _ = env_logger::builder().is_test(true).try_init();
    }

    #[test]
    fn test_cluster_random() {
        log_init_test();
        log::info!("in test_cluster_random");
        // points are generated around 5 centers/labels
        let nbvec = 100usize;
        let dim = 2;
        let width: f64 = 1.;
        let mindist = 5.;
        let unif_01 = Uniform::<f64>::new(0., width).unwrap();
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(234567_u64);
        let mut points: Vec<Point<f64>> = Vec::with_capacity(nbvec);
        for i in 0..nbvec {
            let label = i % 5;
            let offset = label as f64 * 15.;
            let p: Vec<f64> = (0..dim)
                .map(|_| offset + unif_01.sample(&mut rng))
                .collect();
            points.push(Point::<f64>::new(i, p, (i % 5).try_into().unwrap()));
        }
        let refpoints: Vec<&Point<f64>> = points.iter().map(|p| p).collect();
        //
        let mut hcluster = Hcluster::new(refpoints, None);
        hcluster.set_debug_level(1);
        hcluster.cluster(mindist, 5);
        //
    } //end of test_cluster_random

    #[test]
    fn test_cluster_exp() {
        log_init_test();
        log::info!("in test_uniform_random");
        //
        let nbvec = 10_000usize;
        let dim = 5;
        let width: f64 = 100.;
        let mindist = 5.;

        // sample with coordinates following exponential law
        let law = Exp::<f32>::new(50. / width as f32).unwrap();
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(234567_u64);
        let mut points: Vec<Point<f32>> = Vec::with_capacity(nbvec);
        for i in 0..nbvec {
            let p: Vec<f32> = (0..dim)
                .map(|_| law.sample(&mut rng).min(width as f32))
                .collect();
            points.push(Point::<f32>::new(i, p, (i % 5).try_into().unwrap()));
        }
        let refpoints: Vec<&Point<f32>> = points.iter().map(|p| p).collect();
        // Space definition
        //
        let mut hcluster = Hcluster::new(refpoints, None);
        hcluster.cluster(mindist, 5);
        //
    } //end of test_cluster_exp
} // end of tests
