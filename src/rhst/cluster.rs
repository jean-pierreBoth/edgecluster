//! construct agglomerative cluster from rhst2
//!

use dashmap::{iter, mapref::one, rayon::*, DashMap};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use rayon::prelude::*;

// points are assimilated to cells of layer 0. Most cells of layer 0 should have one or very few points.
// The algorithms is translated
use num_traits::{cast::AsPrimitive, Float};

use super::point::*;
use super::rhst2::*;

/// base unit for counting cost.
/// cell is a layer 0 cell, layer is cost of upper tree of cell observed at layer l
struct CostUnit {
    cell_index: Vec<u16>,
    layer: u16,
    cost: u32,
}

impl CostUnit {
    fn new(cell_index: Vec<u16>, layer: u16, cost: u32) -> Self {
        CostUnit {
            cell_index,
            layer,
            cost,
        }
    }
}

//===============

// This structure drives the merge procedure of subtrees in spacemesh
struct CostBenefit<'a, 'b, T: Float> {
    //
    spacemesh: &'b SpaceMesh<'a, T>,
    //
}

impl<'a, 'b, T> CostBenefit<'a, 'b, T>
where
    T: Float + Sync + std::fmt::Debug,
    'a: 'b,
{
    fn new(spacemesh: &'b SpaceMesh<'a, T>) -> Self {
        CostBenefit { spacemesh }
    }
}

//==========================

pub struct Hcluster<'a, T> {
    points: Vec<&'a Point<T>>,
    //
    space: Space,
    //
    mindist: f64,
}

impl<'a, 'b, T> Hcluster<'a, T>
where
    T: Float + std::fmt::Debug + Sync + Send,
    'b: 'a,
{
    pub fn new(points: Vec<&'a Point<T>>) -> Self {
        // construct space
        let (xmin, xmax) = points
            .iter()
            .map(|p| p.get_minmax())
            .into_iter()
            .fold((T::max_value(), T::min_value()), |acc, x| {
                (acc.0.min(x.0), acc.0.max(x.1))
            });
        //
        let xmin: f64 = xmin.to_f64().unwrap();
        let xmax: f64 = xmax.to_f64().unwrap();
        log::debug!("xmin : {:.3e}, xmax : {:.3e}", xmin, xmax);
        // construct spacemesh
        let dim = points[0].get_dimension();
        let space = Space::new(dim, xmin, xmax, 0.);
        Hcluster {
            points,
            space,
            mindist: 0.,
        }
    } // end of new

    pub fn cluster(&'b self, mindist: f64) {
        // construct space
        let (xmin, xmax) = self
            .points
            .iter()
            .map(|p| p.get_minmax())
            .into_iter()
            .fold((T::max_value(), T::min_value()), |acc, x| {
                (acc.0.min(x.0), acc.0.max(x.1))
            });
        //
        let xmin: f64 = xmin.to_f64().unwrap();
        let xmax: f64 = xmax.to_f64().unwrap();
        log::debug!("xmin : {:.3e}, xmax : {:.3e}", xmin, xmax);
        // construct spacemesh
        let dim = self.points[0].get_dimension();
        let space = Space::new(dim, xmin, xmax, mindist);
        // TODO: do we need to keep points in HCluster (we clone a vec of references)
        let mut spacemesh = SpaceMesh::new(&self.space, self.points.clone());
        spacemesh.embed();
        //
        spacemesh.summary();
        spacemesh.compute_benefits();
        //
        //        let cost_analysis = CostBenefit::<'a, 'b, T>::new(&spacemesh);
        //
    }
} // end impl Hcluster

//========================================================

#[cfg(test)]
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
    fn test_cluster_random() {
        log_init_test();
        log::info!("in test_uniform_random");
        //
        let nbvec = 1_000_000usize;
        let dim = 10;
        let width: f64 = 1000.;
        let mindist = 5.;
        let unif_01 = Uniform::<f64>::new(0., width);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(234567_u64);
        let mut points: Vec<Point<f64>> = Vec::with_capacity(nbvec);
        for i in 0..nbvec {
            let p: Vec<f64> = (0..dim).map(|_| unif_01.sample(&mut rng)).collect();
            points.push(Point::<f64>::new(i, p, (i % 5).try_into().unwrap()));
        }
        let refpoints: Vec<&Point<f64>> = points.iter().map(|p| p).collect();
        // Space definition
        let space = Space::new(dim, 0., width, mindist);
        //
        let hcluster = Hcluster::new(refpoints);
        hcluster.cluster(mindist);
        //
    } //end of test_uniform_random
} // end of tests
