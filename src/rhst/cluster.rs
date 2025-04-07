//! construct agglomerative cluster from rhst2
//!
//!

use cpu_time::ProcessTime;
use rand_distr::{Distribution, StandardNormal};
use std::time::{Duration, SystemTime};

use lax::Lapack;
// use ndarray_rand::rand_distr::{Distribution, StandardNormal};

use rayon::prelude::*;
use std::collections::HashMap;

// points are assimilated to cells of layer 0. Most cells of layer 0 should have one or very few points.
// The algorithms is translated
use num_traits::Float;

use super::point::*;
use super::rhst2::*;
use crate::smalld::*;

//============================================

// This structure stores, for a  given layer,  for each sub tree rooted at this layer, the layer 0 cell (alias for a point) with max benefit
pub(crate) struct LayerBestTree {
    layer: u16,
    // at each cell identified by its index at layer l given by Vec<u16> corresponds the cell (point) at layer 0 having the higher benefit
    best: HashMap<Vec<u16>, BenefitUnit>,
    // number of sub trees needed to be full
    nb_subtrees: usize,
}

impl LayerBestTree {
    pub fn new_with_size(layer: u16, nbcells: usize) -> Self {
        // layer 0 has no subtree
        assert!(layer > 0);
        let best: HashMap<Vec<u16>, BenefitUnit> = HashMap::with_capacity(nbcells);
        LayerBestTree {
            layer,
            best,
            nb_subtrees: nbcells,
        }
    }

    #[allow(unused)]
    pub(crate) fn get_layer(&self) -> u16 {
        self.layer
    }

    pub fn get_map(&self) -> &HashMap<Vec<u16>, BenefitUnit> {
        &self.best
    }
    // return true if each sub tree is initialized
    #[allow(unused)]
    pub(crate) fn is_full(&self) -> bool {
        self.nb_subtrees == self.best.len()
    }

    // return true if full
    #[allow(clippy::map_entry)]
    pub(crate) fn insert(&mut self, l_idx: Vec<u16>, unit: &BenefitUnit) -> bool {
        if self.best.contains_key(&l_idx) {
            false
        } else {
            self.best.insert(l_idx, unit.clone());
            if self.best.len() >= self.nb_subtrees {
                log::debug!(
                    "LayerBestTree at layer {} full nb sub tress {}",
                    self.layer,
                    self.nb_subtrees
                );
                true
            } else {
                false
            }
        }
    } // end of insert

    #[allow(unused)]
    pub fn get_best(&self, idx: &Vec<u16>) -> &BenefitUnit {
        let res = self.best.get(idx);
        if res.is_none() {
            log::error!(
                "cannot find best point for root tree at layer : {}, index : {:?}",
                self.layer,
                idx
            );
            panic!("error in RootBestTree::get_best");
        };
        res.unwrap()
    }

    pub fn get_nb_subtree(&self) -> usize {
        self.nb_subtrees
    }
} // end of LayerBestTree

//==================

// gathers subtrees at each layer
pub(crate) struct BestTree {
    // subtrees by layer
    bylayers: Vec<LayerBestTree>,
    // status of each layer subtrees. If all is full we can stop parsing benefits
    filled: Vec<bool>,
}

impl BestTree {
    //
    pub(crate) fn new<T>(nb_layer: usize, spacemesh: &SpaceMesh<'_, T>) -> Self
    where
        T: Float + std::fmt::Debug + Sync,
    {
        let mut bylayers: Vec<LayerBestTree> = Vec::with_capacity(nb_layer);
        for layer in 1..nb_layer {
            let nbcells = spacemesh.get_layer_size(layer);
            let layer_tree = LayerBestTree::new_with_size(layer as u16, nbcells);
            bylayers.push(layer_tree);
        }
        let filled = vec![false; nb_layer];
        BestTree { bylayers, filled }
    }

    // as benefits are sorted in decreasing order, we can scan units and as soon as each cell of each layer
    // has seen its index appearing we are done, every subsequent item will not be the best
    pub(crate) fn get_benefits(&mut self, benefits: &[BenefitUnit]) {
        //
        log::info!("in from_benefits");
        let cpu_start = ProcessTime::now();
        let sys_now = SystemTime::now();
        //
        let mut nb_full = 0;
        for (i, unit) in benefits.iter().enumerate() {
            //
            if i <= 10 && log::log_enabled!(log::Level::Debug) {
                log::debug!(" benefit rank : {}, unit : {:?}", i, unit);
            }
            //
            let (idx, l) = unit.get_id();
            assert!(l >= 1);
            let layer = &mut self.bylayers[l as usize - 1];
            // convert idx from layer 0 to layer l
            let l_idx: Vec<u16> = idx.iter().map(|x| x >> l).collect();
            let full = layer.insert(l_idx, unit);
            if full {
                self.filled[l as usize] = true;
                nb_full += 1;
                if nb_full == self.bylayers.len() {
                    log::info!("from benefits exiting after nb unit scan : {}", i);
                    break;
                }
            }
        }
        log::info!("exiting from_benefits");
        let cpu_time: Duration = cpu_start.elapsed();
        println!(
            " compute_benefits sys time(s) {:?} cpu time {:?}",
            sys_now.elapsed().unwrap().as_secs(),
            cpu_time.as_secs()
        );
        assert_eq!(nb_full, self.bylayers.len());
        //
    } // end of from_benefits

    #[allow(unused)]
    pub(crate) fn get_best(&self, layer: u16, idx: &Vec<u16>) -> &BenefitUnit {
        assert!(layer > 0);
        self.bylayers[(layer - 1) as usize].get_best(idx)
    }

    //
    pub(crate) fn get_filtered_benefits(&self) -> Vec<BenefitUnit> {
        let nb_sub_trees: usize = self
            .bylayers
            .iter()
            .fold(0, |acc, l| acc + l.get_nb_subtree());
        //
        let mut filtered_benefits: Vec<BenefitUnit> = Vec::with_capacity(nb_sub_trees);
        // we collect units and sort them. reverse order as higher benefits should be upper...
        for l in (0..self.bylayers.len()).rev() {
            let layer = &self.bylayers[l];
            for v in layer.get_map().values() {
                filtered_benefits.push(v.clone());
            }
        }
        log::info!(
            "BestTree::get_filtered_benefits, got nb unit : {}",
            filtered_benefits.len()
        );
        // sorting
        filtered_benefits.par_sort_unstable_by(|unita, unitb| {
            unitb
                .get_benefit()
                .partial_cmp(&unita.get_benefit())
                .unwrap()
        });
        //
        filtered_benefits
    }
}

//==========================

pub struct Hcluster<'a, T> {
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
            points,
            mindist: 0.,
            reducer,
            reduced_points: None,
        }
    } // end of new

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
        //
        spacemesh.summary();
        let benefits = spacemesh.compute_benefits();
        // now we extract best subtrees from benefits in mesh
        let mut best_tree = BestTree::new(spacemesh.get_nb_layers(), &spacemesh);
        best_tree.get_benefits(&benefits);

        let filtered_benefits = best_tree.get_filtered_benefits();

        log::info!("dump of BestTree::get_filtered_benefits");
        dump_benefits(&filtered_benefits);
        check_partition(&spacemesh, &filtered_benefits);
        //
        // we have benefits, we can try to cluster
        //
        let _clusters = spacemesh.get_partition_by_size(nb_cluster, &filtered_benefits);
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
        let nbvec = 1_000_000usize;
        let dim = 5;
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
        hcluster.cluster(mindist, 5);
        //
    } //end of test_cluster_random

    #[test]
    fn test_cluster_exp() {
        log_init_test();
        log::info!("in test_uniform_random");
        //
        let nbvec = 1_000_000usize;
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
