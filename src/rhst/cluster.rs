//! construct agglomerative cluster from rhst2
//!
//!

use cpu_time::ProcessTime;
use log::Level::Debug;
use std::time::{Duration, SystemTime};

use dashmap::{iter, mapref::one, rayon::*, DashMap};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use rayon::prelude::*;
use std::collections::HashMap;

// points are assimilated to cells of layer 0. Most cells of layer 0 should have one or very few points.
// The algorithms is translated
use num_traits::{cast::AsPrimitive, Float};

use super::point::*;
use super::rhst2::*;

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

    pub fn get_layer(&self) -> u16 {
        self.layer
    }

    pub fn get_map(&self) -> &HashMap<Vec<u16>, BenefitUnit> {
        &self.best
    }
    // return true if each sub tree is initialized
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

    pub(crate) fn get_best(&self, layer: u16, idx: &Vec<u16>) -> &BenefitUnit {
        assert!(layer > 0);
        self.bylayers[(layer - 1) as usize].get_best(idx)
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

impl<'a, T> Hcluster<'a, T>
where
    T: Float + std::fmt::Debug + Sync + Send,
{
    pub fn new(points: Vec<&'a Point<T>>) -> Self {
        // construct space
        let (xmin, xmax) = points
            .iter()
            .map(|p| p.get_minmax())
            .fold((T::max_value(), T::min_value()), |acc, x| {
                (acc.0.min(x.0), acc.0.max(x.1))
            });
        //
        let xmin: f64 = xmin.to_f64().unwrap();
        let xmax: f64 = xmax.to_f64().unwrap();
        // construct spacemesh
        let dim = points[0].get_dimension();
        log::info!(
            "dimension : {}, xmin : {:.3e}, xmax : {:.3e}",
            dim,
            xmin,
            xmax
        );
        let space = Space::new(dim, xmin, xmax, (xmax - xmin) / 100.);
        Hcluster {
            points,
            space,
            mindist: 0.,
        }
    } // end of new

    pub fn cluster(&self, mindist: f64) {
        // construct space
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
        log::debug!("xmin : {:.3e}, xmax : {:.3e}", xmin, xmax);
        // construct spacemesh
        let dim = self.points[0].get_dimension();
        let mut space = Space::new(dim, xmin, xmax, mindist);
        // TODO: do we need to keep points in HCluster (we clone a vec of references)
        let mut spacemesh = SpaceMesh::new(&mut space, self.points.clone());
        spacemesh.embed();
        //
        spacemesh.summary();
        let benefits = spacemesh.compute_benefits();
        // now we extract best subtrees from benefits in mesh
        let mut best_tree = BestTree::new(spacemesh.get_nb_layers(), &spacemesh);
        best_tree.get_benefits(&benefits);
        // now we search for indexes in the lower layer the highest layer
        // where it registered as a best benefit
        let lower_tree = &best_tree.bylayers[0];
        let lower_map: &HashMap<Vec<u16>, BenefitUnit> = lower_tree.get_map();
        log::info!("lower_map size : {}", lower_map.len());
        let mut filtered_benefits: Vec<BenefitUnit> = Vec::with_capacity(lower_map.len());
        // we loop on space mesh layer 0 cells
        let mut nb_cell_scanned = 0;
        log::info!(
            " loop on cells , nb cells to scan : {}",
            spacemesh.get_layer(0).get_nb_cells()
        );
        for entry in spacemesh.get_layer(0).get_iter() {
            nb_cell_scanned += 1;
            let cell_idx = entry.key();
            if let Some(mut lower_benefit) = lower_map.get(cell_idx) {
                let mut l = 1;
                let upper = loop {
                    let cell_idx = entry.key();
                    let upper_idx: Vec<u16> = cell_idx.iter().map(|x| x >> l).collect();
                    // lowest layer in trees ...
                    let upper_map: &HashMap<Vec<u16>, BenefitUnit> =
                        best_tree.bylayers[l - 1].get_map();
                    let upper_benefit = upper_map.get(&upper_idx).unwrap();
                    // what is cell 0 referred to in max_benefit, if equal to cell_idx level of cell_idx is at least 1, we search upward
                    // else cell_idx is not referenced anywhere
                    if !upper_benefit.get_cell_idx().iter().eq(cell_idx.iter())
                        || l == best_tree.bylayers.len()
                    {
                        break l - 1;
                    } else {
                        // we got upper most reference to cell_idx
                        // cell_idx is reference we go once in loop
                        lower_benefit = upper_benefit;
                        l = l + 1;
                    }
                }; // end of loop on layers
                if upper > 0 {
                    // we reference cell_idx and its upper most appearance
                    log::debug!(
                        "upper level found : {}, benefit {:.2e}, cell idx : {:?}",
                        upper,
                        lower_benefit.get_benefit(),
                        lower_benefit.get_cell_idx()
                    );
                    filtered_benefits.push(lower_benefit.clone());
                }
            }
        } // end loop on cells
          // we sort filtered_benefits in decreasing ordrer
        log::info!("nb cell scanned : {}", nb_cell_scanned);
        log::info!(
            "sorting filtered benefits, len : {}",
            filtered_benefits.len()
        );
        filtered_benefits.par_sort_unstable_by(|unita, unitb| {
            unitb
                .get_benefit()
                .partial_cmp(&unita.get_benefit())
                .unwrap()
        });
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
        let dim = 5;
        let width: f64 = 100.;
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
