//! construct agglomerative cluster from rhst2
//!

use dashmap::{iter, mapref::one, rayon::*, DashMap};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use rayon::prelude::*;

// points are assimilated to cells of layer 0. Most cells of layer 0 should have one or very few points.
// The algorithms is translated
use num_traits::{cast::AsPrimitive, Float};

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
struct CostBenefit<'a, T: Float> {
    //
    spacemesh: &'a SpaceMesh<'a, T>,
    //
}

impl<'a, T> CostBenefit<'a, T>
where
    T: Float + Sync + std::fmt::Debug,
{
    // compute benefits by points (in fact cells at layer 0) and layer (See algo 2 of paper and lemma 3.4)
    fn compute_benefits(&self) {
        let nb_layers = self.spacemesh.get_nb_layers();
        let layer_0 = self.spacemesh.get_layer(0);
        let nb_cells = layer_0.get_nb_cells();
        // allocate benefit array in one array as we will need a sort! with par_sort_unstable from rayon
        // the nb_layers of a given cell are stored contiguously in the order obtained from the iterator
        let benefits = Vec::<usize>::with_capacity(nb_cells * nb_layers);
        // loop on cells of layer_0, keeping track of the order
        let cell_order = Vec::<Vec<u16>>::with_capacity(nb_cells);
        // for each cell we store potential merge benefit at each level!
        let mut benefits: Vec<CostUnit> = Vec::with_capacity(nb_cells * nb_layers);
        // iterate over layer 0 and upper_layers store cost and then sort benefit array in decreasing order!
        let layer0 = self.spacemesh.get_layer(0);
        for cell in layer0.get_hcells().iter() {
            let mut benefit_at_layer = Vec::<u32>::with_capacity(nb_layers);
            let mut previous_tree_size: u32 = cell.get_subtree_size();
            for l in 1..self.spacemesh.get_nb_layers() {
                let upper_index = cell.get_upper_cell_index_at_layer(l as u16);
                let upper_cell = self
                    .spacemesh
                    .get_layer(l as u16)
                    .get_cell(&upper_index)
                    .unwrap();
                upper_cell.get_subtree_size();
                // we can unroll benefit computation ( and lyer 0 give no benefit)
                let benefit =
                    2u32.pow(l as u32) * previous_tree_size + benefit_at_layer.last().unwrap();
                benefit_at_layer.push(benefit);
                benefits.push(CostUnit::new(cell.key().clone(), l as u16, benefit));
            }
        }
        // sort benefits in decreasing (!) order
        benefits.par_sort_unstable_by(|unita, unitb| unitb.cost.partial_cmp(&unita.cost).unwrap());
    } // end of compute_benefits
}

//==========================

pub struct Hcluster<'a, T> {
    points: Vec<&'a Point<T>>,
    //
    nbcluster: usize,
}

impl<'a, T> Hcluster<'a, T>
where
    T: Float + std::fmt::Debug + Sync + Send,
{
    pub fn new(points: Vec<&'a Point<T>>, nbcluster: usize) -> Self {
        Hcluster { points, nbcluster }
    } // end of new

    pub fn cluster(&mut self, mindist: f64) {
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
        //
        //       let spacemesh = SpaceMesh::new(&space, self.points);
    }
} // end impl Hcluster
