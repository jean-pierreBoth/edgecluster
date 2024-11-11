//! construct agglomerative cluster from rhst2
//!

use dashmap::{iter, mapref::one, rayon::*, DashMap};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use rayon::prelude::*;

// points are assimilated to cells of layer 0. Most cells of layer 0 should have one or very few points.
// The algorithms is translated
use num_traits::Float;

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
            for l in 0..self.spacemesh.get_nb_layers() {
                let upper_tree_size: u32 = if l == 0 {
                    cell.get_subtree_size()
                } else {
                    let upper_index = cell.get_upper_cell_index_at_layer(l as u16);

                    let upper_cell = self
                        .spacemesh
                        .get_layer(l as u16)
                        .get_cell(&upper_index)
                        .unwrap();
                    upper_cell.get_subtree_size()
                };
                benefits.push(CostUnit::new(cell.key().clone(), 0, upper_tree_size));
            }
        }
        // sort benefits in decreasing (!) order
        benefits.par_sort_unstable_by(|unita, unitb| unitb.cost.partial_cmp(&unita.cost).unwrap());
    } // end of compute_benefits
}
