//! construct agglomerative cluster from rhst2

// points are assimilated to cells of layer 0. Most cells of layer 0 should have one or very few points.
// The algorithms is translated
use num_traits::Float;

use super::rhst2::SpaceMesh;

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

        // store benefit array in decreasing order!
    }
}
