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
    T: Float,
{
    // compute benefits by points (in fact cells) and layer (See algo 2 of paper and lemma 3.4)
    fn compute_benefits(&self) {}

    // We need to compute cardinal of the tree rooted at a_i(c) the ancestor at layer i of lowest cell c (at layer 0)
    // This is equivalent to computing cardinal of each subtree. The result is contained in the structure SpaceMesh
    // as :
    // - we know the number of points in each cell
    // - knowing the index of cell c at layer i we can deduce the index of parent cell at layer j < i (take  indexes modulo 2^(i-j))
    //
    fn compute_subtree_cardinals(&self) {}
}
