//! contingency table

#![allow(unused)]

use ndarray::{Array1, Array2};
use std::hash::Hash;

use num_traits::int::PrimInt;
use std::marker::PhantomData;

use super::affect::*;
//================================================================================

/// Contingency table associated to the 2 affectations to compare
/// We can compare either a true (reference) labels of data or 2 clusters algorithms
pub struct Contingency<Clusterization, DataId, DataLabel>
where
    Clusterization: Affectation<DataId, DataLabel>,
    DataId: Hash + Eq + Copy + Clone + Send + Sync + std::fmt::Debug,
    DataLabel: PrimInt,
{
    cluster1: Clusterization,
    cluster2: Clusterization,
    // The contingency table. dimension (cluster1.nb_cluster, cluster2.nb_cluster)
    table: Array2<usize>,
    // number of elements in each clusters of cluster1
    c1_size: Array1<usize>,
    // number of elements in each clusters of cluster2
    c2_size: Array1<usize>,
    //
    _t_id: PhantomData<DataId>,
    _t_label: PhantomData<DataLabel>,
}

impl<DataId, DataLabel, Clusterization> Contingency<Clusterization, DataId, DataLabel>
where
    Clusterization: Affectation<DataId, DataLabel>,
    DataId: Hash + Eq + Copy + Clone + Send + Sync + std::fmt::Debug,
    DataLabel: PrimInt,
{
    pub fn new(clusters1: Clusterization, clusters2: Clusterization) -> Self {
        let nbclust1 = clusters1.get_nb_cluster();
        let nbclust2: usize = clusters1.get_nb_cluster();
        let table = Array2::<usize>::zeros((nbclust1, nbclust2));
        // computes
        //
        panic!("not yet implemented");
    }
} // end of Contingency
