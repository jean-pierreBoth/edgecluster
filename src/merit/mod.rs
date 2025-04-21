//! This module is dedicated to information metrics related to clustering merit
//! It builds from :
//! - Vinh.N.X Information Theoretic Measures for clustering comparison: (Vinh 2010)[https://jmlr.csail.mit.edu/papers/volume11/vinh10a/vinh10a.pdf]
//!

#![allow(unused)]

use dashmap::DashMap;
use ndarray::{Array1, Array2};
use std::collections::HashMap;
use std::hash::Hash;

use num_traits::int::PrimInt;
use std::marker::PhantomData;
/// Typically an affectation abstract a clusterization as something giving the label or the rank of the cluster attached to a dataid.  
/// The DataId should satisfy the Hash trait. Morally the label is a discrete value (satisfy the PrimInt trait)
///
pub trait Affectation<DataId, DataLabel> {
    fn get_affectation(&self, dataid: DataId) -> DataLabel;
    fn get_nb_cluster(&self) -> usize;
}

//==============================================================================

/// Clusters defined by a HashMap
pub struct HashAffectation<DataId, DataLabel> {
    affectation: HashMap<DataId, DataLabel>,
}

impl<DataId, DataLabel> Affectation<DataId, DataLabel> for HashAffectation<DataId, DataLabel>
where
    DataId: Hash + Eq + Copy + Clone + Send + Sync + std::fmt::Debug,
    DataLabel: PrimInt,
{
    fn get_affectation(&self, id: DataId) -> DataLabel {
        *self.affectation.get(&id).unwrap()
    }

    fn get_nb_cluster(&self) -> usize {
        self.affectation.values().len()
    }
}

//==============================================================================

/// Clusters defined by a DashMap
pub struct DashAffectation<DataId, DataLabel> {
    affectation: DashMap<DataId, DataLabel>,
    nb_cluster: usize,
}

impl<DataId, DataLabel> Affectation<DataId, DataLabel> for DashAffectation<DataId, DataLabel>
where
    DataId: Hash + Eq + Copy + Clone + Send + Sync + std::fmt::Debug,
    DataLabel: PrimInt,
{
    fn get_affectation(&self, id: DataId) -> DataLabel {
        let item = self.affectation.get(&id).unwrap();
        *item.value()
    }
    fn get_nb_cluster(&self) -> usize {
        self.nb_cluster
    }
}

impl<DataId, DataLabel> DashAffectation<DataId, DataLabel>
where
    DataId: Hash + Eq + Copy + Clone + Send + Sync + std::fmt::Debug,
    DataLabel: PrimInt,
{
    pub fn new(affectation: DashMap<DataId, DataLabel>, nb_cluster: usize) -> Self {
        DashAffectation {
            affectation,
            nb_cluster,
        }
    }
}

//===============================================================================

/// Clusters defined by a Vector, DataId is just a rank in vector, the content of the vector is the label

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
    // The contingency table
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
        // computes
        //
        panic!("not yet implemented");
    }
} // end of Contingency
