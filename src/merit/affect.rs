//! describes affectation of data to clusters
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
