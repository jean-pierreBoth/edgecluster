//! contingency table

#![allow(unused)]

use indexmap::IndexSet;
use ndarray::{Array1, Array2};
use std::hash::Hash;

use num_traits::int::PrimInt;
use std::marker::PhantomData;

use super::affect::*;
//================================================================================

#[cfg_attr(doc, katexit::katexit)]
/// Contingency table associated to the 2 clusterization (affectations) to compare. We can compare either a true (reference) labels of data or 2 clusters algorithms  
/// The various merit functions relies on comparisons entropy of cluster distribution.  
///
///
/// We contruct a contingency matrix $ \left(n_{ij} \right) $  with $ i \le n_{1}, j \le n_{2} $ with $ n_{ij} =   | C_{1}[i] \cap C_{2}[j] | $ with $ C_{1}[i] $ designing the i-th cluster in C1 clusterization.
/// We call:
/// - N : the number of elements to cluster
/// - $NC_{1}$ (resp. $NC_{2}$) the number of clusters of the first (resp. second) clusterization.
///
/// The following entropies are then computed:
/// - $ H(C_{1}) = - \sum_{i \le NC_{1}}  \frac{|C_1[i]|}{N} \log \frac{|C_1[i]|}{N} $
/// - $ H(C_{2}) = - \sum_{i \le NC_{2}}  \frac{|C_2[i]|}{N} \log \frac{|C_2[i]|}{N} $  
/// - $ H(C_{1},C_{2}) = - \sum_{i \le NC_{1}, j \le NC_{2}}  \frac{n_{ij}}{N} \log \frac{n_{ij}}{N} $
/// - $ H(C_{1}| C_{2}) = - \sum_{i \le NC_{1}, j \le NC_{2}}  \frac{n_{ij}}{N} \log \frac{n_{ij}/N} {|C_2[j]|/N} $
///
/// Various indicators can then be computed (some are even metrics), we choose normalized versions i.e there values are in the range [0,1].  
/// See the different functions
pub struct Contingency<Clusterization, DataId, DataLabel>
where
    Clusterization: Affectation<DataId, DataLabel>,
    DataId: Hash + Eq + Copy + Clone + Send + Sync + std::fmt::Debug,
    DataLabel: PrimInt,
{
    // clusters (or reference)
    clusters1: Clusterization,
    // clusters (or reference)
    clusters2: Clusterization,
    // transform labels set to usize range for array indexation
    labels1: IndexSet<DataLabel>,
    labels2: IndexSet<DataLabel>,
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
    DataLabel: PrimInt + Hash,
{
    pub fn new(clusters1: Clusterization, clusters2: Clusterization) -> Self {
        let nbclust1 = clusters1.get_nb_cluster();
        let nbclust2: usize = clusters2.get_nb_cluster();
        let mut table = Array2::<usize>::zeros((nbclust1, nbclust2));
        let mut c1_size: ndarray::ArrayBase<ndarray::OwnedRepr<usize>, ndarray::Dim<[usize; 1]>> =
            Array1::<usize>::zeros(nbclust1);
        let mut c2_size: ndarray::ArrayBase<ndarray::OwnedRepr<usize>, ndarray::Dim<[usize; 1]>> =
            Array1::<usize>::zeros(nbclust2);
        //
        // converts labels to contiguous interval of usize. label_rank = IndexSet::get_index_of(label).unwrap()
        //
        let mut labels1 = IndexSet::<DataLabel>::with_capacity(nbclust1);
        let affect1_iter = clusters1.iter();
        for (id, label) in affect1_iter {
            labels1.insert(label);
        }
        let mut labels2 = IndexSet::<DataLabel>::with_capacity(nbclust1);
        let affect2_iter = clusters2.iter();
        for (id, label) in affect2_iter {
            labels2.insert(label);
        }
        // computes contingency table
        let affect1_iter = clusters1.iter();
        // we loop on affect1_iter, query each item relativeley to clusters2 and dispatch to table
        for (id1, label1) in affect1_iter {
            // we loop on affect2_iter, query each item relativeley to clusters2 and dispatch to table
            let affect2_iter = clusters2.iter();
            let rank_l1 = labels1.get_index_of(&label1).unwrap();
            c1_size[rank_l1] += 1;
            let label2 = clusters2.get_affectation(id1);
            let rank_l2 = labels2.get_index_of(&label2).unwrap();
            c2_size[rank_l2] += 1;
            // summing on columns each item in cluster1 appears exactly once
            // and summing on rows item in cluster2 appears exactly once (as long as the same set of DataId is in both clusterization)
            table[[rank_l1, rank_l2]] += 1;
        }
        // TODO: compute entropies H and I
        let entropy_1 = c1_size
            .iter()
            .fold(0., |acc, x| acc - *x as f64 * log_with0(*x as f64));
        //
        let entropy_2 = c2_size
            .iter()
            .fold(0., |acc, x| acc - *x as f64 * log_with0(*x as f64));
        //
        let entropy_12 = table
            .iter()
            .fold(0., |acc, x| acc - *x as f64 * log_with0(*x as f64));
        //
        //
        Contingency {
            clusters1,
            clusters2,
            labels1,
            labels2,
            table,
            c1_size,
            c2_size,
            _t_id: PhantomData,
            _t_label: PhantomData,
        }
    }

    #[cfg_attr(doc, katexit::katexit)]
    /// compute normalized mutual information joint version
    /// compures $ 1. - \frac{I(C_{1}, C_{2})}{H(C_{1}, C_{2})} $.  
    /// This fonction is a metric.
    pub fn get_nmi_joint(&self) -> f64 {
        panic!("not yet impelemnted");
    }
} // end of Contingency

// for entropy calculations log(0) = 0...
fn log_with0(arg: f64) -> f64 {
    if arg > 0. {
        arg.ln()
    } else if arg < 0. {
        panic!("log cannot have negative arg");
    } else {
        0.
    }
}
