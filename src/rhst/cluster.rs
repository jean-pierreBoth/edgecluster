//! construct agglomerative cluster from rhst2
//!
//!

use cpu_time::ProcessTime;
use portable_atomic::*;
use rand_distr::{Distribution, StandardNormal};
use rayon::iter::IntoParallelIterator;
use rayon::prelude::*;
use std::time::SystemTime;

// io
use csv::WriterBuilder;
use serde::Serialize;
use std::fs::OpenOptions;
//
use dashmap::DashMap;
use lax::Lapack;
use std::path::PathBuf;

// points are assimilated to cells of layer 0. Most cells of layer 0 should have one or very few points.
// The algorithms is translated
use num_traits::Float;

use super::point::*;
use super::rhst2::*;
use crate::smalld::*;

use crate::merit::affect::*;

//   just to dump centers

#[derive(Serialize)]
struct CsvRecord {
    center: Vec<f64>,
    center_str: Vec<String>,
}

impl CsvRecord {
    pub fn from<T: Float>(data: &[T]) -> Self {
        let center: Vec<f64> = data.iter().map(|x| x.to_f64().unwrap()).collect();
        let center_str: Vec<String> = center.iter().map(|x| format!("{:3.3E}", x)).collect();
        CsvRecord { center, center_str }
    }

    #[allow(unused)]
    pub fn get_center_str_iter(&self) -> impl std::iter::Iterator<Item = &String> {
        self.center_str.iter()
    }
}

//===========================

pub struct ClusterResult {
    /// a map from  points id to a cluster num or label
    point_to_cluster: DashMap<usize, u32>,
    /// For its cluster, the pointId of its medoid center
    cluster_center_to_pid: DashMap<u32, usize>,
    /// for each cluster a vector of point id affected to it
    clusters: Vec<Vec<usize>>,
}

impl ClusterResult {
    pub(crate) fn new(
        point_to_cluster: DashMap<usize, u32>,
        cluster_center_to_pid: DashMap<u32, usize>,
        clusters: Vec<Vec<usize>>,
    ) -> ClusterResult {
        ClusterResult {
            point_to_cluster,
            cluster_center_to_pid,
            clusters,
        }
    }

    /// return a map from  points id to a cluster num
    pub fn get_map(&self) -> &DashMap<usize, u32> {
        &self.point_to_cluster
    }

    /// for each cluster a vector of point id affected to it
    pub fn get_clusters(&self) -> &Vec<Vec<usize>> {
        &self.clusters
    }

    /// dump info on clusters: centerid , size, possibly labels of centers
    /// labels : a vector containing points labels
    pub fn dump_cluster_id<LabelId>(&self, labels: Option<Vec<LabelId>>)
    where
        LabelId: Copy + Clone + std::fmt::Display,
    {
        for (rank, v) in self.clusters.iter().enumerate() {
            let r_u32: u32 = rank.try_into().unwrap();
            let center_id = *self.cluster_center_to_pid.get(&r_u32).unwrap().value();
            if labels.is_none() {
                println!(
                    "cluster : {} , center_id : {}, size : {}",
                    rank,
                    center_id,
                    v.len()
                );
            } else {
                let label_id = labels.as_ref().unwrap()[center_id];
                println!(
                    "cluster : {} , center_id : {}, label : {}, size : {}",
                    rank,
                    center_id,
                    label_id,
                    v.len()
                );
            }
        }
    }

    /// returns the number of points in cluster i
    pub fn get_cluster_size(&self, i: usize) -> usize {
        self.clusters[i].len()
    }

    /// returns the number of data to cluster
    pub fn get_nb_points(&self) -> usize {
        self.point_to_cluster.len()
    }

    /// compute centers as in kmean for l2 cost
    pub fn compute_cluster_mean_center<
        T: Float + std::fmt::Debug + std::ops::AddAssign + std::ops::DivAssign,
    >(
        &self,
        points: &[&Point<T>],
    ) -> Vec<Vec<T>> {
        let dim = points[0].get_dimension();
        let mut centers: Vec<Vec<T>> = (0..self.clusters.len())
            .map(|_| vec![T::zero(); dim])
            .collect();
        //
        for item in self.point_to_cluster.iter() {
            let point = points[*item.key()];
            let xyz = point.get_position();
            let center = &mut centers[*item.value() as usize];
            for d in 0..dim {
                center[d] += xyz[d]
            }
        }
        // renormalize
        for (cluster, point) in &mut centers.iter_mut().enumerate() {
            for x in point {
                *x /= T::from(self.clusters[cluster].len()).unwrap();
            }
        }
        //
        centers
    }

    #[cfg_attr(doc, katexit::katexit)]
    /// cost returned is
    /// $ \sum_{1}^{N}  ||x_{i} - C(x_{i})|| $ where $C(x_{i})$ is the nearest cluster center for $ x_{i}$.
    /// result will be written in a csv file with name $filename$
    pub fn compute_cost_medoid_l2<
        T: Float + std::fmt::Debug + std::ops::AddAssign + std::ops::DivAssign,
    >(
        &self,
        points: &[&Point<T>],
        dumpfile: Option<&str>,
    ) -> T {
        log::info!("in ClusterResult compute_cost_medoid_l2");
        //
        let mut norm = T::zero();
        for item in self.point_to_cluster.iter() {
            let point = points[*item.key()];
            let xyz = point.get_position();
            let key = point.get_id();
            let cluster = *self.point_to_cluster.get(&key).unwrap().value();
            let center_id = *self.cluster_center_to_pid.get(&cluster).unwrap().value();
            let center_point = points[center_id];
            norm += xyz
                .iter()
                .zip(center_point.get_position())
                .fold(T::zero(), |acc, (p, c)| acc + (*p - *c) * (*p - *c))
                .sqrt()
        }
        //
        if let Some(csvloc) = dumpfile {
            let mut csvpath = PathBuf::from(".");
            csvpath.push(csvloc);
            let csvfileres = OpenOptions::new()
                .create(true)
                .truncate(true)
                .write(true)
                .open(&csvpath);
            if csvfileres.is_err() {
                println!(" could not open file {:?}", csvpath.as_os_str());
            }
            let wtr = WriterBuilder::new().from_path(&csvpath);
            if wtr.is_err() {
                log::error!("compute_cost_medoid_l1 cannot open dump csv file");
            } else {
                log::info!("dumping csv file in {:?}", csvpath.as_os_str());
                let mut wtr = wtr.unwrap();
                for item in self.cluster_center_to_pid.iter() {
                    let cluster = *item.key();
                    let center_id = *self.cluster_center_to_pid.get(&cluster).unwrap().value();
                    let center_point = points[center_id];
                    let csv_record = CsvRecord::from(center_point.get_position());
                    wtr.serialize(csv_record.center_str).unwrap();
                }
            }
        }
        //
        norm
    }

    //

    #[cfg_attr(doc, katexit::katexit)]
    /// cost returned is
    /// $ \sum_{1}^{N}  ||x_{i} - C(x_{i})|| $ where $C(x_{i})$ is the nearest cluster center for $ x_{i}$.
    /// result will be written in a csv file with name $filename$
    pub fn compute_cost_medoid_l1<
        T: Float + std::fmt::Debug + std::ops::AddAssign + std::ops::DivAssign,
    >(
        &self,
        points: &[&Point<T>],
        dumpfile: Option<&str>,
    ) -> T {
        log::info!("in ClusterResult compute_cost_medoid_l1");
        //
        let mut norm = T::zero();
        for item in self.point_to_cluster.iter() {
            let point = points[*item.key()];
            let xyz = point.get_position();
            let key = point.get_id();
            let cluster = *self.point_to_cluster.get(&key).unwrap().value();
            let center_id = *self.cluster_center_to_pid.get(&cluster).unwrap().value();
            let center_point = points[center_id];
            norm += xyz
                .iter()
                .zip(center_point.get_position())
                .fold(T::zero(), |acc, (p, c)| {
                    acc + num_traits::Float::abs(*p - *c)
                })
        }
        //
        if let Some(csvloc) = dumpfile {
            let mut csvpath = PathBuf::from(".");
            csvpath.push(csvloc);
            let csvfileres = OpenOptions::new()
                .create(true)
                .truncate(true)
                .write(true)
                .open(&csvpath);
            if csvfileres.is_err() {
                println!(" could not open file {:?}", csvpath.as_os_str());
            }
            let wtr = WriterBuilder::new().from_path(&csvpath);
            if wtr.is_err() {
                log::error!("compute_cost_medoid_l1 cannot open dump csv file");
            } else {
                log::info!("dumping csv file in {:?}", csvpath.as_os_str());
                let mut wtr = wtr.unwrap();
                for item in self.cluster_center_to_pid.iter() {
                    let cluster = *item.key();
                    let center_id = *self.cluster_center_to_pid.get(&cluster).unwrap().value();
                    let center_point = points[center_id];
                    let csv_record = CsvRecord::from(center_point.get_position());
                    wtr.serialize(csv_record.center_str).unwrap();
                }
            }
        }
        //
        norm
    }

    //

    /// Return struct DashAffectation which implement trait Affectation<usize, u32>
    pub fn get_dash_affectation(&self) -> DashAffectation<usize, u32> {
        DashAffectation::<usize, u32>::new(&self.point_to_cluster)
    }
} // end of impl ClusterResult

//==========================

pub struct Hcluster<'a, T> {
    debug_level: usize,
    //
    points: Vec<&'a Point<T>>,
    //
    auto_dim: bool,
    //
    reducer: Option<&'a dyn reducer::Reducer<T>>,
    // dimension of reduced points if reduction was used
    reduced_dim: Option<usize>,
    // if we (must) reduce points dimension,
    reduced_points: Option<Vec<Point<T>>>,
}

impl<'a, T> Hcluster<'a, T>
where
    T: Float + std::fmt::Debug + Sync + Send + Lapack + ndarray::ScalarOperand,
    StandardNormal: Distribution<T>,
{
    /// Constructs a structure to cluster the points given in arguments.  
    /// By default the dimension of data is respected, but it is possible to reduce this dimension.
    /// The module [smalld](crate::smalld) provides for this.  
    /// The user can also provide its own dimension reduction algorithm with the argument given as an option.
    /// If not, the algorithm will try to choose one.  
    ///
    /// NOTA: As points are passed as a Vec, the PointId identificator of a Point must be its rank!
    pub fn new(points: Vec<&'a Point<T>>, reducer: Option<&'a dyn reducer::Reducer<T>>) -> Self {
        //
        log::info!("entering Hcluster::new, nb_points : {}", points.len());
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
        //
        let auto_dim = false;
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
            debug_level: 0,
            points,
            auto_dim,
            reducer,
            reduced_dim: None,
            reduced_points: None,
        }
    } // end of new

    pub fn set_debug_level(&mut self, level: usize) {
        self.debug_level = level;
    }

    /// returns references to points
    pub fn get_points(&self) -> &Vec<&'a Point<T>> {
        &self.points
    }

    /// get reduced dimension if reduced
    pub fn get_reduced_dim(&self) -> Option<usize> {
        self.reduced_dim
    }

    pub fn get_data_dim(&self) -> usize {
        self.points[0].get_dimension()
    }

    /// The function that triggers the clustering.  
    /// The arguments are:  
    /// - number of clusters asked for
    /// - auto_dim : set to true, the algorithm will try to reduce data dimension using module [crate::smalld]
    /// - if a reduction  to a given dimension is necessary, this option will use it.
    /// The function returns a map giving for each point id its cluster
    pub fn cluster(
        &mut self,
        nb_cluster: usize,
        auto_dim: bool,
        reduced_dim_opt: Option<usize>,
    ) -> ClusterResult {
        //
        let cpu_start = ProcessTime::now();
        let sys_now = SystemTime::now();
        // construct space
        self.auto_dim = auto_dim;
        let (xmin, xmax) = self
            .points
            .iter()
            .map(|p| p.get_minmax())
            .fold((T::max_value(), T::min_value()), |acc, x| {
                (acc.0.min(x.0), acc.1.max(x.1))
            });
        //
        let xmin: f64 = xmin.to_f64().unwrap();
        let xmax: f64 = xmax.to_f64().unwrap();
        let dim = self.points[0].get_dimension();
        log::debug!("dim : {} xmin : {:.3e}, xmax : {:.3e}", dim, xmin, xmax);
        // TODO: do we need to keep points in HCluster (we clone a vec of references)
        let points_to_cluster: Vec<&Point<T>>;
        let reduced_dim = match reduced_dim_opt {
            Some(d) => d,
            _ => 0,
        };
        if (dim > self.points.len().ilog(2) as usize && self.auto_dim) || reduced_dim > 0 {
            let mut to_dim: usize = self.points.len().ilog(2) as usize;
            if reduced_dim > 0 {
                to_dim = to_dim.min(reduced_dim);
            }
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
            log::info!("clustering keeping original dimension: {} ", dim);
            points_to_cluster = self.points.clone();
        }
        // we have points_to_cluster , we can construct space
        let (xmin, xmax) = points_to_cluster
            .iter()
            .map(|p| p.get_minmax())
            .fold((T::max_value(), T::min_value()), |acc, x| {
                (acc.0.min(x.0), acc.1.max(x.1))
            });
        //
        let xmin: f64 = xmin.to_f64().unwrap();
        let xmax: f64 = xmax.to_f64().unwrap();
        let dim = points_to_cluster[0].get_dimension();
        log::info!("dim : {}, xmin : {:.3e}, xmax : {:.3e}", dim, xmin, xmax);
        // construct spacemesh
        let mut space = Space::new(dim, xmin, xmax);
        let mut spacemesh = SpaceMesh::new(&mut space, points_to_cluster);
        spacemesh.embed();

        if self.debug_level > 1 {
            spacemesh.dump_layer(0, self.debug_level);
        }
        //
        spacemesh.summary();

        let filtered_benefits = spacemesh.compute_benefits();
        if log::log_enabled!(log::Level::Trace) {
            log::trace!(
                "dump of filtered_benefits, nbunits : {}",
                filtered_benefits.len()
            );
            dump_benefits(&spacemesh, &filtered_benefits);
        }
        if log::log_enabled!(log::Level::Debug) {
            check_benefits_cover(&spacemesh, &filtered_benefits);
        }
        //
        // we have benefits, we can try to cluster
        //
        let (point_to_cluster, cluster_to_center_pid) =
            spacemesh.get_partition(nb_cluster, &filtered_benefits);
        assert_eq!(cluster_to_center_pid.len(), nb_cluster);
        // keep centers and reaffect, computing new cost

        let (point_reaffectation, medoid_cost) =
            self.compute_cost_medoid_l2(&point_to_cluster, &cluster_to_center_pid);
        log::info!(
            "medoid cost in original space after reaffectation (l2) : {:.3e} with {} clusters",
            medoid_cost,
            nb_cluster
        );
        //
        let mut clusters: Vec<Vec<usize>> = (0..nb_cluster).map(|_| Vec::<usize>::new()).collect();
        for item in point_to_cluster.iter() {
            clusters[*item.value() as usize].push(*item.key());
        }
        let mut centers_pid = vec![0usize; nb_cluster];
        for item in &cluster_to_center_pid {
            centers_pid[*item.key() as usize] = *item.value();
        }
        //
        println!(
            " Cluster (embedding+dimensionr reduction + partitioning) time(s) {:?} cpu time {:?}",
            sys_now.elapsed().unwrap().as_secs(),
            cpu_start.elapsed().as_secs()
        );
        //
        ClusterResult::new(point_reaffectation, cluster_to_center_pid, clusters)
    }

    // pt_affectation constains the cluster rank,
    // cluster_center contains the point_id designated as the center , resulting from benefit analysis
    // returns mean L2 cost
    #[allow(unused)]
    pub(crate) fn compute_cost_medoid_l2(
        &self,
        pt_affectation: &DashMap<usize, u32>,
        cluster_center: &DashMap<u32, usize>,
    ) -> (DashMap<usize, u32>, f64) {
        //
        log::info!("in HCluster compute_cost_medoid_l2");
        //
        assert_eq!(self.points.len(), pt_affectation.len());

        let new_affectation = DashMap::<usize, u32>::new();
        let norm: AtomicF64 = AtomicF64::new(0.);
        let without_reaffectation_norm = AtomicF64::new(0.);
        let nb_cluster = cluster_center.len();
        let nb_points = self.points.len();
        let points = &self.points;
        log::info!("dimension : {}", self.points[0].get_dimension());
        (0..nb_points).into_par_iter().for_each(|point_rank| {
            let point = points[point_rank];
            let xyz = point.get_position();
            let mut mindist = f64::MAX;
            let mut minclust: usize = usize::MAX;
            for i in 0..nb_cluster {
                let cluster_rank: u32 = i.try_into().unwrap();
                let center_rank = *cluster_center.get(&cluster_rank).unwrap().value();
                let center_pt = points[center_rank].get_position();
                let dist2 = xyz
                    .iter()
                    .zip(center_pt)
                    .fold(T::zero(), |acc, (p, c)| acc + (*p - *c) * (*p - *c));
                let dist = num_traits::Float::sqrt(dist2);
                log::trace!(
                    "point : {},  center : {}, dist : {:.3e}",
                    point.get_id(),
                    center_rank,
                    dist.to_f64().unwrap()
                );
                if cluster_rank == *pt_affectation.get(&point_rank).unwrap().value() {
                    without_reaffectation_norm.fetch_add(dist.to_f64().unwrap(), Ordering::Acquire);
                }
                if dist.to_f64().unwrap() < mindist {
                    mindist = dist.to_f64().unwrap();
                    minclust = i;
                }
                new_affectation.insert(point_rank, minclust as u32);
            }
            norm.fetch_add(mindist, Ordering::Acquire);
        });
        //
        // how many points changed in reaffectation
        //
        let mut nb_changed = 0;
        for item in &new_affectation {
            let pt_id = item.key();
            if *pt_affectation.get(pt_id).unwrap().value() != *item.value() {
                nb_changed += 1;
            }
        }
        log::info!(
            "HCluster compute_cost_medoid_l2 cost before reaffectation : {:.3e}, nb change of affectation : {}",
            without_reaffectation_norm.into_inner(), nb_changed
        );
        //
        let norm_f64 = norm.into_inner();
        log::info!(
            "exiting HCluster compute_cost_medoid_l2, cost after reaffectation : {:.3e}",
            norm_f64
        );
        (new_affectation, norm_f64)
    } // end of compute_cost_medoid_l2

    //

    // cluster_center gives for a cluster rank, the rank of the point in self.points
    // pt_affectation constains the cluster rank,
    // cluster_center contains the point rank designated as the center , resulting from benefit analysis
    #[allow(unused)]
    pub(crate) fn compute_cost_medoid_l1(
        &self,
        pt_affectation: &DashMap<usize, u32>,
        cluster_center: &DashMap<u32, usize>,
    ) -> (DashMap<usize, u32>, f64) {
        //
        log::info!("in HCluster compute_cost_medoid_l1");
        //
        assert_eq!(self.points.len(), pt_affectation.len());

        let new_affectation = DashMap::<usize, u32>::new();
        let norm: AtomicF64 = AtomicF64::new(0.);
        let nb_cluster = cluster_center.len();
        let nb_points = self.points.len();
        let points = &self.points;
        log::info!("dimension : {}", self.points[0].get_dimension());
        (0..nb_points).into_par_iter().for_each(|point_rank| {
            let point = points[point_rank];
            let xyz = point.get_position();
            let mut mindist = f64::MAX;
            let mut minclust: usize = usize::MAX;
            for i in 0..nb_cluster {
                let cluster_rank: u32 = i.try_into().unwrap();
                let center_rank = *cluster_center.get(&cluster_rank).unwrap().value();
                let center_pt = points[center_rank].get_position();
                let dist = xyz.iter().zip(center_pt).fold(T::zero(), |acc, (p, c)| {
                    acc + num_traits::Float::abs(*p - *c)
                });
                log::trace!(
                    "point : {},  center : {}, dist : {:.3e}",
                    point.get_id(),
                    center_rank,
                    dist.to_f64().unwrap()
                );
                if dist.to_f64().unwrap() < mindist {
                    mindist = dist.to_f64().unwrap();
                    minclust = i;
                }
                new_affectation.insert(point_rank, minclust as u32);
            }
            norm.fetch_add(mindist, Ordering::Acquire);
        });
        //
        // how many points changed in reaffectation
        //
        let mut nb_changed = 0;
        for item in &new_affectation {
            let pt_id = item.key();
            if *pt_affectation.get(pt_id).unwrap().value() != *item.value() {
                nb_changed += 1;
            }
        }
        log::info!(
            "HCluster compute_cost_medoid_l1 nb change of affectation : {}",
            nb_changed
        );
        //
        let norm_f64 = norm.into_inner();
        log::info!(
            "exiting HCluster compute_cost_medoid_l1, cost : {:.3e}",
            norm_f64
        );
        (new_affectation, norm_f64)
    } // end of compute_cost_medoid_l1

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
    use crate::merit::*;

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
        //=====================================
        // points are generated around 5 centers/labels
        let nbvec = 1000_000usize;
        let dim = 2;
        let width: f64 = 1.;
        let nb_cluster_asked = 5;
        //=====================================
        let unif_01 = Uniform::<f64>::new(0., width).unwrap();
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(234567_u64);
        let mut points: Vec<Point<f64>> = Vec::with_capacity(nbvec);
        // We construct a  Affectation structure to compare clusters with true labels
        let ref_hashmap = DashMap::<usize, u32>::new();
        for i in 0..nbvec {
            let label = i % 5;
            let offset = label as f64 * 15.;
            let p: Vec<f64> = (0..dim)
                .map(|_| offset + unif_01.sample(&mut rng))
                .collect();
            points.push(Point::<f64>::new(i, p, (i % 5).try_into().unwrap()));
            ref_hashmap.insert(i, label as u32);
        }
        let ref_affectation = DashAffectation::new(&ref_hashmap);
        // SpaceMesh construction
        let refpoints: Vec<&Point<f64>> = points.iter().map(|p| p).collect();
        let mut hcluster = Hcluster::new(refpoints, None);
        hcluster.set_debug_level(1);
        let auto_dim = false;
        let cluster_res = hcluster.cluster(nb_cluster_asked, auto_dim, None);
        cluster_res.dump_cluster_id::<usize>(None);
        let algo_affectation = cluster_res.get_dash_affectation();
        //
        let refpoints = hcluster.get_points();

        let output = Some("cluster_random.csv");
        println!(
            "global cost : {:.3e}",
            cluster_res.compute_cost_medoid_l2(&refpoints, output)
        );
        println!("merit ctatus");
        // entropy merit
        let contingency = Contingency::<DashAffectation<usize, u32>, usize, u32>::new(
            algo_affectation,
            ref_affectation,
        );
        contingency.dump_entropies();
        let nmi_joint: f64 = contingency.get_nmi_joint();
        println!("results with {} clusters", nb_cluster_asked);
        println!("nmi joint : {:.3e}", nmi_joint);
    } //end of test_cluster_random

    #[test]
    fn test_cluster_exp() {
        log_init_test();
        log::info!("in test_uniform_random");
        //
        let nbvec = 10_00_000usize;
        let dim = 5;
        let width: f32 = 100.;
        let nb_cluster_asked = 5;
        let lambda = 5;

        // sample with coordinates following exponential law
        let law = Exp::<f32>::new(lambda as f32).unwrap();
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(234567_u64);
        let mut points: Vec<Point<f32>> = Vec::with_capacity(nbvec);
        // We construct a  Affectation structure to compare clusters with true labels
        let ref_hashmap = DashMap::<usize, u32>::new();
        for i in 0..nbvec {
            let label = i % 5;
            let offset = label as f32 * 15. as f32;
            let p: Vec<f32> = (0..dim)
                .map(|_| offset + law.sample(&mut rng).min(width) as f32)
                .collect();
            points.push(Point::<f32>::new(i, p, (i % 5).try_into().unwrap()));
            ref_hashmap.insert(i, label as u32);
        }
        let ref_affectation = DashAffectation::new(&ref_hashmap);
        // Space definition
        let refpoints: Vec<&Point<f32>> = points.iter().map(|p| p).collect();
        //
        let mut hcluster = Hcluster::new(refpoints, None);
        let auto_dim = false;
        let cluster_res = hcluster.cluster(nb_cluster_asked, auto_dim, None);
        cluster_res.dump_cluster_id::<usize>(None);
        //
        let algo_affectation = cluster_res.get_dash_affectation();
        //
        let refpoints = hcluster.get_points();
        println!(
            "global cost : {:.3e}",
            cluster_res.compute_cost_medoid_l2(&refpoints, None)
        );
        //
        // merit comparison
        println!("merit ctatus");
        //
        let contingency = Contingency::<DashAffectation<usize, u32>, usize, u32>::new(
            algo_affectation,
            ref_affectation,
        );
        contingency.dump_entropies();
        let nmi_joint: f64 = contingency.get_nmi_joint();
        println!("results with {} clusters", nb_cluster_asked);
        println!("nmi joint : {:.3e}", nmi_joint);
    } //end of test_cluster_exp
} // end of tests
