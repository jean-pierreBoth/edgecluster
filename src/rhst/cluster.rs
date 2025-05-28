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

#[derive(Clone)]
pub struct ClusterResult {
    /// a map from  points id to a cluster num or label
    point_to_cluster: DashMap<usize, u32>,
    /// For its cluster, the pointId of its medoid center
    cluster_center_to_pid: DashMap<u32, usize>,
    /// for each cluster a vector of point id affected to it
    clusters: Vec<Vec<usize>>,
    ///
    cost_l2: f64,
}

impl ClusterResult {
    pub(crate) fn new(
        point_to_cluster: DashMap<usize, u32>,
        cluster_center_to_pid: DashMap<u32, usize>,
        nb_cluster: usize,
        cost_l2: f64,
    ) -> ClusterResult {
        //
        let mut clusters: Vec<Vec<usize>> = (0..nb_cluster).map(|_| Vec::<usize>::new()).collect();
        for item in point_to_cluster.iter() {
            clusters[*item.value() as usize].push(*item.key());
        }
        ClusterResult {
            point_to_cluster,
            cluster_center_to_pid,
            clusters,
            cost_l2,
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

    pub fn get_l2_medoid_cost(&self) -> f64 {
        self.cost_l2
    }

    /// dump info on clusters: centerid , size, possibly labels of centers
    /// labels : a vector containing points labels
    pub fn dump_cluster_id<LabelId>(&self, labels: Option<&Vec<LabelId>>)
    where
        LabelId: Copy + Clone + std::fmt::Display,
    {
        let mut nb_info = 1;
        for item in &self.cluster_center_to_pid {
            let rank = *item.key();
            let center_pid = *item.value();
            if labels.is_none() {
                println!(
                    "cluster : {} , center_id : {}, size : {}",
                    nb_info,
                    center_pid,
                    self.clusters[rank as usize].len()
                );
            } else {
                let label_id = labels.as_ref().unwrap()[center_pid];
                println!(
                    "cluster : {} , center_id : {}, label : {}, size : {}",
                    nb_info,
                    center_pid,
                    label_id,
                    self.clusters[rank as usize].len()
                );
            }
            nb_info += 1;
        }
    }

    /// returns the number of points in cluster i
    pub fn get_cluster_size(&self, i: usize) -> usize {
        self.clusters[i].len()
    }

    /// returns number of clusters
    pub fn get_nb_cluster(&self) -> usize {
        self.clusters.len()
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
        //
        log::info!("\n in compute_cluster_mean_center");
        //
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
        T: Float + std::fmt::Debug + std::ops::AddAssign + std::ops::DivAssign + serde::Serialize,
    >(
        &self,
        points: &[&Point<T>],
        dumpfile: Option<&str>,
    ) -> T {
        log::info!("in ClusterResult compute_cost_medoid_l2, checking centers membership");
        // check centers
        for item in &self.cluster_center_to_pid {
            let cluster = item.key();
            let pid = item.value();
            assert_eq!(*cluster, *self.point_to_cluster.get(pid).unwrap().value());
        }
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
        #[derive(Debug, Serialize)]
        struct Record<'a, T> {
            cluster: u32,
            data: &'a [T],
        }
        //
        if let Some(csvloc) = dumpfile {
            let mut csvpath = PathBuf::from(".");
            let csvloc_sized = self.get_nb_cluster().to_string() + "-" + csvloc;
            csvpath.push(csvloc_sized);
            let csvfileres = OpenOptions::new()
                .create(true)
                .truncate(true)
                .write(true)
                .open(&csvpath);
            if csvfileres.is_err() {
                println!(" could not open file {:?}", csvpath.as_os_str());
            }
            // CAVEAT: we need to set header to false to have serialization!!
            let wtr = WriterBuilder::new().has_headers(false).from_path(&csvpath);
            if wtr.is_err() {
                log::error!("compute_cost_medoid_l1 cannot open dump csv file");
            } else {
                log::info!("dumping csv file in {:?}", csvpath.as_os_str());
                let mut wtr = wtr.unwrap();
                for item in self.cluster_center_to_pid.iter() {
                    let cluster = *item.key();
                    let center_id = *self.cluster_center_to_pid.get(&cluster).unwrap().value();
                    let center_point = points[center_id];
                    let record = Record {
                        cluster,
                        data: center_point.get_position(),
                    };
                    wtr.serialize(record).unwrap();
                }
            }
            log::info!("csv dump in {}", csvloc);
        }
        //
        log::info!("exiting ClusterResult::compute_cost_medoid_l2");
        //
        norm
    }

    //
    #[cfg_attr(doc, katexit::katexit)]
    /// we compute mean of cluster and then search point in each cluster thait is
    /// the closest to the mean. this reduces marginally (~5-10%)the cost.
    /// Only useful for kmedian clustering, not for hierarchical clustering.
    ///
    /// cost returned is
    /// $ \sum_{1}^{N}  ||x_{i} - C(x_{i})|| $ where $C(x_{i})$ is the nearest cluster center for $ x_{i}$.
    /// result will be written in a csv file with name $filename$
    pub fn compute_cost_medoid_recenter_l2<
        T: Float + std::fmt::Debug + std::ops::AddAssign + std::ops::DivAssign,
    >(
        &self,
        points: &[&Point<T>],
        dumpfile: Option<&str>,
    ) -> T {
        log::info!("\n ==============================================\n in ClusterResult compute_cost_medoid_recenter_l2");
        //
        let nb_cluster = self.clusters.len();
        let dim = points[0].get_dimension();
        log::info!("dim : {}", dim);
        // first we compute mean of each cluster
        // TODO: to make //
        let mut cluster_mean: Vec<Vec<T>> = (0..nb_cluster).map(|_| vec![T::zero(); dim]).collect();
        let mut cluster_size = vec![0usize; nb_cluster];
        for point in points {
            let xyz = point.get_position();
            let key = point.get_id();
            let cluster = *self.point_to_cluster.get(&key).unwrap().value();
            for (d, x) in xyz.iter().enumerate() {
                cluster_mean[cluster as usize][d] += *x;
            }
            cluster_size[cluster as usize] += 1;
        }
        for i in 0..nb_cluster {
            assert_eq!(cluster_size[i], self.clusters[i].len());
            for x in &mut cluster_mean[i] {
                (*x) /= T::from(cluster_size[i]).unwrap();
            }
        }
        //
        // now for each cluster we must find the nearest point in this cluster to the mean of cluster
        // we loop on clusters
        let mut nb_seen_by_cluster = vec![0usize; nb_cluster];
        let mut best_point_id_by_cluster = vec![0usize; nb_cluster];
        let mut best_dist_to_cluster = vec![T::max_value(); nb_cluster];
        for cluster in self.clusters.iter() {
            for p in cluster {
                let point = points[*p];
                let xyz = point.get_position();
                let key = point.get_id();
                let cluster_rank = *self.point_to_cluster.get(&key).unwrap().value() as usize;
                let dist_to_center = xyz
                    .iter()
                    .zip(&cluster_mean[cluster_rank])
                    .fold(T::zero(), |acc, (p, c)| acc + (*p - *c) * (*p - *c))
                    .sqrt();
                nb_seen_by_cluster[cluster_rank] += 1;
                if dist_to_center < best_dist_to_cluster[cluster_rank] {
                    best_dist_to_cluster[cluster_rank] = dist_to_center;
                    best_point_id_by_cluster[cluster_rank] = point.get_id();
                }
            }
        }
        //
        let mut norm = T::zero();
        for point in points {
            let xyz = point.get_position();
            let key = point.get_id();
            let cluster = *self.point_to_cluster.get(&key).unwrap().value();
            let center_id = best_point_id_by_cluster[cluster as usize];
            let center_point = points[center_id];
            norm += xyz
                .iter()
                .zip(center_point.get_position())
                .fold(T::zero(), |acc, (p, c)| acc + (*p - *c) * (*p - *c))
                .sqrt()
        }
        //
        let total_cost_f: f64 = norm.to_f64().unwrap();
        log::info!(
            "\n cost after recomputing mean center : {:.3e}",
            total_cost_f
        );
        log::info!("===============================================");
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
    //
    user_layer_max: Option<u16>,
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
            user_layer_max: None,
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

    pub fn get_user_layer_max(&self) -> Option<u16> {
        self.user_layer_max
    }

    pub fn cluster_one(
        &mut self,
        nb_cluster: usize,
        auto_dim: bool,
        reduced_dim_opt: Option<usize>,
        user_layer_max: Option<u16>,
    ) -> ClusterResult {
        let partitons_size = vec![nb_cluster];
        let res = self.cluster(&partitons_size, auto_dim, reduced_dim_opt, user_layer_max);
        res[0].clone()
    }

    /// The function that triggers the clustering.  
    /// The arguments are:  
    /// - number of clusters asked for
    /// - auto_dim : set to true, the algorithm will try to reduce data dimension using module [crate::smalld]
    /// - if a reduction  to a given dimension is necessary, this option will use it.  
    ///
    /// The function returns a map giving for each point id its cluster
    pub fn cluster(
        &mut self,
        partitions_size: &Vec<usize>,
        auto_dim: bool,
        reduced_dim_opt: Option<usize>,
        user_layer_max: Option<u16>,
    ) -> Vec<ClusterResult> {
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
        let reduced_dim = reduced_dim_opt.unwrap_or_default();
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
        let mut spacemesh = SpaceMesh::new(&mut space, points_to_cluster, user_layer_max);
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
        let (point_in_clusters, cluster_to_center_pid) =
            spacemesh.get_partition(&partitions_size, &filtered_benefits);
        //
        let nb_partitions = partitions_size.len();
        let mut res = Vec::<ClusterResult>::with_capacity(nb_partitions);
        for i in 0..partitions_size.len() {
            // rebuild point affectation for each partition
            let point_to_cluster: DashMap<usize, u32> = DashMap::<usize, u32>::new();
            let cluster_to_center_pid_for_this_partition = DashMap::<u32, usize>::new();
            //
            let nb_cluster = partitions_size[i];
            for c in 0..nb_cluster as u32 {
                let item = cluster_to_center_pid.get(&c).unwrap();
                let p_id = *item.value();
                cluster_to_center_pid_for_this_partition.insert(c, p_id);
            }
            // point affectation at this partition
            for item in &point_in_clusters {
                let p_id = item.key();
                let c = item.value()[i];
                if log::log_enabled!(log::Level::Info) {
                    if (c as usize) >= partitions_size[i] {
                        log::error!(
                            "partition i : {} c : {}, partitions_size[i] : {}",
                            i,
                            c,
                            partitions_size[i]
                        );
                    }
                }
                assert!((c as usize) < partitions_size[i]);
                point_to_cluster.insert(*p_id, c);
            }

            // keep centers and reaffect, computing new cost
            let (point_reaffectation, medoid_cost) = self.compute_reaffectation_cost_medoid_l2(
                &point_to_cluster,
                &cluster_to_center_pid_for_this_partition,
            );
            log::info!(
                "medoid cost in original space after reaffectation (l2) : {:.3e} with {} clusters",
                medoid_cost,
                partitions_size[i]
            );
            //
            let partition = ClusterResult::new(
                point_reaffectation,
                cluster_to_center_pid_for_this_partition,
                partitions_size[i],
                medoid_cost,
            );
            //        res.compute_cost_medoid_recenter_l2(&self.points, None);
            //
            res.push(partition);
        }
        println!(
            " Cluster (dimension reduction + embedding + partitioning) time(s) {:?} cpu time {:?}",
            sys_now.elapsed().unwrap().as_secs(),
            cpu_start.elapsed().as_secs()
        );
        //
        res
    }

    // pt_affectation constains the cluster rank,
    // cluster_center contains the point_id designated as the center , resulting from benefit analysis
    // returns mean L2 cost
    #[allow(unused)]
    pub(crate) fn compute_reaffectation_cost_medoid_l2(
        &self,
        pt_affectation: &DashMap<usize, u32>,
        cluster_center: &DashMap<u32, usize>,
    ) -> (DashMap<usize, u32>, f64) {
        //
        log::info!("in HCluster compute_cost_medoid_l2");
        //
        assert_eq!(self.points.len(), pt_affectation.len());
        //
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
                let dist = num_traits::Float::sqrt(dist2).to_f64().unwrap();
                log::trace!(
                    "point : {},  center : {}, dist : {:.3e}",
                    point.get_id(),
                    center_rank,
                    dist
                );
                if cluster_rank == *pt_affectation.get(&point.get_id()).unwrap().value() {
                    without_reaffectation_norm.fetch_add(dist, Ordering::Acquire);
                }
                if dist < mindist {
                    mindist = dist;
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
        let norm_f64 = norm.into_inner();
        log::info!(
            "HCluster compute_cost_medoid_l2 cost without reaffectation : {:.3e}, nb change of affectation : {}, cost after reaffectation : {:.3e}",
            without_reaffectation_norm.into_inner(), nb_changed, norm_f64
        );
        //
        (new_affectation, norm_f64)
    } // end of compute_reaffectation_cost_medoid_l2

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
        let user_layer_max = None;
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
        //
        let cluster_res = hcluster.cluster_one(nb_cluster_asked, auto_dim, None, user_layer_max);
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
        let cluster_res = hcluster.cluster_one(nb_cluster_asked, auto_dim, None, None);
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
