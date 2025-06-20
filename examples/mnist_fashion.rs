//! To run the examples change the line :  
//!
//! const MNIST_fashion_DIR : &'static str = "/home/jpboth/Data/MNIST/";
//!
//! to whatever directory you downloaded the [MNIST fashion data](https://www.kaggle.com/datasets/hojjatk/mnist-dataset)

// use clap::{Arg, ArgAction, ArgMatches, Command};

use std::collections::HashMap;

use cpu_time::ProcessTime;
use dashmap::DashMap;
use std::time::{Duration, SystemTime};

use edgecluster::rhst::*;
mod utils;
use nmi::*;
use utils::mnistio::*;

// for data in old non csv format
const MNIST_FASHION_DIR_NOT_CSV: &str = "/home/jpboth/Data/ANN/Fashion-MNIST";

//
//

pub fn main() {
    //
    let _ = env_logger::builder().is_test(true).try_init().unwrap();

    //

    log::info!(
        "in mnist_fashion, reading mnist data in CSV format ...from {}",
        MNIST_FASHION_DIR_NOT_CSV
    );
    let (labels, images_as_v) = io_from_non_csv(MNIST_FASHION_DIR_NOT_CSV).unwrap();

    //
    // define points
    //
    log::info!("start...");
    let points: Vec<Point<f32>> = (0..labels.len())
        .map(|i| Point::<f32>::new(i, images_as_v[i].clone(), labels[i] as u32))
        .collect();
    //
    let mut labels_distribution = HashMap::<u32, u32>::with_capacity(10);
    for p in &points {
        if let Some(count) = labels_distribution.get_mut(&p.get_label()) {
            *count += 1;
        } else {
            labels_distribution.insert(p.get_label(), 1);
        }
    }
    for (l, count) in labels_distribution {
        println!("label : {}, count : {}", l, count);
    }
    //
    let cpu_start = ProcessTime::now();
    let sys_now = SystemTime::now();
    // distance is normalized by pixel. Value of pixel between 0 and 256
    //===================================
    let nb_cluster_asked = vec![10, 15, 25];
    let auto_dim = false;
    let _small_dim = None;
    let user_layer_max = None;
    //===================================
    // cluster without specifying a dimension reducer
    let mut hcluster = Hcluster::new(&points, None);
    let cluster_res = hcluster.cluster(&nb_cluster_asked, auto_dim, _small_dim, user_layer_max);
    for (i, p) in cluster_res.iter().enumerate() {
        log::info!(" \n\n results with {} clusters", nb_cluster_asked[i]);
        log::info!("\n ====================================================");
        let algo_affectation = p.get_dash_affectation();
        // We construct a corresponding Affectation structure to compare clusters with true labels
        let ref_hashmap = DashMap::<usize, u32>::new();
        for (i, l) in labels.iter().enumerate() {
            ref_hashmap.insert(i, *l as u32);
        }
        let ref_affectation = DashAffectation::new(&ref_hashmap);
        //
        let cpu_time: Duration = cpu_start.elapsed();
        println!(
            "  sys time(ms) {:?} cpu time(ms) {:?}",
            sys_now.elapsed().unwrap().as_millis(),
            cpu_time.as_millis()
        );
        //
        let refpoints = hcluster.get_points();
        let output = Some("fashion_centers.csv");
        println!(
            "medoid l2 cost : {:.3e}",
            p.compute_cost_medoid_l2(&refpoints, output)
        );
        p.dump_cluster_id(Some(&labels));
        // merit comparison
        println!("merit ctatus");
        //
        let contingency = Contingency::<DashAffectation<usize, u32>, usize, u32>::new(
            ref_affectation,
            algo_affectation,
        );
        contingency.dump_entropies();
        let nmi_joint: f64 = contingency.get_nmi_joint();
        println!(
            "mnist fashion results with {} clusters",
            nb_cluster_asked[i]
        );
        println!("mnit fashion nmi joint : {:.3e}", nmi_joint);

        let nmi_mean: f64 = contingency.get_nmi_mean();
        println!(
            "mnist fashion results with {} clusters",
            nb_cluster_asked[i]
        );
        println!("mnist fashion nmi mean : {:.3e}", nmi_mean);

        let nmi_sqrt: f64 = contingency.get_nmi_sqrt();
        println!(
            "mnist fashion results with {} clusters",
            nb_cluster_asked[i]
        );
        println!("mnit fashion nmi sqrt : {:.3e}", nmi_sqrt);
    }
}
