//! To run the examples change the line :  
//!
//! const MNIST_DIGITS_DIR : &'static str = "/home/jpboth/Data/MNIST/";
//!
//! to whatever directory you downloaded the [MNIST digits data](https://www.kaggle.com/datasets/hojjatk/mnist-dataset)

// use clap::{Arg, ArgAction, ArgMatches, Command};
use ndarray::s;
use std::path::PathBuf;
use std::{collections::HashMap, fs::OpenOptions};

use cpu_time::ProcessTime;
use dashmap::DashMap;
use std::time::{Duration, SystemTime};

use edgecluster::rhst::*;
mod utils;
use edgecluster::merit::*;
use utils::mnistio::*;

const MNIST_DIGITS_DIR: &str = "/home/jpboth/Data/ANN/MNIST/";

pub fn main() {
    //
    let _ = env_logger::builder().is_test(true).try_init().unwrap();
    log::info!(
        "in mnist_digits, reading mnist data...from {}",
        MNIST_DIGITS_DIR
    );
    //
    let mut image_fname = String::from(MNIST_DIGITS_DIR);
    image_fname.push_str("train-images-idx3-ubyte");
    let image_path = PathBuf::from(image_fname.clone());
    let image_file_res = OpenOptions::new().read(true).open(image_path);
    if image_file_res.is_err() {
        println!("could not open image file : {:?}", image_fname);
        return;
    }
    let mut label_fname = String::from(MNIST_DIGITS_DIR);
    label_fname.push_str("train-labels-idx1-ubyte");
    let label_path = PathBuf::from(label_fname.clone());
    let label_file_res = OpenOptions::new().read(true).open(&label_path);
    if label_file_res.is_err() {
        println!("could not open label file : {:?}", label_fname);
        return;
    }
    let mut images_as_v: Vec<Vec<f32>>;
    let mut labels: Vec<u8>;
    {
        let mnist_train_data = MnistData::new(image_fname, label_fname).unwrap();
        let images = mnist_train_data.get_images();
        labels = mnist_train_data.get_labels().to_vec();
        let (_, _, nbimages) = images.dim();
        //
        images_as_v = Vec::<Vec<f32>>::with_capacity(nbimages);
        for k in 0..nbimages {
            let v: Vec<f32> = images
                .slice(s![.., .., k])
                .iter()
                .map(|v| *v as f32 / (28. * 28.))
                .collect();
            images_as_v.push(v);
        }
    } // drop mnist_train_data
      // now read test data
    let mut image_fname = String::from(MNIST_DIGITS_DIR);
    image_fname.push_str("t10k-images-idx3-ubyte");
    let image_path = PathBuf::from(image_fname.clone());
    let image_file_res = OpenOptions::new().read(true).open(image_path);
    if image_file_res.is_err() {
        println!("could not open image file : {:?}", image_fname);
        return;
    }
    let mut label_fname = String::from(MNIST_DIGITS_DIR);
    label_fname.push_str("t10k-labels-idx1-ubyte");
    let label_file_res = OpenOptions::new().read(true).open(&label_path);
    if label_file_res.is_err() {
        println!("could not open label file : {:?}", label_fname);
        return;
    }
    {
        let mnist_test_data = MnistData::new(image_fname, label_fname).unwrap();
        let test_images = mnist_test_data.get_images();
        let mut test_labels = mnist_test_data.get_labels().to_vec();
        let (_, _, nbimages) = test_images.dim();
        let mut test_images_as_v = Vec::<Vec<f32>>::with_capacity(nbimages);
        //
        for k in 0..nbimages {
            let v: Vec<f32> = test_images
                .slice(s![.., .., k])
                .iter()
                .map(|v| *v as f32 / (28. * 28.))
                .collect();
            test_images_as_v.push(v);
        }
        labels.append(&mut test_labels);
        images_as_v.append(&mut test_images_as_v);
    } // drop mnist_test_data
      //
      // define points
      //
    log::info!("start...");
    let points: Vec<Point<f32>> = (0..labels.len())
        .map(|i| Point::<f32>::new(i, images_as_v[i].clone(), labels[i] as u32))
        .collect();
    let ref_points: Vec<&Point<f32>> = points.iter().map(|p| p).collect();
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
    let nb_cluster_asked = 20;
    let auto_dim = false;
    let _small_dim = Some(3);
    //===================================
    // cluster without specifying a dimension reducer
    let mut hcluster = Hcluster::new(ref_points, None);
    let cluster_res = hcluster.cluster(nb_cluster_asked, auto_dim, _small_dim);
    let algo_affectation = cluster_res.get_dash_affectation();
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
    let output = Some("digits_centers.csv");
    println!(
        "medoid l1 cost : {:.3e}",
        cluster_res.compute_cost_medoid_l1(&refpoints, output)
    );
    cluster_res.dump_cluster_id();
    // merit comparison
    println!("merit ctatus");
    //
    let contingency = Contingency::<DashAffectation<usize, u32>, usize, u32>::new(
        ref_affectation,
        algo_affectation,
    );
    contingency.dump_entropies();
    let nmi_joint: f64 = contingency.get_nmi_joint();
    println!("mnist digits results with {} clusters", nb_cluster_asked);
    println!("mnit disgit nmi joint : {:.3e}", nmi_joint);

    let nmi_mean: f64 = contingency.get_nmi_mean();
    println!("mnist digits results with {} clusters", nb_cluster_asked);
    println!("mnit disgit nmi mean : {:.3e}", nmi_mean);

    let nmi_sqrt: f64 = contingency.get_nmi_sqrt();
    println!("mnist digits results with {} clusters", nb_cluster_asked);
    println!("mnit disgit nmi sqrt : {:.3e}", nmi_sqrt);
}
