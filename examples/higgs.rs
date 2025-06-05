//! test of clustering for HIGGS boson data that consists in 11 millions of points in dimension 21 or 28 if we use
//! also the variables hand crafted by physicists.    
//! The data is described and can be retrieved at <https://archive.ics.uci.edu/ml/datasets/HIGGS>.
//! An example of this data set processing is given in the paper by Amid and Warmuth
//! Cf <https://arxiv.org/abs/1910.00204>
//!
//!
//! - With embedding **dimension 2** and embedding neighbourhood size 100 we get :
//! ```text
//! ```
//!
//! Only 43% of points have some neighbours conserved and 50% of points need at 1.7 * the radius of embedded neighbour considered to retrieve all
//! their neighbours.
//!
//! - With embedding **dimension 15** and embedding neighbourhood size 100 we get :
//! ```text
//! a guess at quality
//!  neighbourhood size used in embedding : 6
//!  nb neighbourhoods without a match : 104259,  mean number of neighbours conserved when match : 5.783e0
//!  embedded radii quantiles at 0.05 : 7.73e-2 , 0.25 : 1.14e-1, 0.5 :  2.20e-1, 0.75 : 5.49e-1, 0.85 : 6.28e-1, 0.95 : 7.52e-1
//!
//! statistics on conservation of neighborhood (of size nbng)
//! neighbourhood size used in target space : 100
//! quantiles on ratio : distance in embedded space of neighbours of origin space / distance of last neighbour in embedded space
//! quantiles at 0.05 : 4.43e-2 , 0.25 : 1.11e-1, 0.5 :  2.49e-1, 0.75 : 5.50e-1, 0.85 : 7.70e-1, 0.95 : 1.27e0
//! ```
//! So over 1_600_000 nodes only 100_000 do not retrieve their neighbours. 85% of points have their neighbours retrieved within 0.8 * the radius of
//! embedded neighbour considered.
//!
//!

#![allow(unused)]

use anyhow::anyhow;

use clap::{Arg, ArgAction, ArgMatches, Command};
use std::collections::HashMap;

use dashmap::DashMap;
use std::fs::OpenOptions;
use std::io::BufReader;
use std::path::{Path, PathBuf};

use rand::distr::{Distribution, Uniform};

use csv::Writer;

use ndarray::{Array2, ArrayView};

use cpu_time::ProcessTime;
use std::time::{Duration, SystemTime};

use clustering::{kmeans, Elem};

use edgecluster::rhst::*;
mod utils;
use nmi::*;

const HIGGS_DIR: &str = "/home/jpboth/Data";

/// return a vector of labels, and a list of vectors to embed
/// First field of record is label, then the 21 following field are the data.
/// 11 millions records!
fn read_higgs_csv(
    fname: PathBuf,
    nb_column: usize,
    // subsampling factor
    subsampling_factor: f64,
) -> anyhow::Result<(Vec<u8>, Array2<f32>)> {
    //
    let nb_fields = 29;
    let to_parse = nb_column;
    let nb_var = nb_column - 1;
    let mut num_record: usize = 0;
    let filepath = PathBuf::from(fname);
    let fileres = OpenOptions::new().read(true).open(&filepath);
    if fileres.is_err() {
        log::error!("read_higgs_csv {:?}", filepath.as_os_str());
        println!("read_higgs_csv {:?}", filepath.as_os_str());
        return Err(anyhow!(
            "directed_from_csv could not open file {}",
            filepath.display()
        ));
    }
    let file = fileres?;
    let bufreader = BufReader::new(file);
    let mut labels = Vec::<u8>::new();
    let mut data = Array2::<f32>::zeros((0, nb_var));
    let mut rdr = csv::Reader::from_reader(bufreader);
    //
    let unif_01 = Uniform::<f64>::new(0., 1.).unwrap();
    let mut rng = rand::rng();
    //
    for result in rdr.records() {
        // The iterator yields Result<StringRecord, Error>, so we check the
        // error here.
        num_record += 1;
        // sample if we load this record
        let xsi = unif_01.sample(&mut rng);
        if xsi >= subsampling_factor {
            continue;
        }
        //
        if num_record % 1_000_000 == 0 {
            log::info!("read {} record", num_record);
        }
        let record = result?;
        if record.len() != nb_fields {
            println!("record {} record has {} fields", num_record, record.len());
            return Err(anyhow!(
                "record {} record has {} fields",
                num_record,
                record.len()
            ));
        }
        let mut new_data = Vec::<f32>::with_capacity(21);
        for j in 0..to_parse {
            let field = record.get(j).unwrap();
            // decode into Ix type
            if let Ok(val) = field.parse::<f32>() {
                match j {
                    0 => {
                        labels.push(if val > 0. { 1 } else { 0 });
                    }
                    _ => {
                        new_data.push(val);
                    }
                };
            } else {
                log::debug!("error decoding field  of record {}", num_record);
                return Err(anyhow!("error decoding field 1of record  {}", num_record));
            }
        } // end for j
        assert_eq!(new_data.len(), nb_var);
        data.push_row(ArrayView::from(&new_data)).unwrap();
    }
    //
    assert_eq!(data.dim().0, labels.len());
    log::info!("number of records loaded : {:?}", data.dim().0);
    //
    Ok((labels, data))
} // end of read_higgs_csv

// refromat and possibly rescale
fn reformat(data: &mut Array2<f32>, rescale: bool) -> Vec<Vec<f32>> {
    let (nb_row, nb_col) = data.dim();
    let mut datavec = Vec::<Vec<f32>>::with_capacity(nb_row);
    //
    if rescale {
        for j in 0..nb_col {
            let mut col = data.column_mut(j);
            let mean = col.mean().unwrap();
            let sigma = col.var(1.).sqrt();
            col.mapv_inplace(|x| (x - mean) / sigma);
        }
    }
    // reformat in vetors
    for i in 0..nb_row {
        datavec.push(data.row(i).to_vec());
    }
    //
    datavec
} // end of reformat

// just to compare computing times
fn do_kmeans(points: &Vec<Point<f32>>, nb_iter: usize, nb_cluster: usize) {
    log::info!("doing do_kmeans comparison");
    // going to kmean
    log::info!("doing kmean clustering on whole data .... takes time");
    let data_points: Vec<&[f32]> = points.iter().map(|p| p.get_position()).collect();
    //
    let cpu_start: ProcessTime = ProcessTime::now();
    let sys_now = SystemTime::now();
    //
    let clustering = kmeans(nb_cluster, &data_points, nb_iter);
    println!(
        "kmeans total sys time(s) {:.2e}  cpu time {:.2e}",
        sys_now.elapsed().unwrap().as_secs(),
        cpu_start.elapsed().as_secs()
    );
    // compute error
    let centroids = &clustering.centroids;
    // conver centroids to vectors
    let mut centers = Vec::<Vec<f32>>::with_capacity(nb_cluster);
    for c in centroids {
        let dim = c.dimensions();
        let mut center = Vec::<f32>::with_capacity(dim);
        for i in 0..dim {
            center.push(c.at(i) as f32);
        }
        centers.push(center);
    }
    let elements = clustering.elements;
    let membership = clustering.membership;
    let mut error = 0.0;
    for i in 0..elements.len() {
        let cluster = membership[i];
        let dist = elements[i]
            .iter()
            .zip(centers[cluster].iter())
            .fold(0., (|acc, (a, b)| acc + (*a - *b) * (*a - *b)))
            .sqrt();

        error += dist;
    }
    log::info!("kmean error : {:.3e}", error / data_points.len() as f32);
}

//====================================================================================

///  By defaut a umap like embedding is done.
///  The command takes the following args:
///
///  * --dmap is used it is a dmap embedding
///   
///  * --factor sampling_factor
///       sampling_factor : if >= 1. full data is embedded, but quality runs only with 64Gb for sampling_factor <= 0.15  
///  * --dist "DistL2" or "DistL1"
///
///  The others variables can be modified in the code
///
///  - nb_col          : number of columns to read, 22 or 29  
///  - rescale         : true, can be set to false to check possible effect (really tiny)  
///  - asked_dim       : default to 2 but in conjunction with sampling factor, can see the impact on quality.  
pub fn main() {
    //
    let _ = env_logger::builder().is_test(true).try_init();
    //
    let higgcmdarg = Command::new("higgs")
        .arg(
            Arg::new("subsampling")
                .long("factor")
                .required(false)
                .action(ArgAction::Set)
                .value_parser(clap::value_parser!(f64))
                .default_value("1.0")
                .help("subsampling factor between 0. and 1."),
        )
        .get_matches();
    //
    //
    let mut fname = PathBuf::from(HIGGS_DIR);
    // parameters to play with
    // choose if we run on 22 or 29 columns id estimation on 21 or 28 variables
    // first column is label. We have one column more than variables
    //====================
    let nb_col = 29;
    let rescale = false;
    // quality estimation requires subsampling factor of 0.15 is Ok with 64Gb
    let sampling_factor = 1.;
    //====================
    let nb_var = nb_col - 1;
    //
    fname.push("HIGGS.csv");
    //
    let res = read_higgs_csv(fname, nb_col, sampling_factor);
    if res.is_err() {
        log::error!("error reading Higgs.csv {:?}", &res.as_ref().err().as_ref());
        std::process::exit(1);
    }
    let mut res = res.unwrap();
    let labels = res.0;
    // =====================
    let data = reformat(&mut res.1, rescale);
    drop(res.1); // we do not need res.1 anymore
    assert_eq!(data.len(), labels.len());
    //
    // define points
    //
    log::info!("start...");
    let points: Vec<Point<f32>> = (0..labels.len())
        .map(|i| Point::<f32>::new(i, data[i].clone(), labels[i] as u32))
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
    // now we do clustering
    //
    let sys_now = SystemTime::now();
    let cpu_start = ProcessTime::now();
    //
    //===================================
    let nb_cluster_asked = vec![10, 100, 200];
    let auto_dim = false;
    let _small_dim = Some(3);
    let user_layer_max = Some(9);
    //===================================
    // cluster without specifying a dimension reducer
    let mut hcluster = Hcluster::new(&points, None);
    let cluster_res = hcluster.cluster(&nb_cluster_asked, auto_dim, None, user_layer_max);
    //
    // dump results
    //
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
        let output = Some("higgs_centers-raw2.csv");
        println!(
            "medoid l2 cost : {:.3e}",
            p.compute_cost_medoid_l2(&refpoints, output)
        );
        //
        let (_, kmean_cost) = p.compute_cluster_kmean_centers::<f32>(refpoints);
        log::info!("\n kmeans cost : {:.3e}", kmean_cost);

        p.dump_cluster_id(Some(&labels));

        // merit comparison
        println!("merit ctatus");
        // reference is first arg so it will correspond to rows
        let contingency = Contingency::<DashAffectation<usize, u32>, usize, u32>::new(
            ref_affectation,
            algo_affectation,
        );
        contingency.dump_entropies();
        let nmi_joint: f64 = contingency.get_nmi_joint();
        println!("higgs results with {} clusters", nb_cluster_asked[i]);
        println!("higgs nmi joint : {:.3e}", nmi_joint);

        let nmi_mean: f64 = contingency.get_nmi_mean();
        println!("higgs results with {} clusters", nb_cluster_asked[i]);
        println!("higgs nmi mean : {:.3e}", nmi_mean);

        let nmi_sqrt: f64 = contingency.get_nmi_sqrt();
        println!("higgs results with {} clusters", nb_cluster_asked[i]);
        println!("higgs nmi sqrt : {:.3e}", nmi_sqrt);
        //
        // detailed contingency analysis. Here we have reference indexed by rows in contingency table
        //
        let labels = contingency.get_labels_rank(0);
        let (nb_row, nb_col) = contingency.get_dim();
        log::info!("\n \n display rows entropies");
        let row_entropies = contingency.get_row_entropies();
        for (i, c) in row_entropies.iter().enumerate() {
            log::info!("cluster : {}, row : {}, entropy : {:.3e}", labels[i], i, c,);
        }
        log::info!("\n");
        for i in 0..nb_row {
            log::info!("row : {} {}", i, contingency.get_row(i));
        }
        // display entropies by column
        // reference is second argument in Contingency allocation, so it in columns
        //
        log::info!("\n \n display colmuns entropies");
        let col_entropies = contingency.get_col_entropies();
        for (i, c) in col_entropies.iter().enumerate() {
            log::info!(
                "cluster : {}, entropy : {:.3e}, col : {}",
                i,
                c,
                contingency.get_col(i)
            );
        }
    }
    println!(
        " clustering total sys time(s) {:.2e}  cpu time {:.2e}",
        sys_now.elapsed().unwrap().as_secs(),
        cpu_start.elapsed().as_secs()
    );
    //
    // kmeans comparison
    //
    // do_kmeans(&points, 50, 200);
} // end of main
