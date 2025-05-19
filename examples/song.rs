//! bench on Song data

//! To run the examples change the line :  
//!
//! const SONG_DIR : &'static str = "/home/jpboth/Data/BenchClustering/";
//!
//! to whatever directory you downloaded the [SONG](https://archive.ics.uci.edu/dataset/203/yearpredictionmsd)
// use clap::{Arg, ArgAction, ArgMatches, Command};

// 91 columns. fist is year as an interger, then f32
// total 515345 rows
// train: first 463,715 examples
// test: last 51,630 examples

use anyhow::*;

use std::io::BufReader;
use std::path::PathBuf;

use std::{collections::HashMap, fs::OpenOptions};

use cpu_time::ProcessTime;
use dashmap::DashMap;
use std::time::{Duration, SystemTime};

use csv;

use edgecluster::merit::*;
use edgecluster::rhst::*;

const SONG_DIR: &str = "/home/jpboth/Data/BenchClustering/";

fn read_song_csv(
    fname: &str,
    nb_column: usize,
    // subsampling factor
) -> anyhow::Result<(Vec<u32>, Vec<Vec<f32>>)> {
    //
    let nb_fields = 91;
    let nb_var = nb_column - 1;
    let nb_record_to_read = 515344;
    let mut num_record: usize = 0;
    let mut filepath = PathBuf::from("/home/jpboth/Data/BenchClustering");
    filepath.push(fname);
    let fileres = OpenOptions::new().read(true).open(&filepath);
    if fileres.is_err() {
        log::error!("read_song_csv {:?}", filepath.as_os_str());
        println!("read_song_csv {:?}", filepath.as_os_str());
        return Err(anyhow!(
            "directed_from_csv could not open file {}",
            filepath.display()
        ));
    }
    let file = fileres?;
    let bufreader = BufReader::new(file);
    let mut labels = Vec::<u32>::new();
    let mut data = Vec::<Vec<f32>>::with_capacity(515345);
    let mut rdr = csv::Reader::from_reader(bufreader);
    //
    //
    for result in rdr.records() {
        // The iterator yields Result<StringRecord, Error>, so we check the
        // error here.
        num_record += 1;
        //
        if num_record % 1_00_000 == 0 {
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
        let mut new_data = Vec::<f32>::with_capacity(nb_var);
        for j in 0..nb_column {
            let field = record.get(j).unwrap();
            // decode into Ix type
            if j == 0 {
                if let std::result::Result::Ok(val) = field.parse::<u32>() {
                    labels.push(val);
                } else {
                    log::debug!("error decoding field  of record {}", num_record);
                    return Err(anyhow!("error decoding field 1of record  {}", num_record));
                }
            } else {
                if let std::result::Result::Ok(val) = field.parse::<f32>() {
                    new_data.push(val);
                } else {
                    log::debug!("error decoding field  of record {}", num_record);
                    return Err(anyhow!("error decoding field 1of record  {}", num_record));
                }
            }
        } // end for j
        assert_eq!(new_data.len(), nb_var);
        data.push(new_data);
    }
    //
    assert_eq!(nb_record_to_read, data.len());
    log::info!("number of records loaded : {:?}", data.len());
    //
    Ok((labels, data))
} // end of read_song_csv

fn main() {
    //
    let _ = env_logger::builder().is_test(true).try_init().unwrap();
    log::info!("in song examples, reading data...from {}", SONG_DIR);
    //
    let nb_column = 91;
    //
    let io_res = read_song_csv("YearPredictionMSD.txt", nb_column);
    if io_res.is_err() {
        log::error!(
            "could not open file : {} from dir : {}",
            "YearPredictionMSD.txt",
            SONG_DIR
        );
        std::process::exit(1);
    }
    let (labels, records) = io_res.unwrap();
    //
    log::info!("start...");
    let points: Vec<Point<f32>> = (0..labels.len())
        .map(|i| Point::<f32>::new(i, records[i].clone(), labels[i] as u32))
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
    //
    //===================================
    let nb_cluster_asked = 100;
    let auto_dim = false;
    let _small_dim = Some(3);
    //===================================
    //
    // cluster without specifying a dimension reducer
    let mut hcluster = Hcluster::new(ref_points, None);
    //
    let cluster_res = hcluster.cluster(nb_cluster_asked, auto_dim, _small_dim);
    //
    // We construct a corresponding Affectation structure to compare clusters with true labels
    let ref_hashmap = DashMap::<usize, u32>::new();
    for (i, l) in labels.iter().enumerate() {
        ref_hashmap.insert(i, *l as u32);
    }
    let ref_affectation = DashAffectation::new(&ref_hashmap);
    let algo_affectation = cluster_res.get_dash_affectation();
    let ref_points = hcluster.get_points();
    //
    let cpu_time: Duration = cpu_start.elapsed();
    println!(
        "  sys time(ms) {:?} cpu time(ms) {:?}",
        sys_now.elapsed().unwrap().as_millis(),
        cpu_time.as_millis()
    );
    //
    let output = Some("song.csv");
    println!(
        "global cost : {:.3e}",
        cluster_res.compute_cost_medoid_l1(&ref_points, output)
    );
    // merit comparison
    println!("merit ctatus");
    //
    let contingency = Contingency::<DashAffectation<usize, u32>, usize, u32>::new(
        ref_affectation,
        algo_affectation,
    );
    contingency.dump_entropies();
    let nmi_joint: f64 = contingency.get_nmi_joint();
    println!("SONG results with {} clusters", nb_cluster_asked);
    println!("SONG nmi joint : {:.3e}", nmi_joint);

    let nmi_mean: f64 = contingency.get_nmi_mean();
    println!("SONG results with {} clusters", nb_cluster_asked);
    println!("SONG nmi mean : {:.3e}", nmi_mean);

    let nmi_sqrt: f64 = contingency.get_nmi_sqrt();
    println!("SONG results with {} clusters", nb_cluster_asked);
    println!("SONG nmi sqrt : {:.3e}", nmi_sqrt);
} // end of main
