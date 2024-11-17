#![allow(clippy::needless_range_loop)]

//! This is a small rhst implementation as described in Cohen-Addad paper
//
// We prefer rhst over cover trees as described in
//  TerraPattern: A Nearest Neighbor Search Service (2019)
//  Zaheer, Guruganesh, Levin Smola
//  as we can group points very close (under lower distance threshold) in one cell

use cpu_time::ProcessTime;
use std::time::{Duration, SystemTime};

use num_traits::cast::*;
use num_traits::float::Float;

use std::fmt::Debug;

use anyhow::Result;
use dashmap::{iter, mapref::one, rayon::*, DashMap};
use ego_tree::{tree, NodeMut, Tree};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use rayon::prelude::*;
use std::collections::HashMap;

use super::point::*;

#[cfg_attr(doc, katexit::katexit)]
/// structure describing space in which data point are supposed to live.
/// Data are supposed to live in an enclosing box $$[xmin, xmax]^d$$
///
/// The algorithm split the data space in cubic cells.
/// At each layer the edge cells are divided by 2, so the number of cells
/// increase  with the data dimension.
/// The increase is not exponential as we keep only cells with data points in it so the numbers of cells is
/// bounded by the number of points in the data set.  
/// Another characteristic is that the volume of a cell decrease by a factor $2^d$ at each layer, so $2^{(d*nblayer)}$ should be related to
/// the number of points.
///
/// Nevertheless it requires data to have small/moderated dimension. This can be achieved using a first step of dimension reduction
/// via random projections or an embedding via the crate [annembed](https://crates.io/crates/annembed)
#[derive(Debug, Copy, Clone)]
pub struct Space {
    // space dimension
    dim: usize,
    // global amplitude in each dimension
    width: f64,
    // smallest coordinate
    xmin: f64,
    // minimum distance we want to discriminate
    mindist: f64,
    //
    nb_layer: usize,
}

impl Space {
    /// define Space.
    /// -
    ///
    pub fn new(dim: usize, xmin: f64, xmax: f64, mindist: f64) -> Self {
        let width = xmax - xmin;
        assert!(
            mindist > 0.,
            "mindist cannot be 0, should related to width so that max layer is not too large"
        );
        let layer_max_u: usize = ((dim as f64).sqrt() * width / mindist).log2().trunc() as usize;
        let nb_layer = layer_max_u + 1;
        let mut accepted_mindist: f64 = mindist;
        log::info!("nb layer : {}", nb_layer);
        if (nb_layer > 16) {
            log::error!("too many layers, mindist is too small");
            accepted_mindist = (dim as f64).sqrt() * width / 2_i32.pow(16) as f64;
            log::warn!("resetting mindist to : {:.3e}", accepted_mindist);
            let check = ((dim as f64).sqrt() * width / accepted_mindist)
                .log2()
                .ceil()
                .trunc();
            log::info!("check : {:.3e}", check);
            assert!(
                ((dim as f64).sqrt() * width / accepted_mindist)
                    .log2()
                    .ceil()
                    .trunc() as usize
                    <= 15
            );
        }
        let layer_max: u32 = layer_max_u.try_into().unwrap();
        Space {
            dim,
            width: xmax - xmin,
            xmin,
            mindist,
            nb_layer,
        }
    }

    /// return space dimension
    pub fn get_dim(&self) -> usize {
        self.dim
    }
    /// return space width
    pub fn get_width(&self) -> f64 {
        self.width
    }

    /// fn get_xmin
    pub fn get_xmin(&self) -> f64 {
        self.xmin
    }
}

//========================================

/// space is split in cells , at upper layer  there is one cell for the whole space
/// at layer 0 cells have size corresponding to mindist, so a cell, even at layer 0 can have more than one point
/// (they must be lesst than mindist apart)
/// - knowing the index of cell c at layer i we can deduce the index of parent cell at layer j < i (take  indexes modulo 2^(i-j))

#[derive(Debug, Clone)]
pub(crate) struct Cell<'a, T> {
    space: &'a Space,
    // we must know one' layer
    layer: u16,
    /// a vector of dimension d, giving center of cell in the mesh coordinates
    index: Vec<u16>,
    //
    subtree_size: u32,
    //
    points_in: Option<Vec<&'a Point<T>>>,
}

impl<'a, T> Cell<'a, T>
where
    T: Float + Debug,
{
    pub fn new(space: &'a Space, layer: u16, index: Vec<u16>) -> Self {
        assert!((layer as usize) <= space.nb_layer);
        Cell {
            space,
            layer,
            index,
            subtree_size: 0,
            points_in: None,
        }
    }

    //
    pub(crate) fn get_layer(&self) -> usize {
        self.layer as usize
    }

    /// Return cel weight. This is also the cardinal of the subtree corresponding to a point in cell
    /// as long as points have no weights
    pub fn get_cell_weight(&self) -> f32 {
        if let Some(points) = self.points_in.as_ref() {
            points.len() as f32
        } else {
            0.
        }
    } // end of get_cell_weight

    /// Return number of points.
    pub(crate) fn get_nb_points(&self) -> usize {
        if let Some(points) = self.points_in.as_ref() {
            points.len()
        } else {
            0
        }
    } // end of get_nb_point

    fn get_cell_index(&self) -> &[u16] {
        &self.index
    }

    // get parent cell in splitting process
    pub(crate) fn get_upper_cell_index(&self) -> Vec<u16> {
        assert!(self.layer as usize != self.space.nb_layer - 1);
        self.index.iter().map(|x| x / 2).collect::<Vec<u16>>()
    }

    // get parent cell in splitting process
    pub(crate) fn get_upper_cell_index_at_layer(&self, l: u16) -> Vec<u16> {
        assert!(self.layer < l && (l as usize) < self.space.nb_layer);
        // compute new index by dividing by 2_u16.pow((l - self.layer) as u32);
        self.index
            .iter()
            .map(|x| x >> (l - self.layer))
            .collect::<Vec<u16>>()
    }

    // subtree size are computed by reverse of splitting in SpaceMesh::compute_subtree_size
    pub(crate) fn get_subtree_size(&self) -> u32 {
        self.subtree_size
    }

    // adds a point in cell
    fn add_point(&mut self, point: &'a Point<T>) {
        match self.points_in.as_mut() {
            Some(points) => points.push(point),
            None => {
                let mut vec = vec![point];
                self.points_in = Some(vec);
            }
        }
    }

    // fill in points
    fn init_points(&mut self, points: &[&'a Point<T>]) {
        assert!(self.points_in.is_none());
        self.points_in = Some(<Vec<&'a Point<T>>>::from(points));
    }

    // given layer and index in mesh, returns cooridnates of center of mesh in space
    // at upper layer cell width is space width divided by 2 and so on
    fn get_cell_center(&self) -> Vec<f64> {
        let exponent: u32 = (self.space.nb_layer - self.layer as usize)
            .try_into()
            .unwrap();
        let cell_width = self.space.width / (2u32.pow(exponent) as f64);
        let position: Vec<f64> = (0..self.space.dim)
            .map(|i| 0.5 * cell_width + self.index[i] as f64 * self.space.width)
            .collect();
        //
        position
    } // end of get_cell_center

    // This function parse points and allocate a subcell when a point requires it.
    fn split(&self) -> Option<Vec<Cell<'a, T>>> {
        //
        assert!(self.layer < u16::MAX);
        //
        if self.points_in.is_none() {
            log::debug!("no points in cell");
            self.points_in.as_ref()?; /* return in case of None */
        }
        //
        let mut hashed_cells: HashMap<Vec<u16>, Cell<T>> = HashMap::new();
        //
        for point in self.points_in.as_ref().unwrap() {
            let xyz = point.get_position();
            let mut split_index = Vec::<u16>::with_capacity(self.space.get_dim());
            let cell_center = self.get_cell_center();
            for i in 0..xyz.len() {
                if xyz[i].to_f64().unwrap() > cell_center[i] {
                    // point is in upper part
                    split_index.push(2 * self.index[i] + 1);
                } else {
                    // point is in lower part for this dimension
                    split_index.push(2 * self.index[i]);
                }
            }
            // must  we must allocate a new cell at lower(!) layer
            if let Some(cell) = hashed_cells.get_mut(&split_index) {
                cell.add_point(point);
            } else {
                let mut new_cell: Cell<T> =
                    Cell::new(self.space, self.layer - 1, split_index.clone());
                new_cell.add_point(point);
                hashed_cells.insert(split_index.clone(), new_cell);
            }
        }
        // do not forget to fill new_cells !!
        let mut new_cells: Vec<Cell<T>> = Vec::new();
        for (k, v) in hashed_cells.drain() {
            new_cells.push(v);
        }
        //
        if new_cells.len() > 1 {
            log::trace!(
                "layer : {}, cell with nb points {:?} splitted in {:?} new cells",
                self.layer,
                self.points_in.as_ref().unwrap().len(),
                new_cells.len()
            );
        }
        //
        Some(new_cells)
    }
} // end of impl Cell

//======

/// a layer gathers nodes if a given layer.
pub(crate) struct Layer<'a, T> {
    space: &'a Space,
    // my layer
    layer: u16,
    //
    cell_diameter: f64,
    // hashmap to index in nodes
    hcells: DashMap<Vec<u16>, Cell<'a, T>>,
}

impl<'a, T> Layer<'a, T>
where
    T: Float + Debug + Sync,
{
    //
    fn new(space: &'a Space, layer: u16, cell_diameter: f64) -> Self {
        let hcells: DashMap<Vec<u16>, Cell<T>> = DashMap::new();
        Layer {
            space,
            layer,
            cell_diameter,
            hcells,
        }
    }
    //
    fn insert(&self, cell: Cell<'a, T>) {
        if self.get_cell(&cell.index).is_some() {
            panic!("internal error");
        }
        self.hcells.insert(cell.index.clone(), cell);
    }

    fn insert_cells(&self, cells: Vec<Cell<'a, T>>) {
        //
        assert!(!cells.is_empty());
        for cell in cells {
            self.hcells.insert(cell.index.clone(), cell);
        }
        assert!(!self.hcells.is_empty());
    }

    // returns a dashmap Ref to cell is it exist
    // used only to check coherence. Sub cells are created inside one thread.
    // TODO: do we need dashmap?
    pub(crate) fn get_cell(&self, index: &[u16]) -> Option<one::Ref<Vec<u16>, Cell<'a, T>>> {
        self.hcells.get(index)
    }

    // get an iterator over cells
    fn get_iter_mut(&self) -> iter::IterMut<Vec<u16>, Cell<'a, T>> {
        self.hcells.iter_mut()
    }

    // get an iterator over cells
    fn get_par_iter_mut(&self) -> map::IterMut<Vec<u16>, Cell<'a, T>> {
        self.hcells.par_iter_mut()
    }

    pub(crate) fn get_nb_cells(&self) -> usize {
        self.hcells.len()
    }

    fn get_diameter(&self) -> f64 {
        self.cell_diameter
    }

    pub(crate) fn get_hcells(&self) -> &DashMap<Vec<u16>, Cell<'a, T>> {
        &self.hcells
    }
    // count points in layer
    fn count_points(&self) -> usize {
        let nbpoints: usize = self
            .hcells
            .iter()
            .fold(0, |acc, entryref| acc + entryref.value().get_nb_points());
        nbpoints
    }
} // end implementation Layer

//============================

/// base unit for counting cost.
/// cell is a layer 0 cell, cell_index is index at layer 0, (it mimics a point),
/// benefit is benefit observed along the path from layer 0 to parent cell observed at layer l
#[derive(Debug, Clone)]
pub(crate) struct BenefitUnit {
    // index in layer 0
    cell_index: Vec<u16>,
    // layer corresponding to observe benefit
    layer: u16,
    // benefit encountered up to layer
    benefit: u32,
}

impl BenefitUnit {
    pub(crate) fn new(cell_index: Vec<u16>, layer: u16, benefit: u32) -> Self {
        BenefitUnit {
            cell_index,
            layer,
            benefit,
        }
    }

    // returns cell index and layer
    pub(crate) fn get_id(&self) -> (&Vec<u16>, u16) {
        (&self.cell_index, self.layer)
    }

    //
    pub(crate) fn get_benefit(&self) -> u32 {
        self.benefit
    }
} // end impl BenefitUnit

//TODO: space must provide for random shift!!!

#[cfg_attr(doc, katexit::katexit)]
/// structure describing space in which data point are supposed to live.
/// Data are supposed to live in an enclosing box $$[xmin, xmax]^dim$$
/// Space is seen at different resolution, each resolution is finer with each cell size divided by 2.
/// The coarser mesh will be upper layer.
///
/// Cells are nodes in a tree, each cell , if some data is present in a cell, it is split and propagated at lower layer
/// and points are dispatched in new (smaller) cells.
pub struct SpaceMesh<'a, T: Float> {
    space: &'a Space,
    // leaves are at 0, root node will be above layer_max (at fictitious level nb_layers)
    layer_max: u16,
    //
    global_cell: Option<Cell<'a, T>>,
    // layers are stroed in vector according to their layer num
    layers: Vec<Layer<'a, T>>,
    //
    points: Option<Vec<&'a Point<T>>>,
}

impl<'a, T> SpaceMesh<'a, T>
where
    T: Float + Debug + Sync,
{
    /// define Space.
    /// - data points to cluster
    /// - mindist to discriminate points. under this distance points will be associated
    ///
    pub fn new(space: &'a mut Space, points: Vec<&'a Point<T>>) -> Self {
        assert!(
            space.mindist > 0.,
            "mindist cannot be 0, should related to width so that max layer is not too large"
        );
        //
        let dim: usize = points[0].get_dimension();
        let recommended_nb_layer: u16 =
            (((1 + points.len().ilog2()) / dim as u32) + 1).min(15) as u16;
        log::info!("recommended nb layer : {}", recommended_nb_layer);
        //
        let layer_max_u: usize = ((space.get_dim() as f64).sqrt() * space.get_width()
            / space.mindist)
            .log2()
            .trunc() as usize;
        let nb_layer = layer_max_u + 1;
        log::info!("nb layer  for mindist asked : {}", nb_layer);
        if (nb_layer >= 8) {
            log::warn!("perhaps increase mindist to reduce nb_layer");
        }
        let layer_max_scale: u16 = layer_max_u.try_into().unwrap();
        let layer_max = (layer_max_scale + recommended_nb_layer) / 2;
        let nb_layer = layer_max;
        space.mindist =
            (space.get_dim() as f64).sqrt() * space.get_width() / 2i32.pow(nb_layer as u32) as f64;
        log::info!(
            "setting nb layer to {}, mindist : {:.3e}",
            nb_layer,
            space.mindist
        );
        //
        let mut layers: Vec<Layer<T>> = Vec::with_capacity(layer_max as usize + 1);
        //
        SpaceMesh {
            space,
            layer_max,
            global_cell: None,
            layers,
            points: Some(points),
        }
    }

    /// returns cell diameter at layer l
    pub fn get_cell_diam(&self, l: usize) -> f64 {
        panic!("not yet implemented");
    }

    /// return space dimension
    pub fn get_dim(&self) -> usize {
        self.space.get_dim()
    }
    /// return space width
    pub fn get_width(&self) -> f64 {
        self.space.get_width()
    }

    pub fn get_nb_points(&self) -> usize {
        match &self.points {
            Some(vec) => vec.len(),
            _ => 0,
        }
    }
    /// returns maximum layer , or layer of root node
    pub fn get_root_layer(&self) -> u16 {
        self.layer_max
    }

    pub fn get_layer_max(&self) -> u16 {
        self.layer_max
    }

    pub(crate) fn get_layer(&self, l: u16) -> &Layer<'a, T> {
        &self.layers[l as usize]
    }

    pub fn get_nb_layers(&self) -> usize {
        self.layer_max as usize + 1
    }

    /// returns minimum distance detectable by mesh
    pub fn get_mindist(&self) -> f64 {
        self.space.mindist
    }

    pub fn get_layer_size(&self, layer: usize) -> usize {
        self.layers[layer].get_nb_cells()
    }

    /// return index in mesh of a cell for a point at layer l layer 0 is at finer scale
    pub fn get_cell_index(&self, p: &[T], l: usize) -> Vec<u16> {
        let exp: u32 = (self.get_nb_layers() - l).try_into().unwrap();
        let cell_size = self.get_width() / 2_usize.pow(exp) as f64;
        let mut index = Vec::<u16>::with_capacity(self.get_dim());
        for d in 0..self.get_dim() {
            let idx_f: f64 = ((p[d].to_f64().unwrap() - self.space.get_xmin()) / cell_size).trunc();
            if idx_f < 0. {
                log::error!(
                    "got negative index , coordinate value : {:.3e}",
                    p[d].to_f64().unwrap()
                );
                panic!("negative coordinate for cell");
            }
            assert!(idx_f <= 65535.0);
            index.push(idx_f.trunc() as u16);
        }
        index
    } // end get_cell_index

    // return reference to dashmap entry
    fn get_cell_with_position(&self, p: &[T], l: usize) -> Option<one::Ref<Vec<u16>, Cell<'a, T>>> {
        let idx = self.get_cell_index(p, l);
        self.layers[l].get_cell(&idx)
    }

    /// for layer 0, the layer with the maximum number of cells,
    /// the diameter of a cell is $$ width * \sqrt(d)/2^(nb_layer + 1 - layer)  $$
    ///  - Delta max value of  extension by dimension
    ///  - d : dimension of space
    ///  - l : layer num in 0..nb_layer
    pub fn get_layer_cell_diameter(&self, layer: u16) -> f64 {
        assert!(layer <= self.layer_max);
        //
        let cell_diameter: f64 = (self.space.get_width() * (self.space.get_dim() as f64).sqrt())
            / 2_u32.pow((self.layer_max + 1 - layer) as u32) as f64;
        cell_diameter
    }

    /// this function embeds data in a 2-rhst
    /// It propagate cells partitioning space from coarser to finer mesh
    pub fn embed(&mut self) {
        log::info!("SpaceMesh::embed");
        let cpu_start = ProcessTime::now();
        let sys_now = SystemTime::now();
        // initialize layers (layer lmax to bottom layer 0
        self.layers = Vec::with_capacity(self.get_nb_layers());
        for l in 0..self.get_nb_layers() {
            let layer =
                Layer::<T>::new(self.space, l as u16, self.get_layer_cell_diameter(l as u16));

            self.layers.push(layer);
        }
        log::info!("SpaceMesh::embed allocated nb layers {}", self.layers.len());
        // initialize root cell
        let center = vec![0u16; self.get_dim()];
        // root cell, it is declared above maximum layer as it is isolated...
        let mut global_cell = Cell::<T>::new(self.space, self.get_layer_max() + 1, center);
        global_cell.init_points(&self.points.as_ref().unwrap().clone());
        self.global_cell = Some(global_cell);
        //
        //
        self.points
            .as_ref()
            .unwrap()
            .iter()
            .map(|p| self.global_cell.as_mut().unwrap().add_point(p));
        // now to first layer (one cell) others layers can be made
        let cells_first_res = self.global_cell.as_ref().unwrap().split();
        if cells_first_res.is_none() {
            log::error!("splitting of  global cell failed");
            panic!();
        }
        let cells_first = cells_first_res.unwrap();
        log::info!("global cell split in nb cells {}", cells_first.len());
        // initialize first layer (layer max)
        let mut upper_layer = &self.layers[self.get_layer_max() as usize];
        upper_layer.insert_cells(cells_first);
        assert!(upper_layer.get_nb_cells() > 0);
        // now we can propagate layer downward, cells can be treated // using par_iter_mut
        for l in (1..self.get_nb_layers()).rev() {
            log::info!("splitting layer : l : {}", l);
            let lower_layer = &self.layers[l - 1];
            // TODO: changing into_iter to into_par_iter is sufficient to get //.
            self.layers[l]
                .get_hcells()
                .into_par_iter()
                .for_each(|cell| {
                    assert_eq!(l, cell.get_layer());
                    if let Some(new_cells) = cell.split() {
                        lower_layer.insert_cells(new_cells);
                    }
                });
            log::info!(
                "layer {} has {} nbcells",
                l - 1,
                self.layers[l - 1].get_nb_cells()
            );
            let nbpt_by_cell: f64 =
                self.get_nb_points() as f64 / self.layers[l].get_nb_cells() as f64;
            if nbpt_by_cell <= 2. {
                log::warn!(
                    "too many layers asked, nb points by cell : {:.3e}",
                    nbpt_by_cell
                );
            }
        }
        //
        self.compute_subtree_size();
        log::info!("exiting SpaceMesh::embed");
        let cpu_time: Duration = cpu_start.elapsed();
        println!(
            " SpaceMesh::embed sys time(s) {:?} cpu time {:?}",
            sys_now.elapsed().unwrap().as_secs(),
            cpu_time.as_secs()
        );
        //
    } // end of embed

    /// gives a summary: number of cells by layer etc
    pub fn summary(&mut self) {
        //
        if self.layers.is_empty() {
            println!("layers not allocated");
            std::process::exit(1);
        }
        println!(" number of layers : {}", self.get_nb_layers());
        for l in (0..self.get_nb_layers()).rev() {
            println!(
                "layer : {}, nb cells : {}, cell diameter : {:.3e}, nbpoints : {}",
                l,
                self.layers[l].get_nb_cells(),
                self.layers[l].get_diameter(),
                self.layers[l].count_points()
            );
        }
    }

    // compute benefits by points (in fact cells at layer 0) and layer (See algo 2 of paper and lemma 3.4)
    pub(crate) fn compute_benefits(&self) -> Vec<BenefitUnit> {
        //
        log::info!("in compute_benefits");
        let cpu_start = ProcessTime::now();
        let sys_now = SystemTime::now();
        //
        let nb_layers = self.get_nb_layers();
        let layer_0 = self.get_layer(0);
        let nb_cells = layer_0.get_nb_cells();
        // allocate benefit array in one array as we will need a sort! with par_sort_unstable from rayon
        // the nb_layers of a given cell are stored contiguously in the order obtained from the iterator
        let benefits = Vec::<usize>::with_capacity(nb_cells * nb_layers);
        // loop on cells of layer_0, keeping track of the order
        let cell_order = Vec::<Vec<u16>>::with_capacity(nb_cells);
        // for each cell we store potential merge benefit at each level!
        let mut benefits: Vec<BenefitUnit> = Vec::with_capacity(nb_cells * nb_layers);
        // iterate over layer 0 and upper_layers store cost and then sort benefit array in decreasing order!
        let layer0 = self.get_layer(0);
        for cell in layer0.get_hcells().iter() {
            let mut benefit_at_layer = Vec::<u32>::with_capacity(nb_layers);
            let mut previous_tree_size: u32 = cell.get_subtree_size();
            for l in 1..self.get_nb_layers() {
                let upper_index = cell.get_upper_cell_index_at_layer(l as u16);
                let upper_cell = self.get_layer(l as u16).get_cell(&upper_index).unwrap();
                upper_cell.get_subtree_size();
                // we can unroll benefit computation ( and lyer 0 give no benefit)
                let benefit =
                    2u32.pow(l as u32) * previous_tree_size + benefit_at_layer.last().unwrap_or(&0);
                benefit_at_layer.push(benefit);
                benefits.push(BenefitUnit::new(cell.key().clone(), l as u16, benefit));
            }
        }
        // sort benefits in decreasing (!) order
        log::info!("sorting benefits");
        benefits.par_sort_unstable_by(|unita, unitb| {
            unitb.benefit.partial_cmp(&unita.benefit).unwrap()
        });
        //
        log::info!("benefits vector size : {}", benefits.len());
        assert!(benefits[0].benefit >= benefits[1].benefit);
        //
        log::info!("exiting compute_benefits");
        let cpu_time: Duration = cpu_start.elapsed();
        println!(
            " compute_benefits sys time(s) {:?} cpu time {:?}",
            sys_now.elapsed().unwrap().as_secs(),
            cpu_time.as_secs()
        );
        //
        benefits
    } // end of compute_benefits

    // we need to compute cardinal of each subtree (all descendants of each cell)
    // just dispatch number of points
    fn compute_subtree_size(&self) {
        //
        log::info!("in compute_subtree_size ");
        for l in 0..self.get_nb_layers() {
            let layer = &self.layers[l];
            layer.get_hcells().par_iter_mut().for_each(|mut cell| {
                cell.subtree_size = cell.points_in.as_ref().unwrap().len() as u32;
            });
            self.get_subtree_size(l as u16);
        }
        log::info!("exiting compute_subtree_size ");
    } // end of compute_subtree_cardinals

    // sum of sub tree size as seen from layer l. Used for debugging purpose
    fn get_subtree_size(&self, l: u16) -> u32 {
        let size: u32 = self.layers[l as usize]
            .get_hcells()
            .iter()
            .map(|mesh_cell| mesh_cell.get_subtree_size())
            .sum();
        log::info!("total sub tree size at layer l : {} = {}", l, size);
        //
        size
    }
} // end of impl SpaceMesh

//===========================

// TODO: should add shift margin
// space must enclose points
fn check_space<T: Float + Debug>(space: &Space, points: &[Point<T>]) {
    assert!(!points.is_empty());
    //
    let mut max_xi = T::min_value();
    let mut min_xi: T = T::max_value();
    for (ipt, pt) in points.iter().enumerate() {
        for (i, xi) in pt.get_position().iter().enumerate() {
            if (space.get_xmin() >= <f64 as num_traits::NumCast>::from(*xi).unwrap()) {
                log::error!(
                    "space.get_xmin() too high,  point of rank {} has xi = {:?} ",
                    ipt,
                    xi
                );
                panic!();
            }
            max_xi = max_xi.max(*xi);
            min_xi = min_xi.min(*xi);
        }
    }
    let delta = max_xi - min_xi;
    //
    log::error!(
        "minimum space, xmin : {:.3e}, width : {:.3e}",
        <f64 as num_traits::NumCast>::from(min_xi).unwrap(),
        <f64 as num_traits::NumCast>::from(delta).unwrap()
    );
} // end of check_space

//========================================================

#[cfg(test)]
mod tests {

    use super::*;

    use rand::distributions::Uniform;
    use rand::prelude::*;
    use rand_xoshiro::Xoshiro256PlusPlus;

    use rand_distr::{Distribution, Exp};

    fn log_init_test() {
        let _ = env_logger::builder().is_test(true).try_init();
    }

    #[test]
    fn test_uniform_random() {
        log_init_test();
        log::info!("in test_uniform_random");
        //
        let nbvec = 40_000_000usize;
        let dim = 10;
        let width: f64 = 1000.;
        let mindist = 5.;
        let unif_01 = Uniform::<f64>::new(0., width);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(234567_u64);
        let mut points: Vec<Point<f64>> = Vec::with_capacity(nbvec);
        for i in 0..nbvec {
            let p: Vec<f64> = (0..dim).map(|_| unif_01.sample(&mut rng)).collect();
            points.push(Point::<f64>::new(i, p, (i % 5).try_into().unwrap()));
        }
        let refpoints: Vec<&Point<f64>> = points.iter().map(|p| p).collect();
        // Space definition
        let mut space = Space::new(dim, 0., width, mindist);

        let mut mesh = SpaceMesh::new(&mut space, refpoints);
        mesh.embed();
        mesh.summary();
        //
        mesh.compute_subtree_size();
        //
        let benefits = mesh.compute_benefits();
    } //end of test_uniform_random

    #[test]
    fn chech_dashmap() {
        let size = 10_000_000;
        let data = vec![1u32; size];
        let hmap = DashMap::<usize, u32>::new();
        (0..size).into_par_iter().for_each(|i| {
            let res = hmap.insert(i, data[i]);
        });

        //
        let total: usize = hmap.into_par_iter().map(|cell| cell.1 as usize).sum();
        println!(
            "total : {}, should be  : {}",
            total,
            data.into_iter().sum::<u32>()
        );
    }

    #[test]
    fn check_mindist() {
        log_init_test();
        log::info!("in check_mindist");
        //
        let nbvec = 10_000_000_usize;
        let dim = 15;
        let width: f64 = 1000.;
        let mindist = 0.001;
        // Space definition
        let space = Space::new(dim, 0., width, mindist);
    } // end of check_mindist

    #[test]
    fn test_exp_random() {
        log_init_test();
        log::info!("in test_exp_random");
        //
        let nbvec = 40_000_000_usize;
        let dim = 15;
        let width: f64 = 1000.;
        let mindist = 5.;
        // sample with coordinates following exponential law
        let law = Exp::<f32>::new(10. / width as f32).unwrap();
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(234567_u64);
        let mut points: Vec<Point<f32>> = Vec::with_capacity(nbvec);
        for i in 0..nbvec {
            let p: Vec<f32> = (0..dim)
                .map(|_| law.sample(&mut rng).min(width as f32))
                .collect();
            points.push(Point::<f32>::new(i, p, (i % 5).try_into().unwrap()));
        }
        let refpoints: Vec<&Point<f32>> = points.iter().map(|p| p).collect();
        // Space definition
        let mut space = Space::new(dim, 0., width, mindist);

        let mut mesh = SpaceMesh::new(&mut space, refpoints);
        mesh.embed();
        mesh.summary();
        // check number of points for cell at origin
        let p = vec![0.001; dim];
        log::info!("cells info for p : {:?}", p);
        for l in (0..mesh.get_layer_max() as usize).rev() {
            let refcell = mesh.get_cell_with_position(&p, l);
            if let Some(cell) = mesh.get_cell_with_position(&p, l) {
                let nbpoints_in = refcell.unwrap().value().get_nb_points();
                log::info!(
                    "nb point in cell corresponding to point at layer {} : {}",
                    l,
                    nbpoints_in
                );
            } else {
                log::info!(" no data points in cell corresponding p at layer {}", l);
            }
        }
        //
    } //end of test_exp_random
} // end of mod test
