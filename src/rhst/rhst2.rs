#![allow(clippy::needless_range_loop)]

//! This is a small rhst implementation as described in Cohen-Addad paper
//
// NOTE: our implementation grows splitting tree with ascending layer num.
//       So we can stop when we have one point by cell naturally while layer 0 always remins the coarser
//       layer with only one cell which covers the whole mesh.
//
// We prefer rhst over cover trees as described in
//  TerraPattern: A Nearest Neighbor Search Service (2019)
//  Zaheer, Guruganesh, Levin Smola
//  as we can group points very close (under lower distance threshold) in one cell
//
// TODO: We must be able to have more than 16 layers, to be able to have cell indexed
// go beyond 2^16 so we need to encode cell indexes in a u32!

use cpu_time::ProcessTime;
use std::time::{Duration, SystemTime};

use num_traits::float::Float;

use std::fmt::Debug;

use anyhow::anyhow;
use dashmap::{iter, mapref::one, rayon::*, DashMap};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::atomic::AtomicUsize;

use super::point::*;

// given data min max , use margin to define enclosing space
pub(crate) fn compute_enclosing_bounds(xmin: f64, xmax: f64, margin: f64) -> (f64, f64) {
    let origin: f64;
    assert!(margin > 0. && margin < 1.);
    if xmin > 0. {
        origin = xmin * (1. - margin);
    } else if xmin < 0. {
        origin = xmin * (1. + margin);
    } else {
        origin = -margin; // TODO: scale with xmax ?
    }
    //
    let upper: f64;
    if xmax > 0. {
        upper = xmax * (1. + margin);
    } else if xmax < 0. {
        upper = xmax * (1. - margin);
    } else {
        upper = margin;
    }
    (origin, upper)
} // end of compute_enclosing_bounds

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
///
#[derive(Debug, Copy, Clone)]
pub struct Space {
    // space dimension
    dim: usize,
    // max over dimension of (xmax-xmin). but not mesh width!
    width: f64,
    // smallest coordinate
    xmin: f64,
    // to strictly enclose all data we go from xmin * (1. - margin) to xmax * (1. +  margin)
    margin: f64,
    // minimum space coordinate
    origin: f64,
    // maximum space coordinate
    upper: f64,
    // minimum distance we want to discriminate
    mindist: Option<f64>,
    //
    nb_layer: usize,
}

impl Space {
    /// define Space.
    /// -
    ///
    pub fn new(dim: usize, xmin: f64, xmax: f64, mindist: Option<f64>) -> Self {
        // to ensure all data points are internal to the mesh
        let margin = 1.0E-2;
        //
        let width = xmax - xmin;
        if let Some(distance) = mindist {
            assert!(
                distance > 0.,
                "mindist cannot be 0, should related to width so that max layer is not too large"
            );
        }
        let (origin, upper) = compute_enclosing_bounds(xmin, xmax, margin);
        log::info!("mesh min : {:.3e}, mesh max : {:.3e}", origin, upper);
        //
        Space {
            dim,
            width,
            xmin,
            margin,
            origin,
            upper,
            mindist,
            nb_layer: 0,
        }
    }

    pub(crate) fn get_margin(&self) -> f64 {
        self.margin
    }
    /// return space dimension
    pub fn get_dim(&self) -> usize {
        self.dim
    }

    /// return space width (larger than data width by 1 + margin)
    pub fn get_mesh_width(&self) -> f64 {
        self.upper - self.origin
    }

    pub fn get_data_width(&self) -> f64 {
        self.width
    }

    /// fn get data xmin
    pub fn get_xmin(&self) -> f64 {
        self.xmin
    }

    pub fn get_origin(&self) -> f64 {
        self.origin
    }

    pub fn dump(&self) {
        log::info!("dim : {}", self.dim);
        log::info!(
            "xmin : {:.2e}, xmax : {:.2e}, margin : {:.2e}",
            self.xmin,
            self.xmin + self.width,
            self.get_margin()
        );
        log::info!("nb layer : {}", self.nb_layer);
        //
        if let Some(epsil) = self.mindist {
            log::info!("mindist : {:.3e}", epsil)
        }
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
    /// at layer l , index[i] is in [0,2^l[ for each i
    index: Vec<u16>,
    // number of points in subtree of cells
    subtree_size: usize,
    //
    points_in: Option<Vec<&'a Point<T>>>,
}

impl<'a, T> Cell<'a, T>
where
    T: Float + Debug + std::fmt::LowerExp,
{
    pub fn new(space: &'a Space, layer: u16, index: Vec<u16>) -> Self {
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
    #[allow(unused)]
    pub(crate) fn get_cell_weight(&self) -> f32 {
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

    pub(crate) fn get_points(&self) -> Option<&Vec<&Point<T>>> {
        self.points_in.as_ref()
    }

    fn get_cell_index(&self) -> &[u16] {
        &self.index
    }

    #[allow(unused)]
    // returns true if a point is in sub tree of cell
    pub(crate) fn has_point(&self, pid: PointId) -> bool {
        if self.points_in.is_none() {
            false
        } else {
            for p in self.points_in.as_ref().unwrap() {
                if p.get_id() == pid {
                    return true;
                }
            }
            false
        }
    }
    // get parent cell in splitting process
    #[allow(unused)]
    pub(crate) fn get_parent_cell_index(&self) -> Vec<u16> {
        assert!(self.layer > 0);
        self.index.iter().map(|x| x / 2).collect::<Vec<u16>>()
    }

    // get parent cell at layer l in splitting process, largest cell is down
    pub(crate) fn get_larger_cell_index_at_layer(&self, l: u16) -> Vec<u16> {
        assert!(self.layer > l && self.layer - l <= 15);
        // compute new index by dividing by 2_u16.pow((self.layer -l) as u32);
        self.index
            .iter()
            .map(|x| x >> (self.layer - l))
            .collect::<Vec<u16>>()
    }

    // subtree size are computed by reverse of splitting in SpaceMesh::compute_subtree_size
    pub(crate) fn get_subtree_size(&self) -> usize {
        self.subtree_size
    }

    // adds a point in cell
    fn add_point(&mut self, point: &'a Point<T>) {
        match self.points_in.as_mut() {
            Some(points) => points.push(point),
            None => {
                let vec = vec![point];
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
    // at coarser (pareint in splitting) layer cell width is space width divided by 2 and so on
    fn get_cell_center(&self) -> Vec<f64> {
        let exponent: u32 = self.layer as u32;
        let cell_width = self.space.get_mesh_width() / (2u32.pow(exponent) as f64);
        let origin = self.space.get_origin();
        let position: Vec<f64> = (0..self.space.dim)
            .map(|i| origin + 0.5 * cell_width + self.index[i] as f64 * cell_width)
            .collect();
        //
        position
    } // end of get_cell_center

    // This function parse points and allocate a subcell when a point requires it.
    // recall the tree grow upward. finer mesh is last layer, contrary to the paper
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
                    Cell::new(self.space, self.layer + 1, split_index.clone());
                new_cell.add_point(point);
                hashed_cells.insert(split_index.clone(), new_cell);
            }
        }
        // do not forget to fill new_cells !!
        let mut new_cells: Vec<Cell<T>> = Vec::new();
        for (_, v) in hashed_cells.drain() {
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
    // my layer
    layer: u16,
    // cell diameter at this layer
    cell_diameter: f64,
    // hashmap to index in nodes
    hcells: DashMap<Vec<u16>, Cell<'a, T>>,
}

impl<'a, T> Layer<'a, T>
where
    T: Float + Debug + std::fmt::LowerExp + Sync,
{
    //
    fn new(layer: u16, cell_diameter: f64) -> Self {
        let hcells: DashMap<Vec<u16>, Cell<T>> = DashMap::new();
        Layer {
            layer,
            cell_diameter,
            hcells,
        }
    }
    //
    #[allow(unused)]
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
            assert_eq!(cell.layer, self.get_layer());
            self.hcells.insert(cell.index.clone(), cell);
        }
        assert!(!self.hcells.is_empty());
    }

    #[allow(unused)]
    pub(crate) fn get_layer(&self) -> u16 {
        self.layer
    }
    // returns a dashmap Ref to cell is it exist
    // used only to check coherence. Sub cells are created inside one thread.
    // TODO: do we need dashmap?
    pub(crate) fn get_cell(&self, index: &[u16]) -> Option<one::Ref<Vec<u16>, Cell<'a, T>>> {
        self.hcells.get(index)
    }

    // get an iterator over cells
    #[allow(unused)]
    pub(crate) fn get_iter(&self) -> iter::Iter<Vec<u16>, Cell<'a, T>> {
        self.hcells.iter()
    }
    // get an iterator over cells
    #[allow(unused)]
    fn get_iter_mut(&self) -> iter::IterMut<Vec<u16>, Cell<'a, T>> {
        self.hcells.iter_mut()
    }

    // get an iterator over cells
    #[allow(unused)]
    fn get_par_iter_mut(&self) -> map::IterMut<Vec<u16>, Cell<'a, T>> {
        self.hcells.par_iter_mut()
    }

    pub(crate) fn get_nb_cells(&self) -> usize {
        self.hcells.len()
    }

    pub(crate) fn get_diameter(&self) -> f64 {
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

    // dump cells in layer. If level >= 1 dump points of each cell
    // debugging purposes
    pub(crate) fn dump(&self, level: usize) {
        log::info!(
            "\n dumping of layer {}, nbcells : {}",
            self.layer,
            self.get_nb_cells()
        );
        for (rank, item) in self.hcells.iter().enumerate() {
            log::info!("idx : {:?}", item.key());
            let cell = item.value();
            log::info!("cell rank : {}, nb points : {}", rank, cell.get_nb_points());
            if level >= 1 {
                let points = cell.get_points().unwrap();
                for p in points.iter() {
                    log::trace!("p : {:?}", p.get_position());
                }
            }
        }
    }
} // end implementation Layer

//============================

/// base unit for counting cost.
/// cell is a layer 0 cell, cell_index is index at layer 0, (at finest mesh , it mimics a point),
/// benefit is benefit observed along the path from layer 0 to parent cell observed at layer l
#[derive(Debug, Clone)]
pub(crate) struct BenefitUnit {
    // index in layer 0, beccause we construct benefit units from 0 to upper layers
    cell_index: Vec<u16>,
    // layer corresponding to observe benefit
    layer: u16,
    // benefit encountered up to layer
    benefit: usize,
}

impl BenefitUnit {
    pub(crate) fn new(cell_index: Vec<u16>, layer: u16, benefit: usize) -> Self {
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

    // returns cell index at layer 0.
    #[allow(unused)]
    pub(crate) fn get_cell_idx(&self) -> &Vec<u16> {
        &self.cell_index
    }

    //
    pub(crate) fn get_benefit(&self) -> usize {
        self.benefit
    }

    // recall index in benefit unit are obtained from layer 0 cells
    pub(crate) fn get_index_at_layer(&self, nb_layer: u16) -> Vec<u16> {
        let shift = nb_layer - 1 - self.layer;
        // compute new index by dividing by 2_u16.pow((shift) as u32);
        self.cell_index
            .iter()
            .map(|x| x >> shift)
            .collect::<Vec<u16>>()
    }
} // end impl BenefitUnit

//TODO: space must provide for random shift!!!

#[cfg_attr(doc, katexit::katexit)]
/// structure describing space in which data point are supposed to live.
/// Data are supposed to live in an enclosing box $$[xmin, xmax]^dim$$.  
/// Space is seen at different resolution, each resolution is finer with each cell size divided by 2, therefore
/// the number of cells grow by a factor $2^d$ at each layer.
///
/// Cells are characterized by their index in the mesh at each layer. Knowing the index of a cell at layer 0
/// the cell at layer l enclosing it is obtained by dividing the index at layer 0 by $2^l$
/// The coarser mesh will be upper layer.
///
/// Cells are nodes in a tree, each cell , if some data is present in it, is split and propagated at lower layer
/// and points are dispatched in new (smaller) cells.
pub struct SpaceMesh<'a, T: Float> {
    space: &'a Space,
    // layers are stored in vector according to their layer num
    layers: Vec<Layer<'a, T>>,
    //
    points: Option<Vec<&'a Point<T>>>,
}

impl<'a, T> SpaceMesh<'a, T>
where
    T: Float + Debug + std::fmt::LowerExp + Sync,
{
    /// define Space.
    /// - data points to cluster
    /// - mindist to discriminate points. under this distance points will be associated
    ///
    pub fn new(space: &'a mut Space, points: Vec<&'a Point<T>>) -> Self {
        //
        let dim: usize = points[0].get_dimension();
        let recommended_nb_layer: usize;
        if let Some(epsil) = space.mindist {
            let aux = ((space.get_dim() as f64).sqrt() * space.get_mesh_width() / epsil)
                .log2()
                .ceil();
            // to get one point by cell in average
            recommended_nb_layer = 15.min(aux as usize);
            log::info!(
                "depending on mindist  {:.3e} got nb_layer {}",
                epsil,
                recommended_nb_layer
            );
            if recommended_nb_layer >= 8 {
                log::warn!("perhaps increase mindist to reduce nb_layer");
            }
        } else {
            // to get one point by cell in average
            let dimprodlayer = points.len().ilog2() as usize;
            recommended_nb_layer = 15.min(dimprodlayer / dim).max(5);
            let epsil = (space.get_dim() as f64).sqrt() * space.get_mesh_width()
                / 2i32.pow(recommended_nb_layer as u32 - 1) as f64;
            space.mindist = Some(epsil);
            log::info!(
                "no mindist got nb_layer {} mindist : {:.3e}",
                recommended_nb_layer,
                epsil
            );
        };

        //
        log::info!(
            "\n ******************************************* \n setting recommanded nb layer to {}, min cell diamter : {:.3e}\n",
            recommended_nb_layer,
            space.mindist.unwrap()
        );
        //
        let layers: Vec<Layer<T>> = Vec::with_capacity(space.nb_layer);
        //
        SpaceMesh {
            space,
            layers,
            points: Some(points),
        }
    }

    /// returns cell diameter at layer l
    pub fn get_cell_diam(&self, _l: usize) -> f64 {
        panic!("not yet implemented");
    }

    /// return space dimension
    pub fn get_dim(&self) -> usize {
        self.space.get_dim()
    }
    /// return space width
    pub fn get_width(&self) -> f64 {
        self.space.get_mesh_width()
    }

    pub fn get_cell_width(&self, l: u16) -> f64 {
        self.space.get_mesh_width() / 2_u32.pow(l as u32) as f64
    }

    pub fn get_nb_points(&self) -> usize {
        match &self.points {
            Some(vec) => vec.len(),
            _ => 0,
        }
    }

    pub(crate) fn get_layer(&self, l: u16) -> &Layer<'a, T> {
        &self.layers[l as usize]
    }

    pub fn get_nb_layers(&self) -> usize {
        self.layers.len()
    }

    /// returns minimum distance detectable by mesh
    pub fn get_mindist(&self) -> f64 {
        self.space.mindist.unwrap()
    }

    /// return number of cells in layer
    pub fn get_layer_size(&self, layer: usize) -> usize {
        self.layers[layer].get_nb_cells()
    }

    /// return index in mesh of a cell for a point, at layer l (layer 0 is at coarsest scale)
    pub fn get_cell_index(&self, p: &[T], l: usize) -> Vec<u16> {
        let exp: u32 = l.try_into().unwrap();
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
            index.push(idx_f as u16);
        }
        index
    } // end get_cell_index

    /// given index of cell at finer mesh , compute index at layer l
    pub fn to_coarser_index(&self, finest_index: &[u16], l: u16) -> Vec<u16> {
        let layer_max: u16 = (self.get_nb_layers() - 1).try_into().unwrap();
        let shift: u16 = layer_max - l;
        finest_index.iter().map(|x| x >> shift).collect()
    }

    /// get cell center knowing its finer index and layer.
    /// Returns an error if no cell corresponds to this index.
    pub fn get_cell_center(&self, idx: &[u16], l: u16) -> anyhow::Result<Vec<f64>> {
        assert!((l as usize) < self.get_nb_layers());
        let index_l = self.to_coarser_index(idx, l);
        let res = self.layers[l as usize].get_cell(&index_l);
        if res.is_none() {
            return Err(anyhow!("no cell at index"));
        }
        Ok(res.unwrap().get_cell_center())
    }

    // return reference to dashmap entry
    #[allow(unused)]
    pub(crate) fn get_cell_with_position(
        &self,
        p: &[T],
        l: usize,
    ) -> Option<one::Ref<Vec<u16>, Cell<'a, T>>> {
        let idx = self.get_cell_index(p, l);
        self.layers[l].get_cell(&idx)
    }

    // if a potential cell at given index and layer do not contain any point, i.e cell is not allocated, we return None
    pub(crate) fn get_cell(
        &self,
        idx: &[u16],
        layer: u16,
    ) -> Option<one::Ref<Vec<u16>, Cell<'a, T>>> {
        self.layers[layer as usize].get_cell(idx)
    }

    /// for layer at finest scale, the layer with the maximum number of cells,
    /// the diameter of a cell is : $$ width * \sqrt(d)/2^(nb_layer + 1 - layer)  $$
    ///  - Delta max value of  extension by dimension
    ///  - d : dimension of space
    ///  - l : layer num in 0..nb_layer
    pub fn get_layer_cell_diameter(&self, layer: u16) -> f64 {
        //
        let cell_diameter: f64 = (self.space.get_mesh_width()
            * (self.space.get_dim() as f64).sqrt())
            / 2_u32.pow((layer) as u32) as f64;
        cell_diameter
    }

    /// this function embeds data in a 2-rhst
    /// It propagate cells partitioning space from coarser to finer mesh
    pub fn embed(&mut self) {
        log::info!("SpaceMesh::embed");
        let cpu_start = ProcessTime::now();
        let sys_now = SystemTime::now();
        // initialize layers (layer lmax to bottom layer 0
        self.layers = Vec::with_capacity(15);
        log::info!("SpaceMesh::embed allocated by default nb layers {}", 15);
        // initialize root cell
        let center = vec![0u16; self.get_dim()];
        // root cell, it is declared above maximum layer as it is isolated...
        let mut global_cell = Cell::<T>::new(self.space, 0u16, center.clone());
        global_cell.init_points(&self.points.as_ref().unwrap().clone());
        let mut l = 0u16;
        let coarser_layer = Layer::<T>::new(l as u16, self.get_layer_cell_diameter(l as u16));
        self.layers.push(coarser_layer);

        let coarser_layer: &Layer<'a, T> = &self.layers[0 as usize];
        coarser_layer.insert_cells(vec![global_cell]);
        // now we can propagate layer downward, cells can be treated // using par_iter_mut
        loop {
            log::info!("splitting layer : l : {}", l);
            let new_layer = l + 1;
            let finer_layer = Layer::<T>::new(
                new_layer as u16,
                self.get_layer_cell_diameter(new_layer as u16),
            );
            self.layers.push(finer_layer);
            let finer_layer = &self.layers[new_layer as usize];
            // changing into_iter to into_par_iter is sufficient to get //.
            self.layers[l as usize]
                .get_hcells()
                .into_par_iter()
                .for_each(|cell| {
                    assert_eq!(l as usize, cell.get_layer());
                    if let Some(new_cells) = cell.split() {
                        finer_layer.insert_cells(new_cells);
                    }
                });
            log::info!(
                "layer {} has {} nbcells",
                new_layer,
                self.layers[new_layer as usize].get_nb_cells()
            );
            if self.layers[new_layer as usize].get_nb_cells() >= self.get_nb_points() {
                break;
            } else {
                if new_layer == 15 {
                    log::info!("stopping at max layer : {}", new_layer);
                    break;
                }
                l = l + 1;
                log::info!("Spacemesh::embed , adding layer {} ", l);
            }
        }
        // debug info : dump coordianates of layer 0 origin cell
        log::info!("got nb layers : {}", self.get_nb_layers());
        for l in 0..self.get_nb_layers() {
            let layer = self.get_layer(l as u16);
            let cell0 = layer.get_hcells().iter().nth(0).unwrap();
            let cell0_idx = cell0.get_cell_index();
            let cell0_center = cell0.get_cell_center();
            assert!(!cell0_center.is_empty());
            log::info!(
                "\n layer : {}, cell diameter {:.3e}",
                l,
                layer.get_diameter()
            );
            log::info!("index of first (in iterator) cell : {:?}", cell0_idx);
            if cell0_center.len() <= 20 || log::log_enabled!(log::Level::Debug) {
                log::info!("coordinates of first cell center : ");
                for c in cell0_center {
                    print!(" {:.3e}", c);
                }
                println!();
            }
        }
        //
        self.compute_subtree_size();
        log::info!("exiting SpaceMesh::embed");
        let cpu_time: Duration = cpu_start.elapsed();
        println!(
            " SpaceMesh::embed sys time(ms) {:?} cpu time {:?}",
            sys_now.elapsed().unwrap().as_millis(),
            cpu_time.as_millis()
        );
        log::debug!("exiting SpaceMesh::embed")
        //
    } // end of embed

    #[allow(unused)]
    pub(crate) fn dump_layer(&self, layer: usize, level: usize) {
        self.layers[layer].dump(level);
    }

    //
    /// gives a summary: number of cells by layer etc
    pub fn summary(&mut self) {
        //
        if self.layers.is_empty() {
            println!("layers not allocated");
            std::process::exit(1);
        }
        println!("\n number of layers : {}", self.get_nb_layers());
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

    //

    // compute benefits by points (in fact cells at layer 0) and layer (See algo 2 of paper and lemma 3.4)
    pub(crate) fn compute_benefits(&self, version: usize) -> Vec<BenefitUnit> {
        match version {
            1 => self.compute_benefits_1(),
            2 => self.compute_benefits_2(),
            _ => {
                panic!("version must be 1  or 2, got {}", version);
            }
        }
    }

    //

    pub(crate) fn compute_benefits_1(&self) -> Vec<BenefitUnit> {
        //
        log::info!("in compute_benefits_1");
        let cpu_start = ProcessTime::now();
        let sys_now = SystemTime::now();
        //
        let nb_layers = self.get_nb_layers();
        let max_layer = (nb_layers - 1).try_into().unwrap();
        // for each cell we store potential merge benefit at each level!
        //
        // iterate over finest layer  and then coarser layers to store cost and then sort benefit array in decreasing order!
        // each cell at coarser layer stores the maximum BenefitUnit observed at its layer
        let mut best_finer_cell_contribution = HashMap::<(Vec<u16>, u16), BenefitUnit>::new();
        let finest_layer = self.get_layer(max_layer);
        for cell in finest_layer.get_hcells().iter() {
            let mut benefit_at_layer = Vec::<usize>::with_capacity(nb_layers);
            let mut previous_tree_size: usize = cell.get_subtree_size();
            for l in (0..max_layer).rev() {
                let coarser_index = cell.get_larger_cell_index_at_layer(l as u16);
                let coarser_cell = self.get_layer(l as u16).get_cell(&coarser_index).unwrap();
                // we can unroll benefit computation ( and finest layer  give no benefit)
                let delta_l = max_layer - l;
                let benefit = 2usize.pow(delta_l as u32) * previous_tree_size
                    + benefit_at_layer.last().unwrap_or(&0);
                let key = (coarser_index, l as u16);
                if let Some(old_best) = best_finer_cell_contribution.get_mut(&key) {
                    if benefit > old_best.get_benefit() {
                        // we found a cell at finest layer  that has better benefit at this level of tree
                        *(old_best) = BenefitUnit::new(cell.key().clone(), l as u16, benefit);
                    }
                } else {
                    best_finer_cell_contribution
                        .insert(key, BenefitUnit::new(cell.key().clone(), l as u16, benefit));
                }
                // loop update
                previous_tree_size = coarser_cell.get_subtree_size();
                benefit_at_layer.push(benefit);
            }
        }
        // now we collect for each cell at finest layer,  the coarser level layer ( > 0 in the paper, but < max_layer in our implementation)
        // at which it is the best contribution
        // TODO: the loop should be made //
        let mut higher_best_finer_layer_contribution = HashMap::<Vec<u16>, BenefitUnit>::new();
        for cell in finest_layer.get_hcells().iter() {
            let mut best_unit: Option<&BenefitUnit> = None;
            for l in (0..max_layer).rev() {
                let coarser_index = cell.get_larger_cell_index_at_layer(l as u16);
                // is this cell the best at upper_index ?
                let key = (coarser_index, l as u16);
                if let Some(unit) = best_finer_cell_contribution.get(&key) {
                    let (idx, level) = unit.get_id();
                    assert_eq!(level, l as u16);
                    if idx == cell.get_cell_index() {
                        best_unit = Some(unit);
                    }
                }
            }
            if let Some(b_unit) = best_unit {
                higher_best_finer_layer_contribution
                    .insert(cell.get_cell_index().to_vec(), b_unit.clone());
            }
        }
        // We have now among the list of finest_layer, cells that have maximum benefits, the coarser layers of their contribution
        // sort benefits in decreasing (!) order
        // We can transfert to a Vec<BenefitUit> as BenefitUnit stores cell0 index
        log::info!("sorting benefits");
        let mut benefits: Vec<BenefitUnit> =
            higher_best_finer_layer_contribution.into_values().collect();
        benefits.par_sort_unstable_by(|unita, unitb| {
            unitb.benefit.partial_cmp(&unita.benefit).unwrap()
        });
        //
        log::info!("benefits vector size : {}", benefits.len());
        assert!(benefits[0].benefit >= benefits[1].benefit);
        log::info!("benefits vector size : {}", benefits.len());
        if log::log_enabled!(log::Level::Debug) {
            for (_, unit) in benefits.iter().enumerate().take(100) {
                log::debug!(" id : {:?} , benef : {}", unit.get_id(), unit.get_benefit());
            }
        }
        //
        log::info!("exiting compute_benefits");
        let cpu_time: Duration = cpu_start.elapsed();
        println!(
            " compute_benefits sys time(ms) {:?} cpu time {:?}",
            sys_now.elapsed().unwrap().as_millis(),
            cpu_time.as_millis()
        );
        //
        benefits
    } // end of compute_benefits_1

    //

    // compute benefits by points (in fact cells at finest layer, at max_level in our implementation, and at level 0 in paper ) and layer (See algo 2 of paper and lemma 3.4)
    pub(crate) fn compute_benefits_2(&self) -> Vec<BenefitUnit> {
        //
        log::info!("in compute_benefits_2");
        let cpu_start = ProcessTime::now();
        let sys_now = SystemTime::now();
        //
        let nb_layers = self.get_nb_layers();
        let max_layer = (nb_layers - 1).try_into().unwrap();
        // for each cell at finest level we compute merge benefit at each level!
        // keys are indexes at finest layer!
        let mut all_benefits = HashMap::<Vec<u16>, Vec<usize>>::new();
        // TODO: to be made parallel with a DashMap
        let finest_layer = self.get_layer(max_layer);
        for cell in finest_layer.get_hcells().iter() {
            let mut benefit_layer = vec![0usize; nb_layers];
            let mut previous_tree_size: usize = cell.get_subtree_size();
            for l in (0..max_layer).rev() {
                let upper_index = cell.get_larger_cell_index_at_layer(l as u16);
                let upper_cell = self.get_layer(l as u16).get_cell(&upper_index).unwrap();
                let delta_l = max_layer - l;
                benefit_layer[l as usize] = 2_usize.pow(delta_l as u32) * previous_tree_size
                    + benefit_layer[l as usize + 1];
                // loop update
                previous_tree_size = upper_cell.get_subtree_size();
            }
            all_benefits.insert(cell.get_cell_index().to_vec(), benefit_layer);
        }
        //
        // we map cells above finest layer  (index at layer, layer) to (finest layer cell, benefit)
        // to keep track which the (index of cell at finest layer , layer of oberved benefit, benefit) provides highest benefit at l
        type CellAtL = (Vec<u16>, u16); // index at l, l
        type BestFinestCell = (Vec<u16>, u16, usize); // index at finest layer, layer , benefit
                                                      //
        let mut best_cell_at_finer_level_contribution = HashMap::<CellAtL, BestFinestCell>::new();
        let finest_layer = self.get_layer(max_layer);
        for cell in finest_layer.get_hcells().iter() {
            let index = cell.get_cell_index();
            let benefit = all_benefits.get(index).unwrap(); // we get benefits at all layers for this cell at finest level
            for l in (0..max_layer).rev() {
                let upper_index = cell.get_larger_cell_index_at_layer(l as u16);
                let cell_l: CellAtL = (upper_index, l as u16);
                if let Some(best_item) = best_cell_at_finer_level_contribution.get_mut(&cell_l) {
                    if benefit[l as usize] > best_item.2 {
                        // our current cell at finest layer has better benefit at layer l!
                        *(best_item) = (index.to_vec(), l as u16, benefit[l as usize]);
                    }
                } else {
                    best_cell_at_finer_level_contribution.insert(
                        (cell_l.0, l as u16),
                        (index.to_vec(), l as u16, benefit[l as usize]),
                    );
                }
            }
        }
        log::info!(
            "compute_benefits.  best_cell_at_finer_level_contribution len {}",
            best_cell_at_finer_level_contribution.len()
        );

        // for each cell at layer l we know from which  finest level cell the best benefit originates
        // and we keep cells in best_cell_at_finer_level_contribution with the LOWER!!! l
        let best_cell_finest: Vec<BestFinestCell> = best_cell_at_finer_level_contribution
            .into_values()
            .collect();
        let mut best_cell_finest_at_higher_l = HashMap::<Vec<u16>, (u16, usize)>::new();
        for bcf in best_cell_finest {
            log::trace!("bcf (index at 0, layer, benefit): {:?}", bcf);
            if let Some(l_and_benefit) = best_cell_finest_at_higher_l.get_mut(&bcf.0) {
                // lower l ?
                if bcf.1 < l_and_benefit.0 {
                    ////////////////                assert!(bcf.2 > l_and_benefit.1);
                    if bcf.2 > l_and_benefit.1 {
                        // we have a lower l
                        *(l_and_benefit) = (bcf.1, bcf.2);
                    } else {
                        log::debug!("bcf {:?},l_and_benefit : {:?} ", bcf, l_and_benefit);
                        panic!();
                    }
                }
            } else {
                best_cell_finest_at_higher_l.insert(bcf.0, (bcf.1, bcf.2));
            }
        }
        // we can collect BenefitUnit and sort them
        let mut benefits = Vec::<BenefitUnit>::with_capacity(best_cell_finest_at_higher_l.len());
        for bc0_l in best_cell_finest_at_higher_l {
            log::trace!("bc0l (index at 0, layer, benefit): {:?}", bc0_l);
            let benefit = BenefitUnit::new(bc0_l.0, bc0_l.1 .0, bc0_l.1 .1);
            benefits.push(benefit);
        }
        benefits.par_sort_unstable_by(|unita, unitb| {
            unitb.benefit.partial_cmp(&unita.benefit).unwrap()
        });
        log::info!("benefits vector size : {}", benefits.len());
        if log::log_enabled!(log::Level::Debug) {
            for (_, unit) in benefits.iter().enumerate().take(100) {
                log::debug!(" id : {:?} , benef : {}", unit.get_id(), unit.get_benefit());
            }
        }
        //
        log::info!("exiting compute_benefits");
        let cpu_time: Duration = cpu_start.elapsed();
        println!(
            " compute_benefits sys time(ms) {:?} cpu time {:?}",
            sys_now.elapsed().unwrap().as_millis(),
            cpu_time.as_millis()
        );
        //
        benefits
    } // end of compute_benefits_2

    /// builds a partition of size p_size from sorted benefits
    pub(crate) fn get_partition(
        &self,
        p_size: usize,
        benefit_units: &[BenefitUnit],
    ) -> DashMap<PointId, u32> {
        //
        let nb_points = self.get_nb_points();
        log::info!(
            "in SpaceMesh::get_partiton dispatching nbpoints: {}, in {} clusters",
            nb_points,
            p_size
        );
        let clusters = DashMap::<PointId, u32>::with_capacity(nb_points);
        //
        let loop_min_size = p_size.min(benefit_units.len());
        for i in 0..loop_min_size {
            log::info!("benefit unit : {}", i);
            let unit = &benefit_units[i];
            let (_, layer) = unit.get_id();
            let idx_l = unit.get_index_at_layer(self.get_nb_layers() as u16);
            // get cell
            let ref_cell_opt = self.get_cell(&idx_l, layer);
            if ref_cell_opt.is_none() {
                log::error!(
                    "unit {} has no cell , idx : {:?}, layer : {}",
                    i,
                    idx_l,
                    layer
                );
                panic!("get_partition_by_size failed");
            }
            let ref_cell = ref_cell_opt.unwrap();
            let unit_cell = ref_cell.value();
            log::debug!(
                "cell : {:?}, layer : {}, nb points : {}",
                idx_l,
                layer,
                unit_cell.get_nb_points()
            );
            let points = unit_cell.get_points().unwrap();
            // what is exclusively in unit i and not in sub-sequent units
            let nbpoint_i = AtomicUsize::new(0);
            points.par_iter().for_each(|point| {
                let mut j = i + 1;

                let found: bool = loop {
                    if j >= benefit_units.len() {
                        break false;
                    }
                    let unit_j = &benefit_units[j];
                    let (_, layer_j) = unit_j.get_id();
                    let idx_j = unit_j.get_index_at_layer(self.get_nb_layers() as u16);
                    let ref_cell_j = self.get_cell(&idx_j, layer_j).unwrap();
                    if ref_cell_j.has_point(point.get_id()) {
                        break true;
                    } else if j < p_size - 1 {
                        j += 1;
                    } else {
                        break false;
                    }
                };
                if !found {
                    // point is exclusevily in cell i
                    if let Some(old) = clusters.insert(point.get_id(), i as u32) {
                        log::error!(
                            "point id {:?} was already inserted in cluster : {:?}",
                            point.get_id(),
                            old
                        );
                        log::info!(
                            "number of points dispatched in clusters : {:?}",
                            clusters.len()
                        );
                        panic!();
                    }
                    log::trace!(
                        "point {:?} x... = {:?}  inserted in cluster {}",
                        point.get_id(),
                        point.get_position(),
                        i
                    );
                } // end !found
                let old = nbpoint_i.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                if log::log_enabled!(log::Level::Debug) && old % 10000 == 0 {
                    log::debug!("nb points dispacthed to clusters : {}", old);
                }
            }); // end for on points
        } // end for i <= p_size
        log::info!(
            "points to dispatch : {}, nb points in clusters : {}",
            nb_points,
            clusters.len()
        );
        //
        clusters
    }

    // we need to compute cardinal of each subtree (all descendants of each cell)
    // just dispatch number of points
    fn compute_subtree_size(&self) {
        //
        log::info!("in compute_subtree_size ");
        for l in (0..self.get_nb_layers()).rev() {
            let layer = &self.layers[l];
            layer.get_hcells().par_iter_mut().for_each(|mut cell| {
                cell.subtree_size = cell.points_in.as_ref().unwrap().len();
            });
            self.get_subtree_size(l as u16);
        }
        log::info!("exiting compute_subtree_size ");
    } // end of compute_subtree_cardinals

    // sum of sub tree size as seen from layer l. Used for debugging purpose
    fn get_subtree_size(&self, l: u16) -> usize {
        let size: usize = self.layers[l as usize]
            .get_hcells()
            .iter()
            .map(|mesh_cell| mesh_cell.get_subtree_size())
            .sum();
        log::trace!("total sub tree size at layer l : {} = {}", l, size);
        assert_eq!(size, self.get_nb_points());
        //
        size
    }
} // end of impl SpaceMesh

//==================

pub(crate) fn dump_benefits<T>(spacemesh: &SpaceMesh<T>, benefits: &[BenefitUnit])
where
    T: Float + Debug + std::fmt::LowerExp + Sync,
{
    let dump_size = 100;
    log::info!("dumping first {} units", dump_size);
    for (rank, unit) in benefits.iter().enumerate().take(dump_size) {
        log::debug!("\n\n unit rank {}, {:?}", rank, unit);
        let cell_idx0 = unit.get_cell_idx();
        let cell = spacemesh.get_cell(cell_idx0, 0).unwrap();
        let center = spacemesh.get_cell_center(cell_idx0, 0).unwrap();
        log::debug!("nb point : {}", cell.get_nb_points());
        if log::log_enabled!(log::Level::Debug) {
            print!("\n center : ");
            for x in center {
                print!(" {:.3e}", x);
            }
            println!();
        }
    }
}

// we check we have a partition
pub(crate) fn check_benefits_cover<T>(spacemesh: &SpaceMesh<T>, benefits: &Vec<BenefitUnit>)
where
    T: Float + Debug + std::fmt::LowerExp + Sync,
{
    log::debug!("entering in check_benefits_cover");
    let mut nb_points_in: usize = 0;
    for unit in benefits {
        let (finer_idx, l) = unit.get_id();
        let shift = spacemesh.get_nb_layers() - 1 - l as usize;
        let up_idx: Vec<u16> = finer_idx.iter().map(|x| x >> shift).collect();
        if let Some(cell) = spacemesh.get_layer(l).get_cell(&up_idx) {
            assert!(cell.get_nb_points() > 0);
            nb_points_in += cell.get_nb_points();
        } else {
            log::error!(
                "spacemesh benefit unit {:?}  cannot find its cell, up_idx : {:?}, nb_points found {}",
                unit,
                up_idx,
                nb_points_in,
            );
            panic!();
        }
    }
    log::info!("nb points referenced in benefits : {}", nb_points_in);
    log::info!("nb points in space : {}", spacemesh.get_nb_points());
}

//===========================

// space must enclose points
#[allow(unused)]
fn check_space<T: Float + Debug>(space: &Space, points: &[Point<T>]) {
    assert!(!points.is_empty());
    //
    let mut max_xi = T::min_value();
    let mut min_xi: T = T::max_value();
    for (ipt, pt) in points.iter().enumerate() {
        for (i, xi) in pt.get_position().iter().enumerate() {
            if space.get_xmin() > <f64 as num_traits::NumCast>::from(*xi).unwrap() {
                log::error!(
                    "space.get_xmin() too high,  point of rank {} has coordinate i : {}, xi = {:?} ",
                    ipt,
                    i,
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
    log::debug!(
        "minimum space, xmin : {:.3e}, width : {:.3e}",
        <f64 as num_traits::NumCast>::from(min_xi).unwrap(),
        <f64 as num_traits::NumCast>::from(delta).unwrap()
    );
} // end of check_space

//========================================================

#[cfg(test)]
mod tests {

    use super::*;

    use rand::distr::Uniform;
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
        let nbvec = 1_000usize;
        let dim = 5;
        let width: f64 = 1000.;
        let mindist = 5.;
        let unif_width = Uniform::<f64>::new(0., width).unwrap();
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(234567_u64);
        let mut points: Vec<Point<f64>> = Vec::with_capacity(nbvec);
        for i in 0..nbvec {
            let p: Vec<f64> = (0..dim).map(|_| unif_width.sample(&mut rng)).collect();
            points.push(Point::<f64>::new(i, p, (i % 5).try_into().unwrap()));
        }
        let refpoints: Vec<&Point<f64>> = points.iter().map(|p| p).collect();
        // Space definition
        let mut space = Space::new(dim, 0., width, Some(mindist));
        let mut mesh = SpaceMesh::new(&mut space, refpoints);

        //
        mesh.embed();
        mesh.summary();
        // check cell basics, center, index conversion
        let pt = points[0].get_position();
        log::info!("locating pt : {:?}", pt);
        for l in 0..mesh.get_nb_layers() {
            let idx = mesh.get_cell_index(pt, l);
            log::info!("get_cell_index pt at idx : {:?} for l : {}", idx, l);
            let idx = mesh.get_cell_with_position(pt, l).unwrap();
            log::info!(
                "get_cell_with_position pt at idx : {:?} for l : {}",
                idx.key(),
                l
            );
        }
        //
        mesh.compute_subtree_size();
        //
        let _benefits = mesh.compute_benefits(2);
    } //end of test_uniform_random

    #[test]
    fn chech_dashmap() {
        let size = 10_000_000;
        let data = vec![1u32; size];
        let hmap = DashMap::<usize, u32>::new();
        (0..size).into_par_iter().for_each(|i| {
            let _res = hmap.insert(i, data[i]);
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
        let dim = 15;
        let width: f64 = 1000.;
        let mindist = 0.001;
        // Space definition
        let _space = Space::new(dim, 0., width, Some(mindist));
    } // end of check_mindist

    #[test]
    fn test_exp_random() {
        log_init_test();
        log::info!("in test_exp_random");
        //
        let nbvec = 1_000_000_usize;
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
        let mut space = Space::new(dim, 0., width, Some(mindist));

        let mut mesh = SpaceMesh::new(&mut space, refpoints);
        mesh.embed();
        mesh.summary();
        //
        let _benefits = mesh.compute_benefits(2);
        // check number of points for cell at origin
        log::info!("number of points at orgin 0.001 .... 0.001]");
        let p = vec![0.001; dim];
        log::info!("cells info for p : {:?}", p);
        for l in 0..mesh.get_nb_layers() {
            if let Some(cell) = mesh.get_cell_with_position(&p, l) {
                let nbpoints_in = cell.value().get_nb_points();
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
