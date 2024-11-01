//! This is a small rhst implementation as described in cohen-addad paper
//! It requires data to have small/moderated dimension. A first step can be dimension reduction
//!
#![allow(clippy::needless_range_loop)]

// We prefer rhst over cover trees as described in
//  TerraPattern: A Nearest Neighbor Search Service (2019)
//  Zaheer, Guruganesh, Levin Smola
//  as we can group points very close (under lower distance threshold)

use num_traits::cast::*;
use num_traits::float::Float;

use std::fmt::Debug;

use dashmap::{mapref::one, DashMap};
use ego_tree::{tree, NodeMut, Tree};

type NodeId = usize;

/// data to cluster identifier
pub type PointId = usize;

// TODO: possibly Point should store only a ref to data?

#[derive(Debug, Clone)]
pub struct Point<T> {
    // id to identify points as coming from external client
    id: PointId,
    // data point
    p: Vec<T>,
    /// original label
    label: u32,
}

impl<T> Point<T>
where
    T: Float + Debug,
{
    pub fn get_label(&self) -> u32 {
        self.label
    }

    pub fn get_position(&self) -> &[T] {
        &self.p
    }
} // end of impl Point

//====

#[cfg_attr(doc, katexit::katexit)]
/// structure describing space in which data point are supposed to live.
/// Data are supposed to live in an enclosing box $$[xmin, xmax]^dim$$
#[derive(Debug, Copy, Clone)]
pub struct Space {
    // space dimension
    dim: usize,
    // global amplitude in each dimension
    width: f64,
    // smallest coordinate
    xmin: f64,
}

impl Space {
    /// define Space.
    /// -
    ///
    pub fn new(dim: usize, xmin: f64, xmax: f64) -> Self {
        let width = xmax - xmin;
        Space {
            dim,
            xmin,
            width: xmax - xmin,
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

    pub fn get_cell_center(&self, layer: usize, index: &[usize]) -> Vec<f64> {
        panic!("not yet implemented");
    }
}

//========================================

/// space is split in cells , at upper layer  there is one cell for the whole space
/// at layer 0 cells have size corresponding to mindist, so a cell, even at layer 0 can have more than one point
/// (they must be lesst than mindist apart)
#[derive(Debug, Clone)]
pub struct Cell<'a, T> {
    space: &'a Space,
    // we must know one' layer
    layer: usize,
    /// a vector of dimension d, giving center of cell in the mesh coordinates
    index: Vec<usize>,
    //
    points_in: Option<Vec<&'a Point<T>>>,
}

impl<'a, T> Cell<'a, T>
where
    T: Float + Debug,
{
    pub fn new(space: &'a Space, layer: usize, index: Vec<usize>) -> Self {
        Cell {
            space,
            layer,
            index,
            points_in: None,
        }
    }

    /// Return cel weight. This is also the cardinal of the subtree corresponding to a point in cell
    pub fn get_cell_weight(&self) -> f32 {
        if let Some(points) = self.points_in.as_ref() {
            points.len() as f32
        } else {
            0.
        }
    } // end of get_cell_weight

    fn get_cell_index(&self) -> &[usize] {
        &self.index
    }

    // adds a point in cell
    fn add_point(&mut self, point: &'a Point<T>) {
        match self.points_in.as_mut() {
            Some(points) => points.push(point),
            None => {
                let mut vec = Vec::new();
                vec.push(point);
                self.points_in = Some(vec);
            }
        }
    }

    // fill in points
    fn init_points(&mut self, points: &[&'a Point<T>]) {
        assert!(self.points_in.is_none());
        self.points_in = Some(<Vec<&'a Point<T>>>::from(points));
    }

    // This function parse points and allocate a subcell when a point requires it.
    fn split_cell(&self) {
        if self.points_in.is_none() {
            return;
        }
        for point in self.points_in.as_ref().unwrap() {
            let xyz = point.get_position();
            let mut split_index = Vec::<usize>::with_capacity(self.space.get_dim());
            let cell_center = self.space.get_cell_center(self.layer, &self.index);
            for i in 0..xyz.len() {
                if xyz[i].to_f64().unwrap() > cell_center[i] {
                    // point is in upper part
                    split_index.push(2 * self.index[i] + 1);
                } else {
                    // point is in lower part for this dimension
                    split_index.push(2 * self.index[i]);
                }
            }
        }
        panic!("not yet implemented");
    }
} // end of impl Cell

//======

/// a layer gathers nodes if a given layer.
struct Layer<'a, T> {
    space: Space,
    // my layer
    layer: usize,
    //
    cell_diameter: f64,
    // hashmap to index in nodes
    cells: DashMap<Vec<usize>, Cell<'a, T>>,
}

impl<'a, T> Layer<'a, T>
where
    T: Float + Debug,
{
    //
    fn new(space: Space, layer: usize, cell_diameter: f64) -> Self {
        let cells: DashMap<Vec<usize>, Cell<T>> = DashMap::new();
        Layer {
            space,
            layer,
            cell_diameter,
            cells,
        }
    }
    //
    fn insert(&self, cell: &Cell<'a, T>) {
        if self.get_cell(&cell.index).is_some() {
            panic!("internal error");
        }
        self.cells.insert(cell.index.clone(), cell.clone());
    }

    // returns a dashmap Ref to cell is it exist
    // used only to check coherence. Sub cells are created inside one thread.
    // TODO: do we need dashmap?
    fn get_cell(&self, position: &[usize]) -> Option<one::Ref<Vec<usize>, Cell<'a, T>>> {
        self.cells.get(position)
    }
} // end implementation Layer

//============================

//TODO: space must provide for random shift!!!

#[cfg_attr(doc, katexit::katexit)]
/// structure describing space in which data point are supposed to live.
/// Data are supposed to live in an enclosing box $$[xmin, xmax]^dim$$
/// Space is seen at different resolution, each resolution is finer with each cell size divided by 2.
/// The coarser mesh will be upper layer.
///
/// The algo will slice space in different layers with cells side decreasing by a factor 2 at each layer
/// Cells are nodes in a tree, each cell , if some data is present, will have children in lower cells.
pub struct SpaceMesh<'a, T: Float> {
    space: Space,
    //
    mindist: f64,
    // leaves are at 0, root node will be at layer_max
    layer_max: usize,
    //
    layers: Vec<Layer<'a, T>>,
    //
    points: Option<Vec<Point<T>>>,
    // benefit of each couple (point rank, layer)
    // benefits[r][l] gives  gives benefit of point of rank r in points at layer l
    benefits: Option<Vec<Vec<f32>>>,
}

impl<'a, T> SpaceMesh<'a, T>
where
    T: Float + Debug,
{
    /// define Space.
    /// -
    ///
    pub fn new(space: Space, points: Vec<Point<T>>, mindist: f64) -> Self {
        assert!(
            mindist > 0.,
            "mindist cannot be 0, should related to width so that max layer is not too large"
        );
        let layer_max: usize = (space.get_dim() as f64 * space.get_width().sqrt() / mindist)
            .log2()
            .ceil() as usize;
        log::info!("nb layer max : {}", layer_max);
        //
        let mut layers: Vec<Layer<T>> = Vec::with_capacity(layer_max + 1);
        //
        SpaceMesh {
            space,
            mindist,
            layer_max,
            layers,
            points: Some(points),
            benefits: None,
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

    /// returns maximum layer , or layer of root node
    pub fn get_root_layer(&self) -> usize {
        self.layer_max
    }

    pub fn get_layer_max(&self) -> usize {
        self.layer_max
    }
    ///
    pub fn get_mindist(&self) -> f64 {
        self.mindist
    }
    /// return coordinate of a cell for a point at layer l layer 0 is at finer scale
    pub fn get_cell_center(&self, p: &[T], l: usize) -> Vec<usize> {
        let exp: u32 = (self.layer_max - l).try_into().unwrap();
        let cell_size = self.get_width() / 2usize.pow(exp) as f64;
        let mut coordinates = Vec::<usize>::with_capacity(self.get_dim());
        for d in 0..self.get_dim() {
            let idx_f: f64 = ((p[d].to_f64().unwrap() - self.space.get_xmin()) / cell_size).trunc();
            if idx_f < 0. {
                log::error!(
                    "got negative index , coordinate value : {:.3e}",
                    p[d].to_f64().unwrap()
                );
                panic!("negative coordinate for cell");
            }
            coordinates.push(idx_f as usize);
        }
        coordinates
    }

    pub fn get_layer_cell_diameter(&self, layer: usize) -> f64 {
        let cell_diameter: f64 = (self.space.get_width() * self.space.get_dim() as f64).sqrt()
            / 2_u32.pow(self.layer_max as u32 - layer as u32) as f64;
        cell_diameter
    }

    pub fn cluster(&mut self) {
        // propagate points through layers form to (layer lmax to bottom layer 0
        // initialize root cell
        let mut upper_layer: Layer<T> = Layer::new(
            self.space,
            self.get_layer_max(),
            self.get_layer_cell_diameter(self.get_layer_max()),
        );
        let center = vec![0usize; self.get_dim()]; // root cell
        let mut global_cell: Cell<T> = Cell::new(&self.space, self.get_layer_max(), center);
        //
        self.points
            .as_ref()
            .unwrap()
            .iter()
            .map(|p| global_cell.add_point(p));
        // now to first layer (one cell) others layers can be made //
    }
} // end of impl SpaceMesh

//===========================

//=======

struct CostBenefit {}
//
// All points coordinates are in [-xmin, xmax]^d
// lower layer (leaves) is 0 and correspond to smaller cells
// the number of layers is log2(width/mindist)
// a cell coordinate have value between 0 and log2(ratio)
//

/*

//
pub(crate) struct Tree<'a, T: Float> {
    // space description
    space: Space,
    // largest coordinate value
    root: Node<'a>,
    // nodes indexed by their id
    layers: Vec<Layer<'a>>,
    // random shift
    rshift: [T; 3],
}

impl<'a, T> Tree<'a, T>
where
    T: Float,
{
    pub fn new(points: Vec<Point<T>>, mindist: f64) -> Self {
        // determine space,
        // allocate benefits, root node, layers
        // sample rshift
        panic!("not yet implemented");
    }

    // return indices of cell at cells
    fn get_cell_at_layer(&'a self, point: &[T], layer: usize) -> Cell<'a> {
        self.space.get_cell(point, layer)
    }

    /// insert a set of points and do clusterization
    pub fn cluster_set(&mut self, points: &[Vec<T>]) -> anyhow::Result<()> {
        panic!("not yet implemented")
    }

    /// generate random shift
    fn generate_shift(&mut self) -> Vec<T> {
        panic!("not yet implemented");
    }
} // end of impl Tree


*/
