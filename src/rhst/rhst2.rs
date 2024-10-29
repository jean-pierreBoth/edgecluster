//! This is a small rhst implementation as described in cohen-addad paper
//!
#![allow(clippy::needless_range_loop)]

use num_traits::cast::*;
use num_traits::float::Float;

use dashmap::DashMap;

type NodeId = usize;
type PointId = usize;

pub(crate) struct Point<T: Float> {
    // id to identify points
    id: PointId,
    // data point
    p: Vec<T>,
    /// original label
    label: u32,
}

impl<T> Point<T>
where
    T: Float,
{
    pub fn get_label(&self) -> u32 {
        self.label
    }
} // end of impl Point

//====

//TODO: space must provide for random shift!!!

#[cfg_attr(doc, katexit::katexit)]
/// structure describing space in which data point are supposed to live.
/// Data are supposed to live in an enclosing box $$[xmin, xmax]^dim$$
/// Space is seen at different resolution, each resolution is finer with each cell size divided by 2.
/// The coarser mesh will be upper layer.
///
/// The algo will slice space in different layers with cells side decreasing by a factor 2 at each layer
/// Cells are nodes in a tree, each cell , if some data is present, will have children in lower cells.
#[derive(Debug, Copy, Clone)]
pub struct Space {
    // space dimension
    dim: usize,
    // global amplitude in each dimension
    width: f64,
    // smallest coordinate
    xmin: f64,
    // distance under which point are considered equal, points kept in current node
    mindist: f64,
    // leaves are at 0, root node will be at layer_max
    layer_max: usize,
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
        let layer_max: usize = (width / mindist).log2().ceil() as usize;
        Space {
            dim,
            xmin,
            width: xmax - xmin,
            mindist,
            layer_max,
        }
    }
    /// returns cell diameter at layer l
    pub fn get_cell_diam(&self, l: usize) -> f64 {
        panic!("not yet implemented");
    }

    /// return space dimension
    pub fn get_dim(&self) -> usize {
        self.dim
    }
    /// return space width
    pub fn get_width(&self) -> f64 {
        self.width
    }

    /// returns maximum layer , or layer of root node
    pub fn get_root_layer(&self) -> usize {
        self.layer_max
    }

    /// return coordinate of a cell for a point at layer l layer 0 is at finer scale
    pub fn get_cell<T: Float>(&self, p: &[T], l: usize) -> Cell {
        let exp: u32 = (self.layer_max - l).try_into().unwrap();
        let cell_size = self.width / 2usize.pow(exp) as f64;
        let mut coordinates = Vec::<usize>::with_capacity(self.dim);
        for d in 0..self.dim {
            let idx: usize = ((p[d].to_f64().unwrap() - self.xmin) / cell_size).trunc() as usize;
            coordinates.push(idx);
        }
        Cell {
            space: &self,
            layer: l,
            position: coordinates,
        }
    }
} // end of impl Space

/// space is split in cells , at upper layer  there is one cell for the whole space
/// at layer 0 cells have size corresponding to mindit
#[derive(Debug, Clone)]
pub struct Cell<'a> {
    space: &'a Space,
    // we must know one' layer
    layer: usize,
    /// a vector of dimension d, giving center of cell
    position: Vec<usize>,
}

impl<'a> Cell<'a> {
    pub fn new(space: &'a Space, layer: usize, position: &[usize]) -> Self {
        Cell {
            space,
            layer,
            position: Vec::from(position),
        }
    }
}
// a node correspond to cubic cell mesh at scale depending on layer.
// a node must keep track of points in it (excluding those in children)
// at the end of subdivision process only leaves have a list of point
pub struct Node<'a> {
    id: NodeId,
    // corresponding cell
    cell: Cell<'a>,
    // coordinates in the layer
    parent: Option<NodeId>,
    //
    children: Option<Vec<NodeId>>,
    //
    points: Option<Vec<PointId>>,
}

impl<'a> Node<'a> {
    //

    fn new(id: NodeId, cell: Cell<'a>, parent: Option<NodeId>) -> Self {
        Node {
            id,
            cell,
            parent,
            children: None,
            points: None,
        }
    }
    //
    fn insert<T: Float>(&mut self, point: Point<T>) {
        panic!("not yet implemented")
    }
} // end of impl Node

//======

/// a layer gathers nodes if a given layer.
struct Layer<'a> {
    space: Space,
    // my layer
    layer: usize,
    //
    cell_diameter: f64,
    // nodes in layer
    nodes: Vec<Node<'a>>,
    // hashmap to index in nodes
    hnodes: DashMap<Vec<usize>, usize>,
}

impl<'a> Layer<'a> {
    //
    fn new(space: Space, layer: usize) -> Self {
        let nodes = Vec::<Node>::new();
        let hnodes: DashMap<Vec<usize>, usize> = DashMap::new();
        let cell_diameter = (space.get_width() * space.get_dim() as f64).sqrt();
        Layer {
            space,
            layer,
            cell_diameter,
            nodes,
            hnodes,
        }
    }
    //
    fn insert<T: Float>(&self, point: &Point<T>) {
        panic!("not yet implemented");
    }
}

//=======

// We prefer rhst over cover trees as described in
//  TerraPattern: A Nearest Neighbor Search Service (2019)
//  Zaheer, Guruganesh, Levin Smola
//  as we can group points very close (under lower distance threshold)
//
// All points coordinates are in [-xmin, xmax]^d
// lower layer (leaves) is 0 and correspond to smaller cells
// the number of layers is log2(width/mindist)
// a cell coordinate have value between 0 and log2(ratio)
//

//
pub(crate) struct Tree<'a, T: Float> {
    points: Vec<Point<T>>,
    // benefit of each points. For
    benefits: Vec<Vec<T>>,
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
