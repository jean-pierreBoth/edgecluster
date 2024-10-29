//! This is a small rhst implementation as described in cohen-addad paper
//!
#![allow(clippy::needless_range_loop)]

use num_traits::cast::*;
use num_traits::float::Float;

use dashmap::DashMap;

type NodeId = usize;
type PointId = usize;

struct Point<T: Float> {
    // id to identify points
    id: PointId,
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
    /// return coordinate of a cell for a point at layer l layer 0 is at finer scale
    pub fn get_cell<T: Float>(self, p: Vec<T>, l: usize) -> Vec<usize> {
        let exp: u32 = (self.layer_max - l).try_into().unwrap();
        let cell_size = self.width / 2usize.pow(exp) as f64;
        let mut coordinates = Vec::<usize>::with_capacity(self.dim);
        for d in 0..self.dim {
            let idx: usize = ((p[d].to_f64().unwrap() - self.xmin) / cell_size).trunc() as usize;
            coordinates.push(idx);
        }
        coordinates
    }
} // end of impl Space

// a node correspond to cubic cell mesh at scale depending on layer.
// a node must keep track of points in it (excluding those in children)
// at the end of subdivision process only leaves have a list of point
pub struct Node<T: Float> {
    id: NodeId,
    // we must know one' layer
    layer: usize,
    /// a vector of dimension d
    position: Vec<usize>,
    // coordinates in the layer
    parent: Option<NodeId>,
    //
    children: Option<Vec<NodeId>>,
    // value of an upper edge
    up_edge_value: T,
    //
    points: Option<Vec<PointId>>,
}

impl<T: Float> Node<T> {
    //
    fn insert(&mut self, point: Point<T>) {
        panic!("not yet implemented")
    }
} // end of impl Node

//======

/// a layer gathers nodes if a given layer.
struct Layer<T: Float> {
    space: Space,
    // my layer
    layer: usize,
    //
    cell_diameter: T,
    // nodes in layer
    nodes: Vec<Node<T>>,
    // hashmap to index in nodes
    hnodes: DashMap<Vec<usize>, usize>,
}

impl<T: Float> Layer<T> {
    //
    fn insert(&self, point: &Point<T>) {
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
pub struct Tree<T: Float> {
    points: Vec<Point<T>>,
    // benefit of each points. For
    benefits: Vec<Vec<T>>,
    // space description
    space: Space,
    // largest coordinate value
    root: Node<T>,
    // nodes indexed by their id
    layers: Vec<Layer<T>>,
    // random shift
    rshift: [T; 3],
}

impl<T> Tree<T>
where
    T: Float,
{
    pub fn new(space: Space) -> Self {
        panic!("not yet implemented");
    }
    /// returns cell diameter at layer l
    pub fn get_cell_diam(&self, l: usize) -> f64 {
        panic!("not yet implemented");
    }

    fn get_cell_at_layer(&self, point: &[T], layer: usize) -> Vec<usize> {
        panic!("not yet implemented");
    }

    /// insert a point
    pub fn insert(&mut self, point: &[T]) {
        panic!("not yet implemented");
    }

    /// generate random shift
    fn generate_shift(&mut self) -> Vec<T> {
        panic!("not yet implemented");
    }
} // end of impl Tree
