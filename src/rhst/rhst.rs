//! This is a small rhst implementation as described in cohen-addad paper

use num_traits::float::Float;

use dashmap::DashMap;

type NodeId = usize;
type PointId = usize;

struct Point<T: Float> {
    id: PointId,
    p: Vec<T>,
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
    /// my layer
    layer: usize,
    ///
    cell_diameter: T,
    /// nodes in layer
    nodes: Vec<Node<T>>,
    /// hashmap to index in nodes
    hnodes: DashMap<Vec<usize>, usize>,
}

//=======

// We prefer rhst over cover trees as described in
//  TerraPattern: A Nearest Neighbor Search Service (2019)
//  Zaheer, Guruganesh, Levin Smola
//  as we can group points very close (under lower distance threshold)
//
// All points cooridnates are in [-xmin, xmax]^d
// lower layer (leaves) is 0 and correspond to smaller cells
// the number of layers is log2(width/mindist)
// a cell coordinate have value between 0 and log2(ratio)
//

///
pub struct Tree<T>
where
    T: Float,
{
    // space dimension
    dim: usize,
    // distance under which point are considered equal, points kept in current node
    mindist: f64,
    // global amplitude in each dimension
    width: f64,
    // smallest coordinate
    xmin: T,
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
    /// returns cell diameter at layer l
    pub fn get_cell_diam(&self, l: usize) -> f64 {
        panic!("not yet implemented");
    }

    fn get_cell_at_layer(&self, point: &Vec<T>, layer: usize) -> Vec<usize> {
        panic!("not yet implemented");
    }

    /// insert a point
    pub fn insert(&mut self, point: &Vec<T>) {
        panic!("not yet implemented");
    }

    /// generate random shift
    fn generate_shift(&mut self) -> Vec<T> {
        panic!("not yet implemented");
    }
} // end of impl Tree
