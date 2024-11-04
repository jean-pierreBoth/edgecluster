#![allow(clippy::needless_range_loop)]

/// This is a small rhst implementation as described in Cohen-Addad paper  
/// It requires data to have small/moderated dimension. This can be achieved using a first step of dimension reduction
/// via random projections or an embedding via the crate [annembed](https://crates.io/crates/annembed)
///
///
// We prefer rhst over cover trees as described in
//  TerraPattern: A Nearest Neighbor Search Service (2019)
//  Zaheer, Guruganesh, Levin Smola
//  as we can group points very close (under lower distance threshold) in one cell
use num_traits::cast::*;
use num_traits::float::Float;

use std::fmt::Debug;

use dashmap::{iter, mapref::one, rayon, DashMap};

use ego_tree::{tree, NodeMut, Tree};
use std::collections::HashMap;

type NodeId = usize;

/// data to cluster identifier
pub type PointId = usize;

// TODO: possibly Point should store only a ref to data?

#[derive(Debug, Clone)]
pub struct Point<T> {
    // id to identify points as coming from external client
    id: PointId,
    /// data point
    p: Vec<T>,
    /// original label
    label: u32,
}

impl<T> Point<T>
where
    T: Float + Debug,
{
    pub fn new(id: PointId, p: Vec<T>, label: u32) -> Self {
        Point { id, p, label }
    }
    /// get the original label
    pub fn get_label(&self) -> u32 {
        self.label
    }

    /// gets the points coordinate
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
        let layer_max_u: usize =
            (dim as f64 * width.sqrt() / mindist).log2().ceil().trunc() as usize;
        let nb_layer = layer_max_u + 1;
        log::info!("nb layer : {}", nb_layer);
        if (nb_layer >= 8) {
            log::warn!("perhaps increase mindist to reduce nb_layer");
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
#[derive(Debug, Clone)]
pub(crate) struct Cell<'a, T> {
    space: &'a Space,
    // we must know one' layer
    layer: u32,
    /// a vector of dimension d, giving center of cell in the mesh coordinates
    index: Vec<u32>,
    //
    points_in: Option<Vec<&'a Point<T>>>,
}

impl<'a, T> Cell<'a, T>
where
    T: Float + Debug,
{
    pub fn new(space: &'a Space, layer: u32, index: Vec<u32>) -> Self {
        Cell {
            space,
            layer,
            index,
            points_in: None,
        }
    }

    //
    pub fn get_layer(&self) -> usize {
        self.layer as usize
    }

    /// Return cel weight. This is also the cardinal of the subtree corresponding to a point in cell
    pub fn get_cell_weight(&self) -> f32 {
        if let Some(points) = self.points_in.as_ref() {
            points.len() as f32
        } else {
            0.
        }
    } // end of get_cell_weight

    fn get_cell_index(&self) -> &[u32] {
        &self.index
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
        let exponent: i32 = (self.space.nb_layer - self.layer as usize)
            .try_into()
            .unwrap();
        let cell_width = self.space.width / 2.0_f64.powi(exponent);
        let position: Vec<f64> = (0..self.space.dim)
            .map(|i| 0.5 * cell_width + self.index[i] as f64 * self.space.width)
            .collect();
        //
        position
    } // end of get_cell_center

    // This function parse points and allocate a subcell when a point requires it.
    fn split(&self) -> Option<Vec<Cell<'a, T>>> {
        //
        if self.points_in.is_none() {
            log::debug!("no points in cell");
            self.points_in.as_ref()?; /* return in case of None */
        }
        //
        let mut hashed_cells: HashMap<Vec<u32>, Cell<T>> = HashMap::new();
        //
        for point in self.points_in.as_ref().unwrap() {
            let xyz = point.get_position();
            let mut split_index = Vec::<u32>::with_capacity(self.space.get_dim());
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
                let new_cell: Cell<T> = Cell::new(self.space, self.layer - 1, split_index.clone());
                hashed_cells.insert(split_index.clone(), new_cell);
            }
        }
        // do not forget to fill new_cells !!
        let mut new_cells: Vec<Cell<T>> = Vec::new();
        for (k, v) in hashed_cells.drain() {
            new_cells.push(v);
        }
        //
        log::debug!(
            "layer : {}, cell with nb points {:?} splitted in {:?} new cells",
            self.layer,
            self.points_in.as_ref().unwrap().len(),
            new_cells.len()
        );
        //
        Some(new_cells)
    }
} // end of impl Cell

//======

/// a layer gathers nodes if a given layer.
struct Layer<'a, T> {
    space: &'a Space,
    // my layer
    layer: u32,
    //
    cell_diameter: f64,
    // hashmap to index in nodes
    hcells: DashMap<Vec<u32>, Cell<'a, T>>,
}

impl<'a, T> Layer<'a, T>
where
    T: Float + Debug + Sync,
{
    //
    fn new(space: &'a Space, layer: u32, cell_diameter: f64) -> Self {
        let hcells: DashMap<Vec<u32>, Cell<T>> = DashMap::new();
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
    fn get_cell(&self, position: &[u32]) -> Option<one::Ref<Vec<u32>, Cell<'a, T>>> {
        self.hcells.get(position)
    }

    // get an iterator over cells
    fn get_iter_mut(&self) -> iter::IterMut<Vec<u32>, Cell<'a, T>> {
        self.hcells.iter_mut()
    }

    // get an iterator over cells
    fn get_par_iter_mut(&self) -> rayon::map::IterMut<Vec<u32>, Cell<'a, T>> {
        self.hcells.par_iter_mut()
    }

    fn get_nb_cells(&self) -> usize {
        self.hcells.len()
    }

    fn get_diameter(&self) -> f64 {
        self.cell_diameter
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
    space: &'a Space,
    // leaves are at 0, root node will be at layer_max
    layer_max: u32,
    //
    global_cell: Option<Cell<'a, T>>,
    // layers are stroed in vector according to their layer num
    layers: Vec<Layer<'a, T>>,
    //
    points: Option<Vec<&'a Point<T>>>,
    // benefit of each couple (point rank, layer)
    // benefits[r][l] gives  gives benefit of point of rank r in points at layer l
    benefits: Option<Vec<Vec<f32>>>,
}

impl<'a, T> SpaceMesh<'a, T>
where
    T: Float + Debug + Sync,
{
    /// define Space.
    /// - data points to cluster
    /// - mindist to discriminate points. under this distance points will be associated
    ///
    pub fn new(space: &'a Space, points: Vec<&'a Point<T>>) -> Self {
        assert!(
            space.mindist > 0.,
            "mindist cannot be 0, should related to width so that max layer is not too large"
        );
        let layer_max_u: usize = (space.get_dim() as f64 * space.get_width().sqrt() / space.mindist)
            .log2()
            .ceil()
            .trunc() as usize;
        let nb_layer = layer_max_u + 1;
        log::info!("nb layer : {}", nb_layer);
        if (nb_layer >= 8) {
            log::warn!("perhaps increase mindist to reduce nb_layer");
        }
        let layer_max: u32 = layer_max_u.try_into().unwrap();
        //
        let mut layers: Vec<Layer<T>> = Vec::with_capacity(layer_max as usize + 1);
        //
        SpaceMesh {
            space,
            layer_max,
            global_cell: None,
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
    pub fn get_root_layer(&self) -> u32 {
        self.layer_max
    }

    pub fn get_layer_max(&self) -> u32 {
        self.layer_max
    }
    pub fn get_nb_layers(&self) -> usize {
        self.layer_max as usize + 1
    }

    /// returns minimum distance detectable by mesh
    pub fn get_mindist(&self) -> f64 {
        self.space.mindist
    }

    /// return index in mesh of a cell for a point at layer l layer 0 is at finer scale
    pub fn get_cell_index(&self, p: &[T], l: usize) -> Vec<usize> {
        let exp: u32 = (self.layer_max as usize - l).try_into().unwrap();
        let cell_size = self.get_width() / 2_usize.pow(exp) as f64;
        let mut index = Vec::<usize>::with_capacity(self.get_dim());
        for d in 0..self.get_dim() {
            let idx_f: f64 = ((p[d].to_f64().unwrap() - self.space.get_xmin()) / cell_size).trunc();
            if idx_f < 0. {
                log::error!(
                    "got negative index , coordinate value : {:.3e}",
                    p[d].to_f64().unwrap()
                );
                panic!("negative coordinate for cell");
            }
            index.push(idx_f as usize);
        }
        index
    } // end get_cell_index

    /// for layer 0, the layer with the least number of cells,
    /// the diameter of a cell is $$ width * \sqrt(d)/2^(nb_layer - layer_max)  $$
    ///  - Delta max value of  extension by dimension
    ///  - d : dimension of space
    ///  - l : layer num in 0..nb_layer
    pub fn get_layer_cell_diameter(&self, layer: u32) -> f64 {
        assert!(layer <= self.layer_max);
        //
        let cell_diameter: f64 = (self.space.get_width() * (self.space.get_dim() as f64).sqrt())
            / 2_u32.pow(self.layer_max + 1 - layer) as f64;
        cell_diameter
    }

    // we propagate cell of  upper layer (index 0 in vec) to lower (smaller cells) layers
    fn upper_layer_downward(&'a self, cell: &Cell<T>) {
        let upper_layer = self.layers.last().unwrap();
    }

    /// this function embeds data in a 2-rhst
    /// It propagate cells partitioning space from coarser to finer mesh
    pub fn embed(&mut self) {
        // initialize layers (layer lmax to bottom layer 0
        self.layers = Vec::with_capacity(self.get_nb_layers());
        for l in 0..self.get_nb_layers() {
            let layer =
                Layer::<T>::new(self.space, l as u32, self.get_layer_cell_diameter(l as u32));

            self.layers.push(layer);
        }
        log::info!("SpaceMesh::embed allocated nb layers {}", self.layers.len());
        // initialize root cell
        let center = vec![0u32; self.get_dim()];
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
            let cell_iter = self.layers[l].get_iter_mut();
            let lower_layer = &self.layers[l - 1];
            for cell in cell_iter {
                assert_eq!(l, cell.get_layer());
                let new_cells = cell.split();
                if new_cells.is_some() {
                    lower_layer.insert_cells(new_cells.unwrap());
                }
            }
        }
        //
        log::info!("end of downward cell propagation");
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
                "layer : {}, nb cells : {}, cell diameter : {:.3e}",
                l,
                self.layers[l].get_nb_cells(),
                self.layers[l].get_diameter()
            );
        }
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

//========================================================

#[cfg(test)]
mod tests {

    use super::*;

    use rand::distributions::Uniform;
    use rand::prelude::*;
    use rand_xoshiro::Xoshiro256PlusPlus;

    fn log_init_test() {
        let _ = env_logger::builder().is_test(true).try_init();
    }

    #[test]
    fn test_uniform_random() {
        log_init_test();
        log::info!("in test_uniform_random");
        //
        let nbvec = 2000usize;
        let dim = 5;
        let width: f64 = 1000.;
        let mindist = 0.1;
        let unif_01 = Uniform::<f64>::new(0., width);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(234567_u64);
        let mut points: Vec<Point<f64>> = Vec::with_capacity(nbvec);
        for i in 0..nbvec {
            let p: Vec<f64> = (0..dim).map(|_| unif_01.sample(&mut rng)).collect();
            points.push(Point::<f64>::new(i, p, (i % 5).try_into().unwrap()));
        }
        let refpoints: Vec<&Point<f64>> = points.iter().map(|p| p).collect();
        // Space definition
        let space = Space::new(dim, 0., width, 0.1);

        let mut mesh = SpaceMesh::new(&space, refpoints);
        mesh.embed();
        mesh.summary();

        //
    } //end of test_uniform_random
} // end of mod test
