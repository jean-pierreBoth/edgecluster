//! This module implement a restricted hierarchicall well separated tree.
//!
//! The genesis of this algorithm relies on:
//!  - Probabilistic approximation of metric spaces and its algorithmic applications Bartal Y.  1996.
//!  - Algorithms for dynamic geometric problems over data streams. Indyk 2004.

pub mod cluster;
pub mod point;
pub mod rhst2;

pub use cluster::*;
pub use point::*;
