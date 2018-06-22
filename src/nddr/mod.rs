// This module implements the NDDR data structure described in the paper "Efficient Forest Data
// Structure for Evolutionary Algorithms Applied to Network Design", Alexandre C. B. Delbem, Telma
// W. de Lima and Guilherme P. Telles.
//
// FIXME: Cannot find optimal solution for OTMP using NddrOneTreeForest...

mod collect;
mod ndd;
mod one_tree;
mod one_tree_forest;

pub use self::collect::*;
pub use self::ndd::*;
pub use self::one_tree::*;
pub use self::one_tree_forest::*;
