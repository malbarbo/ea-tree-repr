// This module implements the NDDR data structure described in the paper "Efficient Forest Data
// Structure for Evolutionary Algorithms Applied to Network Design", Alexandre C. B. Delbem, Telma
// W. de Lima and Guilherme P. Telles.
//
// The following is missing to get the paper implementation
//     - Reset pi_* (PI_x) and history (L), so they do not get too large
//
// There is a lot of TODOs in the nddr implementation, some of them are aesthetics, others are
// small optimizations. I don't know if its worth to implement that...
//
// FIXME: Cannot find optimal solution for OTMP using NddrOneTreeForest...

mod collect;
mod ndd;
mod one_tree_forest;

pub use self::collect::*;
pub use self::ndd::*;
pub use self::one_tree_forest::*;
