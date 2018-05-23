use fera::graph::prelude::*;
use rand::{Rng, SeedableRng, XorShiftRng};

pub fn new_rng(seed: u32) -> XorShiftRng {
    XorShiftRng::from_seed([seed, seed, seed, seed]).gen()
}

pub fn random_sp<R: Rng>(g: &CompleteGraph, rng: R) -> Vec<Edge<CompleteGraph>> {
    StaticGraph::new_random_tree(g.num_vertices(), rng)
        .edges_ends()
        .map(|(u, v)| g.edge_by_ends(u, v))
        .collect()
}

pub fn random_sp_with_diameter<R: Rng>(
    g: &CompleteGraph,
    d: usize,
    rng: R,
) -> Vec<Edge<CompleteGraph>> {
    StaticGraph::new_random_tree_with_diameter(g.num_vertices() as u32, d as u32, rng)
        .unwrap()
        .edges_ends()
        .map(|(u, v)| g.edge_by_ends(u, v))
        .collect()
}
