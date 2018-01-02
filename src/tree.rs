use fera::graph::prelude::*;
use fera_array::Array;
use rand::Rng;

use std::cell::RefCell;

use {EulerTourTree, FindOpStrategy, FindVertexStrategy, NddrOneTree, NddrOneTreeForest, ParentTree};

// This trait creates a uniform interface to make it easy to run the experiments
pub trait Tree: Clone {
    fn new<R: Rng>(g: CompleteGraph, edges: &[Edge<CompleteGraph>], rng: R) -> Self;

    fn op1<R: Rng>(&mut self, rng: R);

    fn op2<R: Rng>(&mut self, rng: R);
}

#[derive(Clone)]
pub struct NddrAdjTree(NddrOneTreeForest<CompleteGraph>);

impl Tree for NddrAdjTree {
    fn new<R: Rng>(g: CompleteGraph, edges: &[Edge<CompleteGraph>], rng: R) -> Self {
        let nddr = NddrOneTreeForest::new_with_strategies(
            g,
            edges.to_vec(),
            FindOpStrategy::Adj,
            FindVertexStrategy::Map,
            rng,
        );
        NddrAdjTree(nddr)
    }

    fn op1<R: Rng>(&mut self, mut rng: R) {
        // TODO: why &mut?
        self.0.op1(&mut rng);
    }

    fn op2<R: Rng>(&mut self, mut rng: R) {
        self.0.op2(&mut rng);
    }
}

#[derive(Clone)]
pub struct NddrBalancedTree(NddrOneTreeForest<CompleteGraph>);

impl Tree for NddrBalancedTree {
    fn new<R: Rng>(g: CompleteGraph, edges: &[Edge<CompleteGraph>], rng: R) -> Self {
        let nddr = NddrOneTreeForest::new_with_strategies(
            g,
            edges.to_vec(),
            FindOpStrategy::Balanced,
            FindVertexStrategy::Map,
            rng,
        );
        NddrBalancedTree(nddr)
    }

    fn op1<R: Rng>(&mut self, mut rng: R) {
        self.0.op1(&mut rng);
    }

    fn op2<R: Rng>(&mut self, mut rng: R) {
        self.0.op2(&mut rng);
    }
}

impl Tree for NddrOneTree<CompleteGraph> {
    fn new<R: Rng>(g: CompleteGraph, edges: &[Edge<CompleteGraph>], _rng: R) -> Self {
        NddrOneTree::new(g, edges)
    }

    fn op1<R: Rng>(&mut self, rng: R) {
        NddrOneTree::op1(self, rng);
    }

    fn op2<R: Rng>(&mut self, rng: R) {
        NddrOneTree::op2(self, rng);
    }
}

impl<A> Tree for ParentTree<CompleteGraph, A>
where
    A: Clone + Array<OptionEdge<CompleteGraph>>,
{
    fn new<R: Rng>(g: CompleteGraph, edges: &[Edge<CompleteGraph>], _rng: R) -> Self {
        ParentTree::from_iter(g, edges.iter().cloned())
    }

    fn op1<R: Rng>(&mut self, rng: R) {
        self.change_parent(rng);
    }

    fn op2<R: Rng>(&mut self, rng: R) {
        thread_local! { static BUFFER: RefCell<Vec<Edge<CompleteGraph>>> = RefCell::new(vec![]); }

        BUFFER.with(|buffer| {
            self.change_any(&mut *buffer.borrow_mut(), rng);
        });
    }
}

impl Tree for EulerTourTree<CompleteGraph> {
    fn new<R: Rng>(g: CompleteGraph, edges: &[Edge<CompleteGraph>], _rng: R) -> Self {
        EulerTourTree::new(g, edges)
    }

    fn op1<R: Rng>(&mut self, rng: R) {
        self.change_parent(rng);
    }

    fn op2<R: Rng>(&mut self, _rng: R) {
        unimplemented!()
    }
}
