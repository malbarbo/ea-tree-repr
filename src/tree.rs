use fera::graph::prelude::*;
use fera::graph::choose::Choose;
use fera_array::Array;
use rand::{Rng, XorShiftRng};

use std::rc::Rc;

use {EulerTourTree, FindOpStrategy, FindVertexStrategy, NddrOneTree, NddrOneTreeForest,
     PredecessorTree};

// This trait creates a uniform interface to make it easy to run the experiments
pub trait Tree<G: WithEdge>: Clone {
    fn new<R: Rng>(g: Rc<G>, edges: &[Edge<G>], rng: R) -> Self;

    fn set_edges(&mut self, edges: &[Edge<G>]);

    fn change_pred<R: Rng>(&mut self, rng: R) -> (Edge<G>, Edge<G>);

    fn change_any<R: Rng>(&mut self, rng: R) -> (Edge<G>, Edge<G>);

    fn graph(&self) -> &Rc<G>;
}

#[derive(Clone)]
pub struct NddrAdjTree<G>(NddrOneTreeForest<G>, XorShiftRng)
where
    G: AdjacencyGraph
        + WithVertexIndexProp
        + WithVertexProp<DefaultVertexPropMut<G, bool>>
        + Clone
        + Choose,
    DefaultVertexPropMut<G, bool>: Clone;

impl<G> Tree<G> for NddrAdjTree<G>
where
    G: AdjacencyGraph
        + WithVertexIndexProp
        + WithVertexProp<DefaultVertexPropMut<G, bool>>
        + Clone
        + Choose,
    DefaultVertexPropMut<G, bool>: Clone,
{
    fn new<R: Rng>(g: Rc<G>, edges: &[Edge<G>], mut rng: R) -> Self {
        let xrng = rng.gen();
        let nddr = NddrOneTreeForest::new_with_strategies(
            g,
            edges.to_vec(),
            FindOpStrategy::Adj,
            FindVertexStrategy::FatNode,
            rng,
        );
        NddrAdjTree(nddr, xrng)
    }

    fn set_edges(&mut self, edges: &[Edge<G>]) {
        self.0.set_edges(edges, &mut self.1);
    }

    fn change_pred<R: Rng>(&mut self, rng: R) -> (Edge<G>, Edge<G>) {
        self.0.op1(rng)
    }

    fn change_any<R: Rng>(&mut self, rng: R) -> (Edge<G>, Edge<G>) {
        self.0.op2(rng)
    }

    fn graph(&self) -> &Rc<G> {
        self.0.graph()
    }
}

#[derive(Clone)]
pub struct NddrBalancedTree<G>(NddrOneTreeForest<G>, XorShiftRng)
where
    G: AdjacencyGraph
        + WithVertexIndexProp
        + WithVertexProp<DefaultVertexPropMut<G, bool>>
        + Clone
        + Choose,
    DefaultVertexPropMut<G, bool>: Clone;

impl<G> Tree<G> for NddrBalancedTree<G>
where
    G: AdjacencyGraph
        + WithVertexIndexProp
        + WithVertexProp<DefaultVertexPropMut<G, bool>>
        + Clone
        + Choose,
    DefaultVertexPropMut<G, bool>: Clone,
{
    fn new<R: Rng>(g: Rc<G>, edges: &[Edge<G>], mut rng: R) -> Self {
        let xrng = rng.gen();
        let nddr = NddrOneTreeForest::new_with_strategies(
            g,
            edges.to_vec(),
            FindOpStrategy::Balanced,
            FindVertexStrategy::FatNode,
            rng,
        );
        NddrBalancedTree(nddr, xrng)
    }

    fn set_edges(&mut self, edges: &[Edge<G>]) {
        self.0.set_edges(edges, &mut self.1);
    }

    fn change_pred<R: Rng>(&mut self, rng: R) -> (Edge<G>, Edge<G>) {
        self.0.op1(rng)
    }

    fn change_any<R: Rng>(&mut self, rng: R) -> (Edge<G>, Edge<G>) {
        self.0.op2(rng)
    }

    fn graph(&self) -> &Rc<G> {
        self.0.graph()
    }
}

impl<G> Tree<G> for NddrOneTree<G>
where
    G: AdjacencyGraph + Choose,
{
    fn new<R: Rng>(g: Rc<G>, edges: &[Edge<G>], _rng: R) -> Self {
        NddrOneTree::new(g, edges)
    }

    fn set_edges(&mut self, _edges: &[Edge<G>]) {
        unimplemented!()
    }

    fn change_pred<R: Rng>(&mut self, rng: R) -> (Edge<G>, Edge<G>) {
        NddrOneTree::op1(self, rng)
    }

    fn change_any<R: Rng>(&mut self, rng: R) -> (Edge<G>, Edge<G>) {
        NddrOneTree::op2(self, rng)
    }

    fn graph(&self) -> &Rc<G> {
        NddrOneTree::graph(self)
    }
}

impl<G, A> Tree<G> for PredecessorTree<G, A>
where
    G: Graph + WithVertexIndexProp + Choose,
    A: Clone + Array<OptionEdge<G>>,
{
    fn new<R: Rng>(g: Rc<G>, edges: &[Edge<G>], _rng: R) -> Self {
        PredecessorTree::from_iter(g, edges.iter().cloned())
    }

    fn set_edges(&mut self, edges: &[Edge<G>]) {
        PredecessorTree::set_edges(self, edges.iter().cloned());
    }

    fn change_pred<R: Rng>(&mut self, rng: R) -> (Edge<G>, Edge<G>) {
        let (ins, rem) = PredecessorTree::change_pred(self, rng);
        // FIXME: change Tree::pred signature
        (ins, rem.unwrap())
    }

    fn change_any<R: Rng>(&mut self, rng: R) -> (Edge<G>, Edge<G>) {
        let buffer = self.buffer();
        let mut buffer = buffer.borrow_mut();
        PredecessorTree::change_any(self, &mut *buffer, rng)
    }

    fn graph(&self) -> &Rc<G> {
        PredecessorTree::graph(self)
    }
}

impl<G> Tree<G> for EulerTourTree<G>
where
    G: IncidenceGraph + Choose + WithVertexIndexProp + Clone,
{
    fn new<R: Rng>(g: Rc<G>, edges: &[Edge<G>], _rng: R) -> Self {
        EulerTourTree::new(g, edges)
    }

    fn set_edges(&mut self, edges: &[Edge<G>]) {
        EulerTourTree::set_edges(self, edges);
    }

    fn change_pred<R: Rng>(&mut self, rng: R) -> (Edge<G>, Edge<G>) {
        EulerTourTree::change_pred(self, rng)
    }

    fn change_any<R: Rng>(&mut self, rng: R) -> (Edge<G>, Edge<G>) {
        EulerTourTree::change_any(self, rng)
    }

    fn graph(&self) -> &Rc<G> {
        EulerTourTree::graph(self)
    }
}
