use fera::graph::prelude::*;
use fera::graph::choose::Choose;
use fera_array::Array;
use rand::Rng;

use std::rc::Rc;

use {EulerTourTree, FindOpStrategy, FindVertexStrategy, NddrOneTree, NddrOneTreeForest, ParentTree};

// This trait creates a uniform interface to make it easy to run the experiments
pub trait Tree<G: WithEdge>: Clone {
    fn new<R: Rng>(g: Rc<G>, edges: &[Edge<G>], rng: R) -> Self;

    fn change_parent<R: Rng>(&mut self, rng: R) -> (Edge<G>, Edge<G>);

    fn change_any<R: Rng>(&mut self, rng: R) -> (Edge<G>, Edge<G>);

    fn graph(&self) -> &Rc<G>;
}

#[derive(Clone)]
pub struct NddrAdjTree<G>(NddrOneTreeForest<G>)
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
    fn new<R: Rng>(g: Rc<G>, edges: &[Edge<G>], rng: R) -> Self {
        let nddr = NddrOneTreeForest::new_with_strategies(
            g,
            edges.to_vec(),
            FindOpStrategy::Adj,
            FindVertexStrategy::Map,
            rng,
        );
        NddrAdjTree(nddr)
    }

    fn change_parent<R: Rng>(&mut self, rng: R) -> (Edge<G>, Edge<G>) {
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
pub struct NddrBalancedTree<G>(NddrOneTreeForest<G>)
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
    fn new<R: Rng>(g: Rc<G>, edges: &[Edge<G>], rng: R) -> Self {
        let nddr = NddrOneTreeForest::new_with_strategies(
            g,
            edges.to_vec(),
            FindOpStrategy::Balanced,
            FindVertexStrategy::Map,
            rng,
        );
        NddrBalancedTree(nddr)
    }

    fn change_parent<R: Rng>(&mut self, rng: R) -> (Edge<G>, Edge<G>) {
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

    fn change_parent<R: Rng>(&mut self, rng: R) -> (Edge<G>, Edge<G>) {
        NddrOneTree::op1(self, rng)
    }

    fn change_any<R: Rng>(&mut self, rng: R) -> (Edge<G>, Edge<G>) {
        NddrOneTree::op2(self, rng)
    }

    fn graph(&self) -> &Rc<G> {
        NddrOneTree::graph(self)
    }
}

impl<G, A> Tree<G> for ParentTree<G, A>
where
    G: Graph + WithVertexIndexProp + Choose,
    A: Clone + Array<OptionEdge<G>>,
{
    fn new<R: Rng>(g: Rc<G>, edges: &[Edge<G>], _rng: R) -> Self {
        ParentTree::from_iter(g, edges.iter().cloned())
    }

    fn change_parent<R: Rng>(&mut self, rng: R) -> (Edge<G>, Edge<G>) {
        let (ins, rem) = ParentTree::change_parent(self, rng);
        // FIXME: change Tree::parent signature
        (ins, rem.unwrap())
    }

    fn change_any<R: Rng>(&mut self, rng: R) -> (Edge<G>, Edge<G>) {
        let buffer = self.buffer();
        let mut buffer = buffer.borrow_mut();
        ParentTree::change_any(self, &mut *buffer, rng)
    }

    fn graph(&self) -> &Rc<G> {
        ParentTree::graph(self)
    }
}

impl<G> Tree<G> for EulerTourTree<G>
where
    G: IncidenceGraph + Choose + WithVertexIndexProp + Clone,
{
    fn new<R: Rng>(g: Rc<G>, edges: &[Edge<G>], _rng: R) -> Self {
        EulerTourTree::new(g, edges)
    }

    fn change_parent<R: Rng>(&mut self, rng: R) -> (Edge<G>, Edge<G>) {
        EulerTourTree::change_parent(self, rng)
    }

    fn change_any<R: Rng>(&mut self, rng: R) -> (Edge<G>, Edge<G>) {
        EulerTourTree::change_any(self, rng)
    }

    fn graph(&self) -> &Rc<G> {
        EulerTourTree::graph(self)
    }
}
