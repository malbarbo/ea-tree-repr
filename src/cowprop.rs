use fera::graph::prelude::*;
use fera_array::{Array, CowNestedArray};

use std::ops::{Index, IndexMut};

// Based on graph::fera::prop::ArrayProp

pub type CowNestedArrayVertexProp<G, T> = CowNestedArrayProp<VertexIndexProp<G>, CowNestedArray<T>>;

#[derive(Clone, Debug)]
pub struct CowNestedArrayProp<P, D> {
    index: P,
    data: D,
}

impl<P, D> CowNestedArrayProp<P, D> {
    fn new(index: P, data: D) -> Self {
        Self { index, data }
    }
}

impl<I, P, D> PropGet<I> for CowNestedArrayProp<P, D>
where
    P: PropGet<I, Output = usize>,
    D: Index<usize>,
    D::Output: Clone + Sized,
{
    type Output = D::Output;

    #[inline(always)]
    fn get(&self, item: I) -> D::Output {
        self.data.index(self.index.get(item)).clone()
    }
}

impl<I, P, D> Index<I> for CowNestedArrayProp<P, D>
where
    P: PropGet<I, Output = usize>,
    D: Index<usize>,
{
    type Output = D::Output;

    #[inline(always)]
    fn index(&self, item: I) -> &Self::Output {
        self.data.index(self.index.get(item))
    }
}

impl<I, P, D> IndexMut<I> for CowNestedArrayProp<P, D>
where
    P: PropGet<I, Output = usize>,
    D: IndexMut<usize>,
{
    #[inline(always)]
    fn index_mut(&mut self, item: I) -> &mut Self::Output {
        self.data.index_mut(self.index.get(item))
    }
}

impl<T, G> VertexPropMutNew<G, T> for CowNestedArrayProp<VertexIndexProp<G>, CowNestedArray<T>>
where
    G: VertexList + WithVertexIndexProp,
    T: Clone,
{
    fn new_vertex_prop(g: &G, value: T) -> Self {
        CowNestedArrayProp::new(
            g.vertex_index(),
            CowNestedArray::with_value(value, g.num_vertices()),
        )
    }
}

impl<T, G> EdgePropMutNew<G, T> for CowNestedArrayProp<EdgeIndexProp<G>, CowNestedArray<T>>
where
    G: EdgeList + WithEdgeIndexProp,
    T: Clone,
{
    fn new_edge_prop(g: &G, value: T) -> Self {
        CowNestedArrayProp::new(
            g.edge_index(),
            CowNestedArray::with_value(value, g.num_edges()),
        )
    }
}
