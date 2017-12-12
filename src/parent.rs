use fera::graph::prelude::*;
use fera::graph::algs::Trees;
use fera::graph::choose::Choose;
use fera::graph::traverse::{Dfs, OnDiscoverTreeEdge};
use rand::Rng;

// This also works with forests, maybe we should change the name.
pub struct ParentTree<'a, G: 'a + Graph + WithVertexIndexProp> {
    g: &'a G,
    index: VertexIndexProp<G>,
    parent: Vec<OptionEdge<G>>,
}

impl<'a, G> ParentTree<'a, G>
where
    G: Graph + WithVertexIndexProp,
{
    pub fn from_iter<I>(g: &'a G, edges: I) -> Self
    where
        I: IntoIterator<Item = Edge<G>>,
    {
        let index = g.vertex_index();

        let mut parent = vec![G::edge_none(); g.num_vertices()];
        let sub = g.spanning_subgraph(edges);
        let r = sub.edges().next().map(|e| g.source(e));
        sub.dfs(OnDiscoverTreeEdge(|e| {
            parent[index.get(g.target(e))] = g.reverse(e).into();
        })).roots(r)
            .run();

        ParentTree { g, index, parent }
    }

    pub fn add_edge_change_parent(&mut self, e: Edge<G>) -> Result<Option<Edge<G>>, ()> {
        let (u, v) = self.g.ends(e);
        if self.contains(e) {
            Err(())
        } else if self.is_ancestor_of(u, v) {
            Ok(self.set_parent(v, self.g.reverse(e)))
        } else {
            Ok(self.set_parent(u, e))
        }
    }

    pub fn parent(&self, v: Vertex<G>) -> Option<Edge<G>> {
        self.parent[self.index.get(v)].into_option()
    }

    pub fn is_ancestor_of(&self, ancestor: Vertex<G>, u: Vertex<G>) -> bool {
        let mut v = u;
        while let Some(e) = self.parent(v) {
            v = self.g.target(e);
            if v == ancestor {
                return true;
            }
            assert_ne!(v, u, "cycle detected");
        }
        false
    }

    fn set_parent(&mut self, v: Vertex<G>, e: Edge<G>) -> Option<Edge<G>> {
        debug_assert_eq!(v, self.g.source(e));
        let i = self.index.get(v);
        let old = self.parent[i].into_option();
        self.parent[i] = e.into();
        old
    }

    fn num_edges(&self) -> usize {
        self.g.vertices().filter_map(|v| self.parent(v)).count()
    }

    fn contains(&self, e: Edge<G>) -> bool {
        let (u, v) = self.g.ends(e);
        self.parent(u) == Some(e) || self.parent(v) == Some(e)
    }

    pub fn check(&self) {
        assert!(
            self.g
                .spanning_subgraph(self.g.vertices().filter_map(|v| self.parent(v)))
                .is_tree()
        )
    }
}

impl<'a, G> ParentTree<'a, G>
where
    G: Graph + Choose + WithVertexIndexProp,
{
    pub fn change_parent<R: Rng>(&mut self, mut rng: R) -> (Edge<G>, Option<Edge<G>>) {
        loop {
            let ins = self.g.choose_edge(&mut rng).unwrap();
            if let Ok(rem) = self.add_edge_change_parent(ins) {
                return (ins, rem);
            }
        }
    }
}

impl<'a, G> Clone for ParentTree<'a, G>
where
    G: Graph + WithVertexIndexProp,
{
    fn clone(&self) -> Self {
        Self {
            g: self.g,
            index: self.g.vertex_index(),
            parent: self.parent.clone(),
        }
    }

    fn clone_from(&mut self, other: &Self) {
        self.g = other.g;
        self.index = other.g.vertex_index();
        self.parent.clone_from(&other.parent);
    }
}

impl<'a, G> PartialEq for ParentTree<'a, G>
where
    G: Graph + WithVertexIndexProp,
{
    fn eq(&self, other: &Self) -> bool {
        self.num_edges() == other.num_edges() &&
            self.g.vertices().filter_map(|v| self.parent(v)).all(|e| {
                other.contains(e)
            })
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use fera::fun::vec;
    use rand::{self, Rng};

    fn graph_tree(n: u32) -> (CompleteGraph, Vec<Edge<CompleteGraph>>) {
        let mut rng = rand::weak_rng();
        let g = CompleteGraph::new(n);
        let tree = vec(
            StaticGraph::new_random_tree(n as usize, &mut rng)
                .edges_ends()
                .map(|(u, v)| g.edge_by_ends(u, v)),
        );

        (g, tree)
    }

    #[test]
    fn eq() {
        let mut rng = rand::weak_rng();
        let n = 30;
        let (g, mut tree) = graph_tree(n);
        let trees = vec((0..n).map(|_| {
            rng.shuffle(&mut *tree);
            ParentTree::from_iter(&g, tree.clone())
        }));
        for i in 0..(n as usize) {
            for j in 0..(n as usize) {
                assert!(trees[i] == trees[j]);
            }
        }
    }

    #[test]
    fn change_parent() {
        let mut rng = rand::weak_rng();
        let n = 30;
        let (g, tree) = graph_tree(n);
        let mut tree = ParentTree::from_iter(&g, tree);
        for _ in 0..1000 {
            let mut new = tree.clone();
            new.change_parent(&mut rng);
            new.check();
            assert!(tree != new);
            tree = new;
        }
    }
}
