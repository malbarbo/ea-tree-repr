use fera::graph::prelude::*;
use fera::graph::choose::Choose;
use fera::graph::traverse::{Dfs, OnDiscoverTreeEdge};
use rand::Rng;

use std::mem;

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
    pub fn new(g: &'a G) -> Self {
        ParentTree {
            g,
            index: g.vertex_index(),
            parent: vec![G::edge_none(); g.num_vertices()],
        }
    }

    pub fn from_iter<I>(g: &'a G, edges: I) -> Self
    where
        I: IntoIterator<Item = Edge<G>>,
    {
        let index = g.vertex_index();
        let sub = g.spanning_subgraph(edges);
        let r = sub.edges().next().map(|e| g.source(e));
        let mut parent = vec![G::edge_none(); g.num_vertices()];
        sub.dfs(OnDiscoverTreeEdge(|e| {
            parent[index.get(g.target(e))] = g.reverse(e).into();
        })).roots(r)
            .run();

        ParentTree { g, index, parent }
    }

    pub fn contains(&self, e: Edge<G>) -> bool {
        let (u, v) = self.g.ends(e);
        self.parent(u) == Some(e) || self.parent(v) == Some(e)
    }

    pub fn is_connected(&self, u: Vertex<G>, v: Vertex<G>) -> bool {
        self.find_root(u) == self.find_root(v)
    }

    pub fn link(&mut self, e: Edge<G>) -> bool {
        let (u, v) = self.g.ends(e);
        if self.is_connected(u, v) {
            false
        } else if self.is_root(u) {
            self.set_parent(u, e);
            true
        } else {
            self.make_root(v);
            self.set_parent(v, self.g.reverse(e));
            true
        }
    }

    pub fn cut(&mut self, e: Edge<G>) -> bool {
        let (u, v) = self.g.ends(e);
        if self.parent(u) == Some(e) {
            self.cut_parent(u);
            true
        } else if self.parent(v) == Some(e) {
            self.cut_parent(v);
            true
        } else {
            false
        }
    }

    pub fn cut_parent(&mut self, v: Vertex<G>) -> Option<Edge<G>> {
        mem::replace(&mut self.parent[self.index.get(v)], G::edge_none()).into_option()
    }

    pub fn parent(&self, v: Vertex<G>) -> Option<Edge<G>> {
        self.parent[self.index.get(v)].into_option()
    }

    pub fn find_root(&self, v: Vertex<G>) -> Vertex<G> {
        self.path_to_root(v)
            .last()
            .map(|e| self.g.target(e))
            .unwrap_or(v)
    }

    pub fn is_root(&self, u: Vertex<G>) -> bool {
        self.parent(u).is_none()
    }

    pub fn make_root(&mut self, u: Vertex<G>) {
        let mut parent = self.cut_parent(u);
        while let Some(e) = parent {
            parent = self.set_parent(self.g.target(e), self.g.reverse(e));
        }
    }

    pub fn path_to_root(&self, v: Vertex<G>) -> PathToRoot<G> {
        PathToRoot { tree: self, cur: v }
    }

    pub fn find_path(&self, u: Vertex<G>, v: Vertex<G>, path: &mut Vec<Edge<G>>) {
        path.clear();

        if u == v {
            return;
        }

        for e in self.path_to_root(u) {
            path.push(e);
            if self.g.target(e) == v {
                return;
            }
        }
        let s = path.len();

        for e in self.path_to_root(v) {
            path.push(e);
            if self.g.target(e) == u {
                path.drain(..s);
                path.reverse();
                for e in &mut path[..] {
                    *e = self.g.reverse(*e);
                }
                return;
            }
        }

        let mut i = s - 1;
        let mut j = path.len() - 1;
        while path[i] == path[j] {
            i -= 1;
            j -= 1;
        }
        // remove the common path
        path.truncate(j + 1);
        path.drain((i + 1)..s);

        // reverse the path from v to the common parent
        path[(i + 1)..].reverse();
        for e in &mut path[(i + 1)..] {
            *e = self.g.reverse(*e);
        }
    }

    fn is_ancestor_of(&self, ancestor: Vertex<G>, u: Vertex<G>) -> bool {
        self.path_to_root(u).any(|e| self.g.target(e) == ancestor)
    }

    fn num_edges(&self) -> usize {
        self.g.vertices().filter_map(|v| self.parent(v)).count()
    }

    fn set_parent(&mut self, v: Vertex<G>, e: Edge<G>) -> Option<Edge<G>> {
        debug_assert_eq!(v, self.g.source(e));
        let i = self.index.get(v);
        let old = self.parent[i].into_option();
        self.parent[i] = e.into();
        old
    }

    #[cfg(test)]
    fn check(&self) {
        use fera::graph::algs::Trees;

        assert!(
            self.g
                .spanning_subgraph(self.g.vertices().filter_map(|v| self.parent(v)))
                .is_tree()
        );

        for v in self.g.vertices() {
            for _ in self.path_to_root(v) {}
        }
    }
}

impl<'a, G> ParentTree<'a, G>
where
    G: Graph + Choose + WithVertexIndexProp,
{
    pub fn change_parent<R: Rng>(&mut self, mut rng: R) -> (Edge<G>, Option<Edge<G>>) {
        for ins in self.g.choose_edge_iter(&mut rng) {
            let (u, v) = self.g.ends(ins);
            if self.contains(ins) {
                continue
            } else if self.is_ancestor_of(u, v) {
                return (ins, self.set_parent(v, self.g.reverse(ins)));
            } else {
                return (ins, self.set_parent(u, ins));
            }
        }
        unreachable!("cannot find an edge to insert");
    }

    pub fn change_any<R: Rng>(&mut self, buffer: &mut Vec<Edge<G>>, mut rng: R) -> (Edge<G>, Edge<G>) {
        // TODO: make this works with forests
        let ins = self.choose_nontree_edge(&mut rng);
        let (u, v) = self.g.ends(ins);
        self.find_path(u, v, buffer);
        let rem = *rng.choose(buffer).unwrap();
        assert!(self.cut(rem));
        assert!(self.link(ins));
        (ins, rem)
    }

    fn choose_nontree_edge<R: Rng>(&self, mut rng: R) -> Edge<G> {
        self.g.choose_edge_iter(&mut rng).filter(|e| !self.contains(*e)).next().unwrap()
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


pub struct PathToRoot<'a, G: 'a + Graph + WithVertexIndexProp> {
    tree: &'a ParentTree<'a, G>,
    cur: Vertex<G>,
}

impl<'a, G> Iterator for PathToRoot<'a, G>
where
    G: 'a + Graph + WithVertexIndexProp,
{
    type Item = Edge<G>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.tree.parent(self.cur).map(|e| {
            self.cur = self.tree.g.target(e);
            e
        })
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use fera::fun::vec;
    use fera::graph::algs::Paths;
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
            let (ins, rem) = new.change_parent(&mut rng);
            new.check();
            assert!(new.contains(ins));
            assert!(rem.map(|rem| !new.contains(rem)).unwrap_or(true));
            assert!(tree != new);
            tree = new;
        }
    }

    #[test]
    fn change_any() {
        let mut rng = rand::weak_rng();
        let mut buffer = vec![];
        let n = 30;
        let (g, tree) = graph_tree(n);
        let mut tree = ParentTree::from_iter(&g, tree);
        for _ in 0..1000 {
            let mut new = tree.clone();
            let (ins, rem) = new.change_any(&mut buffer, &mut rng);
            assert!(new.contains(ins));
            assert!(!new.contains(rem));
            new.check();
            assert!(tree != new);
            tree = new;
        }
    }

    #[test]
    fn make_root() {
        let mut rng = rand::weak_rng();
        let n = 30;
        let (g, tree) = graph_tree(n);
        let mut tree = ParentTree::from_iter(&g, tree);
        for u in g.choose_vertex_iter(&mut rng).take(1000) {
            let mut new = tree.clone();
            new.make_root(u);
            assert!(new.is_root(u));
            new.check();
            assert!(tree == new);
            tree = new;
        }
    }

    #[test]
    fn find_root() {
        let n = 30;
        let (g, tree) = graph_tree(n);
        let r = g.source(tree[0]);
        let tree = ParentTree::from_iter(&g, tree);
        for v in g.vertices() {
            assert_eq!(r == v, tree.is_root(v));
            assert_eq!(r, tree.find_root(v));
        }
    }

    #[test]
    fn paths() {
        //     0
        //    / \
        //   1   4
        //  / \   \
        // 2   3   5
        // |   |
        // 6   7
        //     |
        //     8
        let n = 9;
        let g = CompleteGraph::new(n);
        let e = |u, v| g.edge_by_ends(u, v);

        let tree = ParentTree::from_iter(
            &g,
            vec![
                e(0, 1),
                e(2, 1),
                e(3, 1),
                e(4, 0),
                e(5, 4),
                e(6, 2),
                e(7, 3),
                e(8, 7),
            ],
        );

        assert_eq!(vec![e(1, 0)], vec(tree.path_to_root(1)));
        assert_eq!(vec![e(2, 1), e(1, 0)], vec(tree.path_to_root(2)));
        assert_eq!(vec![e(3, 1), e(1, 0)], vec(tree.path_to_root(3)));
        assert_eq!(vec![e(4, 0)], vec(tree.path_to_root(4)));
        assert_eq!(vec![e(5, 4), e(4, 0)], vec(tree.path_to_root(5)));
        assert_eq!(vec![e(6, 2), e(2, 1), e(1, 0)], vec(tree.path_to_root(6)));
        assert_eq!(vec![e(7, 3), e(3, 1), e(1, 0)], vec(tree.path_to_root(7)));

        let mut path = vec![];
        tree.find_path(1, 0, &mut path);
        assert_eq!(vec![e(1, 0)], path);

        tree.find_path(0, 1, &mut path);
        assert_eq!(vec![e(0, 1)], path);

        tree.find_path(6, 1, &mut path);
        assert_eq!(vec![e(6, 2), e(2, 1)], path);

        tree.find_path(1, 6, &mut path);
        assert_eq!(vec![e(1, 2), e(2, 6)], path);

        tree.find_path(1, 4, &mut path);
        assert_eq!(vec![e(1, 0), e(0, 4)], path);

        tree.find_path(4, 1, &mut path);
        assert_eq!(vec![e(4, 0), e(0, 1)], path);

        tree.find_path(6, 8, &mut path);
        assert_eq!(vec![e(6, 2), e(2, 1), e(1, 3), e(3, 7), e(7, 8)], path);

        tree.find_path(7, 6, &mut path);
        assert_eq!(vec![e(7, 3), e(3, 1), e(1, 2), e(2, 6)], path);

        for u in 0..n {
            for v in 0..n {
                tree.find_path(u, v, &mut path);
                if u == v {
                    assert!(path.is_empty());
                } else {
                    assert!(
                        g.is_path(&path),
                        "{:?} -> {:?} = {:?}",
                        u,
                        v,
                        vec(g.ends(&path))
                    );
                    assert_eq!(u, g.source(*path.first().unwrap()));
                    assert_eq!(v, g.target(*path.last().unwrap()));
                }
            }
        }
    }
}
