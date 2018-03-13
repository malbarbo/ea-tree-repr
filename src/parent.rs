use fera::graph::prelude::*;
use fera::graph::choose::Choose;
use fera::graph::traverse::{Dfs, OnDiscoverTreeEdge};
use fera_array::{Array, CowNestedArray, CowNestedNestedArray, VecArray};
use rand::Rng;

use std::cell::RefCell;
use std::rc::Rc;
use std::mem;

pub type Parent2Tree<G> = ParentTree<G, CowNestedArray<OptionEdge<G>>>;
pub type Parent3Tree<G> = ParentTree<G, CowNestedNestedArray<OptionEdge<G>>>;

// This also works with forests, maybe we should change the name.
pub struct ParentTree<G, A = VecArray<OptionEdge<G>>>
where
    G: Graph + WithVertexIndexProp,
    A: Array<OptionEdge<G>>,
{
    g: Rc<G>,
    index: VertexIndexProp<G>,
    parent: A,
    buffer: Rc<RefCell<Vec<Edge<G>>>>,
}

impl<G, A> ParentTree<G, A>
where
    G: Graph + WithVertexIndexProp,
    A: Array<OptionEdge<G>>,
{
    pub fn from_iter<I>(g: Rc<G>, edges: I) -> Self
    where
        I: IntoIterator<Item = Edge<G>>,
    {
        let mut tree = ParentTree {
            g: g.clone(),
            index: g.vertex_index(),
            parent: A::with_value(G::edge_none(), g.num_vertices()),
            buffer: Rc::new(RefCell::new(vec![])),
        };

        tree.set_edges(edges);
        tree
    }

    pub fn set_edges<I>(&mut self, edges: I)
    where
        I: IntoIterator<Item = Edge<G>>,
    {
        let g = &self.g;
        let index = g.vertex_index();
        let parent = &mut self.parent;
        let sub = g.spanning_subgraph(edges);
        let r = sub.edges().next().map(|e| g.source(e));
        sub.dfs(OnDiscoverTreeEdge(|e| {
            parent[index.get(g.target(e))] = g.reverse(e).into();
        })).roots(r)
            .run();
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
            let e = self.g.reverse(e);
            self.set_parent(v, e);
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
            let t = self.g.target(e);
            let e = self.g.reverse(e);
            parent = self.set_parent(t, e);
        }
    }

    pub fn path_to_root(&self, v: Vertex<G>) -> PathToRoot<G, A> {
        PathToRoot { tree: self, cur: v }
    }

    pub fn choose_path_edge<R: Rng>(
        &self,
        u: Vertex<G>,
        v: Vertex<G>,
        buffer: &mut Vec<Edge<G>>,
        mut rng: R,
    ) -> Edge<G> {
        assert_ne!(u, v);
        buffer.clear();

        for e in self.path_to_root(u) {
            buffer.push(e);
            if self.g.target(e) == v {
                return *rng.choose(buffer).unwrap();
            }
        }
        let s = buffer.len();

        for e in self.path_to_root(v) {
            buffer.push(e);
            if self.g.target(e) == u {
                return *rng.choose(&buffer[s..]).unwrap();
            }
        }

        let mut i = s - 1;
        let mut j = buffer.len() - 1;
        while buffer[i] == buffer[j] {
            i -= 1;
            j -= 1;
        }
        i += 1;
        j += 1;

        let total = (0..i).len() + (s..j).len();
        let x = rng.gen_range(0, total);
        if x < i {
            buffer[x]
        } else {
            buffer[s..j][x - i]
        }
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

    pub fn graph(&self) -> &Rc<G> {
        &self.g
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

impl<G, A> ParentTree<G, A>
where
    G: Graph + Choose + WithVertexIndexProp,
    A: Array<OptionEdge<G>>,
{
    pub fn change_parent<R: Rng>(&mut self, mut rng: R) -> (Edge<G>, Option<Edge<G>>) {
        let ins = self.choose_nontree_edge(&mut rng);
        let (u, v) = self.g.ends(ins);
        if self.is_ancestor_of(u, v) {
            let ins = self.g.reverse(ins);
            (ins, self.set_parent(v, ins))
        } else if self.is_ancestor_of(v, u) {
            (ins, self.set_parent(u, ins))
        // u is not ancestor of v and v is not ancestor of u, so we randomly choose which
        // parent we change
        } else if rng.gen() {
            let ins = self.g.reverse(ins);
            (ins, self.set_parent(v, ins))
        } else {
            (ins, self.set_parent(u, ins))
        }
    }

    pub fn change_any<R: Rng>(
        &mut self,
        buffer: &mut Vec<Edge<G>>,
        mut rng: R,
    ) -> (Edge<G>, Edge<G>) {
        // TODO: make this works with forests
        let ins = self.choose_nontree_edge(&mut rng);
        let (u, v) = self.g.ends(ins);
        // using choose_path_edge is a bit faster than find_path
        let rem = self.choose_path_edge(u, v, buffer, &mut rng);
        assert!(self.cut(rem));
        if self.is_root(u) {
            self.set_parent(u, ins);
        } else {
            self.make_root(v);
            let ins = self.g.reverse(ins);
            self.set_parent(v, ins);
        }
        (ins, rem)
    }

    fn choose_nontree_edge<R: Rng>(&self, mut rng: R) -> Edge<G> {
        self.g
            .choose_edge_iter(&mut rng)
            .find(|e| !self.contains(*e))
            .unwrap()
    }

    pub fn buffer(&self) -> Rc<RefCell<Vec<Edge<G>>>> {
        self.buffer.clone()
    }
}

impl<G, A> Clone for ParentTree<G, A>
where
    G: Graph + WithVertexIndexProp,
    A: Array<OptionEdge<G>> + Clone,
{
    fn clone(&self) -> Self {
        Self {
            g: Rc::clone(&self.g),
            index: self.g.vertex_index(),
            parent: self.parent.clone(),
            buffer: self.buffer.clone(),
        }
    }

    fn clone_from(&mut self, other: &Self) {
        self.g.clone_from(&other.g);
        self.index = other.g.vertex_index();
        self.parent.clone_from(&other.parent);
        self.buffer.clone_from(&other.buffer);
    }
}

impl<G, A> PartialEq for ParentTree<G, A>
where
    G: Graph + WithVertexIndexProp,
    A: Array<OptionEdge<G>>,
{
    fn eq(&self, other: &Self) -> bool {
        self.num_edges() == other.num_edges()
            && self.g
                .vertices()
                .filter_map(|v| self.parent(v))
                .all(|e| other.contains(e))
    }
}

pub struct PathToRoot<'a, G, A>
where
    G: 'a + Graph + WithVertexIndexProp,
    A: 'a + Array<OptionEdge<G>>,
{
    tree: &'a ParentTree<G, A>,
    cur: Vertex<G>,
}

impl<'a, G, A> Iterator for PathToRoot<'a, G, A>
where
    G: Graph + WithVertexIndexProp,
    A: Array<OptionEdge<G>>,
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
    use fera_array::Array;
    use rand::{self, Rng};
    use random_sp;

    macro_rules! def_tests {
        ($m:ident, $t:ty, $($name:ident),+) => (
            mod $m {
                use fera_array::*;
                $(
                    #[test]
                    fn $name() {
                        super::$name::<$t>();
                    }
                )*
            }
        )
    }

    def_tests!(
        parent,
        VecArray<_>,
        eq,
        change_parent,
        change_any,
        make_root,
        find_root,
        paths
    );

    def_tests!(
        parent2,
        CowNestedArray<_>,
        eq,
        change_parent,
        change_any,
        make_root,
        find_root,
        paths
    );

    def_tests!(
        parent3,
        CowNestedNestedArray<_>,
        eq,
        change_parent,
        change_any,
        make_root,
        find_root,
        paths
    );

    fn graph_tree(n: u32) -> (Rc<CompleteGraph>, Vec<Edge<CompleteGraph>>) {
        let g = CompleteGraph::new(n);
        let tree = random_sp(&g, rand::weak_rng());
        (Rc::new(g), tree)
    }

    fn eq<A: Array<OptionEdge<CompleteGraph>>>() {
        let mut rng = rand::weak_rng();
        let n = 30;
        let (g, mut tree) = graph_tree(n);
        let trees: Vec<ParentTree<CompleteGraph, A>> = vec((0..n).map(|_| {
            rng.shuffle(&mut *tree);
            ParentTree::from_iter(g.clone(), tree.clone())
        }));
        for i in 0..(n as usize) {
            for j in 0..(n as usize) {
                assert!(trees[i] == trees[j]);
            }
        }
    }

    fn change_parent<A: Clone + Array<OptionEdge<CompleteGraph>>>() {
        let mut rng = rand::weak_rng();
        let n = 30;
        let (g, tree) = graph_tree(n);
        let mut tree: ParentTree<CompleteGraph, A> = ParentTree::from_iter(g, tree);
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

    fn change_any<A: Clone + Array<OptionEdge<CompleteGraph>>>() {
        let mut rng = rand::weak_rng();
        let mut buffer = vec![];
        let n = 30;
        let (g, tree) = graph_tree(n);
        let mut tree: ParentTree<CompleteGraph, A> = ParentTree::from_iter(g, tree);
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

    fn make_root<A: Clone + Array<OptionEdge<CompleteGraph>>>() {
        let mut rng = rand::weak_rng();
        let n = 30;
        let (g, tree) = graph_tree(n);
        let mut tree: ParentTree<CompleteGraph, A> = ParentTree::from_iter(g.clone(), tree);
        for u in g.choose_vertex_iter(&mut rng).take(1000) {
            let mut new = tree.clone();
            new.make_root(u);
            assert!(new.is_root(u));
            new.check();
            assert!(tree == new);
            tree = new;
        }
    }

    fn find_root<A: Array<OptionEdge<CompleteGraph>>>() {
        let n = 30;
        let (g, tree) = graph_tree(n);
        let r = g.source(tree[0]);
        let tree: ParentTree<CompleteGraph, A> = ParentTree::from_iter(g.clone(), tree);
        for v in g.vertices() {
            assert_eq!(r == v, tree.is_root(v));
            assert_eq!(r, tree.find_root(v));
        }
    }

    fn paths<A: Array<OptionEdge<CompleteGraph>>>() {
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
        let g = Rc::new(CompleteGraph::new(n));
        let e = |u, v| g.edge_by_ends(u, v);

        let tree: ParentTree<CompleteGraph, A> = ParentTree::from_iter(
            g.clone(),
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
