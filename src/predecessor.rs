use fera::graph::choose::Choose;
use fera::graph::prelude::*;
use fera::graph::traverse::{Dfs, OnDiscoverTreeEdge};
use fera_array::{Array, CowNestedArray, CowNestedNestedArray, VecArray};
use rand::Rng;

use std::mem;
use std::rc::Rc;

pub type PredecessorTree2<G> = PredecessorTree<G, CowNestedArray<OptionVertex<G>>>;
pub type PredecessorTree3<G> = PredecessorTree<G, CowNestedNestedArray<OptionVertex<G>>>;

// This also works with forests, maybe we should change the name.
pub struct PredecessorTree<G, A = VecArray<OptionVertex<G>>>
where
    G: Graph + WithVertexIndexProp,
    A: Array<OptionVertex<G>>,
{
    g: Rc<G>,
    index: VertexIndexProp<G>,
    pred: A,
}

impl<G, A> PredecessorTree<G, A>
where
    G: Graph + WithVertexIndexProp,
    A: Array<OptionVertex<G>>,
{
    pub fn from_iter<I>(g: Rc<G>, edges: I) -> Self
    where
        I: IntoIterator<Item = Edge<G>>,
    {
        let n = g.num_vertices();
        let index = g.vertex_index();
        let mut tree = PredecessorTree {
            g,
            index,
            pred: A::with_value(G::vertex_none(), n),
        };

        tree.set_edges(edges);
        tree
    }

    pub fn set_edges<I>(&mut self, edges: I)
    where
        I: IntoIterator<Item = Edge<G>>,
    {
        let g = &self.g;
        let index = &self.index;
        let pred = &mut self.pred;
        for v in g.vertices() {
            pred[index.get(v)] = G::vertex_none();
        }
        let sub = g.spanning_subgraph(edges);
        let r = sub.edges().next().map(|e| g.source(e));
        sub.dfs(OnDiscoverTreeEdge(|e| {
            let (u, v) = g.end_vertices(e);
            pred[index.get(v)] = u.into();
        })).roots(r)
            .run();
    }

    pub fn graph(&self) -> &Rc<G> {
        &self.g
    }

    pub fn contains(&self, e: Edge<G>) -> bool {
        let (u, v) = self.g.ends(e);
        self.pred_vertex(u) == Some(v) || self.pred_vertex(v) == Some(u)
    }

    pub fn is_connected(&self, u: Vertex<G>, v: Vertex<G>) -> bool {
        self.find_root(u) == self.find_root(v)
    }

    pub fn link(&mut self, e: Edge<G>) -> bool {
        let (u, v) = self.g.ends(e);
        if self.is_connected(u, v) {
            false
        } else if self.is_root(u) {
            self.set_pred(u, v);
            true
        } else {
            self.make_root(v);
            self.set_pred(v, u);
            true
        }
    }

    pub fn cut(&mut self, e: Edge<G>) -> bool {
        let (u, v) = self.g.ends(e);
        if self.pred_vertex(u) == Some(v) {
            self.cut_pred(u);
            true
        } else if self.pred_vertex(v) == Some(u) {
            self.cut_pred(v);
            true
        } else {
            false
        }
    }

    pub fn cut_pred(&mut self, v: Vertex<G>) -> Option<Vertex<G>> {
        mem::replace(&mut self.pred[self.index.get(v)], G::vertex_none()).into_option()
    }

    fn set_pred(&mut self, v: Vertex<G>, p: Vertex<G>) -> Option<Vertex<G>> {
        mem::replace(&mut self.pred[self.index.get(v)], p.into()).into_option()
    }

    // FIXME: should not be pub (used by dc.rs)
    pub fn set_pred_edge(&mut self, v: Vertex<G>, p: Vertex<G>) -> Option<Edge<G>> {
        self.set_pred(v, p).map(|u| self.g.edge_by_ends(v, u))
    }

    pub fn pred_edge(&self, v: Vertex<G>) -> Option<Edge<G>> {
        self.pred_vertex(v).map(|u| self.g.edge_by_ends(v, u))
    }

    pub fn pred_vertex(&self, v: Vertex<G>) -> Option<Vertex<G>> {
        self.pred[self.index.get(v)].into_option()
    }

    pub fn find_root(&self, v: Vertex<G>) -> Vertex<G> {
        self.path_to_root(v).last().unwrap_or(v)
    }

    pub fn is_root(&self, u: Vertex<G>) -> bool {
        self.pred_vertex(u).is_none()
    }

    pub fn is_ancestor_of(&self, ancestor: Vertex<G>, u: Vertex<G>) -> bool {
        self.path_to_root(u).any(|p| p == ancestor)
    }

    pub fn make_root(&mut self, u: Vertex<G>) {
        let mut pred = self.cut_pred(u);
        let mut cur = u;
        while let Some(p) = pred {
            pred = self.set_pred(p, cur);
            cur = p;
        }
    }

    pub fn path_to_root(&self, v: Vertex<G>) -> PathToRoot<G, A> {
        PathToRoot { tree: self, cur: v }
    }

    pub fn find_path(&self, u: Vertex<G>, v: Vertex<G>, path: &mut Vec<Vertex<G>>) {
        path.clear();

        if u == v {
            return;
        }

        path.push(u);
        for p in self.path_to_root(u) {
            path.push(p);
            if p == v {
                return;
            }
        }
        let s = path.len();

        path.push(v);
        for p in self.path_to_root(v) {
            path.push(p);
            if p == u {
                path.drain(..s);
                path.reverse();
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
        path.truncate(j + 2);
        path.drain((i + 1)..s);

        // reverse the path from v to the common predecessor
        path[(i + 1)..].reverse();
    }

    fn num_edges(&self) -> usize {
        self.g
            .vertices()
            .filter_map(|v| self.pred_vertex(v))
            .count()
    }

    #[cfg(test)]
    fn edges_to_root(&self, v: Vertex<G>) -> Vec<Edge<G>> {
        let mut path = vec![v];
        path.extend(self.path_to_root(v));
        path.windows(2)
            .map(|s| self.g.edge_by_ends(s[0], s[1]))
            .collect()
    }

    #[cfg(test)]
    fn check(&self) {
        use fera::graph::algs::Trees;

        assert!(
            self.g
                .spanning_subgraph(self.g.vertices().filter_map(|v| self.pred_edge(v)))
                .is_tree()
        );

        for v in self.g.vertices() {
            for _ in self.path_to_root(v) {}
        }
    }
}

impl<G, A> PredecessorTree<G, A>
where
    G: Graph + Choose + WithVertexIndexProp,
    A: Array<OptionVertex<G>>,
{
    pub fn change_pred<R: Rng>(&mut self, mut rng: R) -> (Edge<G>, Option<Edge<G>>) {
        let ins = self.choose_nontree_edge(&mut rng);
        let (u, v) = self.g.ends(ins);
        let rem = if self.is_ancestor_of(u, v) {
            self.set_pred_edge(v, u)
        } else if self.is_ancestor_of(v, u) {
            self.set_pred_edge(u, v)
        // u is not ancestor of v and v is not ancestor of u, so we randomly choose which
        // pred we change
        } else if rng.gen() {
            self.set_pred_edge(v, u)
        } else {
            self.set_pred_edge(u, v)
        };
        (ins, rem)
    }

    pub fn change_any<R: Rng>(&mut self, mut rng: R) -> (Edge<G>, Edge<G>) {
        let ins = self.choose_nontree_edge(&mut rng);
        let rem = self.insert_remove(ins, TargetEdge::Any, rng);
        (ins, rem)
    }

    pub fn insert_remove<R: Rng>(&mut self, ins: Edge<G>, e: TargetEdge, rng: R) -> Edge<G> {
        let (u, v) = self.g.ends(ins);
        self.make_root(u);
        let x = match e {
            TargetEdge::First => v,
            TargetEdge::Last => {
                let mut x = v;
                let mut last = x;
                for w in self.path_to_root(v) {
                    x = last;
                    last = w;
                }
                x
            }
            TargetEdge::Any => self.choose_path_to_root(v, rng),
        };
        let rem = self.pred_edge(x).unwrap();
        self.set_pred(u, v);
        self.cut_pred(x);
        rem
    }

    fn choose_path_to_root<R: Rng>(&self, v: Vertex<G>, mut rng: R) -> Vertex<G> {
        let len = self.path_to_root(v).count();
        let i = rng.gen_range(0, len);
        if i == len - 1 {
            v
        } else {
            self.path_to_root(v).nth(i).unwrap()
        }
    }

    fn choose_nontree_edge<R: Rng>(&self, mut rng: R) -> Edge<G> {
        self.g
            .choose_edge_iter(&mut rng)
            .find(|e| !self.contains(*e))
            .unwrap()
    }
}

impl<G, A> Clone for PredecessorTree<G, A>
where
    G: Graph + WithVertexIndexProp,
    A: Array<OptionVertex<G>> + Clone,
{
    fn clone(&self) -> Self {
        Self {
            g: Rc::clone(&self.g),
            index: self.g.vertex_index(),
            pred: self.pred.clone(),
        }
    }

    fn clone_from(&mut self, other: &Self) {
        self.g.clone_from(&other.g);
        self.index = other.g.vertex_index();
        self.pred.clone_from(&other.pred);
    }
}

impl<G, A> PartialEq for PredecessorTree<G, A>
where
    G: Graph + WithVertexIndexProp,
    A: Array<OptionVertex<G>>,
{
    fn eq(&self, other: &Self) -> bool {
        self.num_edges() == other.num_edges()
            && self
                .g
                .vertices()
                .filter_map(|v| self.pred_edge(v))
                .all(|e| other.contains(e))
    }
}

pub struct PathToRoot<'a, G, A>
where
    G: 'a + Graph + WithVertexIndexProp,
    A: 'a + Array<OptionVertex<G>>,
{
    tree: &'a PredecessorTree<G, A>,
    cur: Vertex<G>,
}

impl<'a, G, A> Iterator for PathToRoot<'a, G, A>
where
    G: Graph + WithVertexIndexProp,
    A: Array<OptionVertex<G>>,
{
    type Item = Vertex<G>;

    fn next(&mut self) -> Option<Self::Item> {
        self.tree.pred_vertex(self.cur).map(|v| {
            self.cur = v;
            v
        })
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum TargetEdge {
    First,
    Last,
    Any,
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
        pred,
        VecArray<_>,
        eq,
        change_pred,
        change_any,
        make_root,
        find_root,
        paths
    );

    def_tests!(
        pred2,
        CowNestedArray<_>,
        eq,
        change_pred,
        change_any,
        make_root,
        find_root,
        paths
    );

    def_tests!(
        pred3,
        CowNestedNestedArray<_>,
        eq,
        change_pred,
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

    fn eq<A: Array<OptionVertex<CompleteGraph>>>() {
        let mut rng = rand::weak_rng();
        let n = 30;
        let (g, mut tree) = graph_tree(n);
        let trees: Vec<PredecessorTree<CompleteGraph, A>> = vec((0..n).map(|_| {
            rng.shuffle(&mut *tree);
            PredecessorTree::from_iter(g.clone(), tree.clone())
        }));
        for i in 0..(n as usize) {
            for j in 0..(n as usize) {
                assert!(trees[i] == trees[j]);
            }
        }
    }

    fn change_pred<A: Clone + Array<OptionVertex<CompleteGraph>>>() {
        let mut rng = rand::weak_rng();
        let n = 30;
        let (g, tree) = graph_tree(n);
        let mut tree: PredecessorTree<CompleteGraph, A> = PredecessorTree::from_iter(g, tree);
        for _ in 0..1000 {
            let mut new = tree.clone();
            let (ins, rem) = new.change_pred(&mut rng);
            new.check();
            assert!(new.contains(ins));
            assert!(rem.map(|rem| !new.contains(rem)).unwrap_or(true));
            assert!(tree != new);
            tree = new;
        }
    }

    fn change_any<A: Clone + Array<OptionVertex<CompleteGraph>>>() {
        let mut rng = rand::weak_rng();
        let n = 30;
        let (g, tree) = graph_tree(n);
        let mut tree: PredecessorTree<CompleteGraph, A> = PredecessorTree::from_iter(g, tree);
        for _ in 0..1000 {
            let mut new = tree.clone();
            let (ins, rem) = new.change_any(&mut rng);
            assert!(new.contains(ins));
            assert!(!new.contains(rem));
            new.check();
            assert!(tree != new);
            tree = new;
        }
    }

    fn make_root<A: Clone + Array<OptionVertex<CompleteGraph>>>() {
        let mut rng = rand::weak_rng();
        let n = 30;
        let (g, tree) = graph_tree(n);
        let mut tree: PredecessorTree<CompleteGraph, A> =
            PredecessorTree::from_iter(g.clone(), tree);
        for u in g.choose_vertex_iter(&mut rng).take(1000) {
            let mut new = tree.clone();
            new.make_root(u);
            assert!(new.is_root(u));
            new.check();
            assert!(tree == new);
            tree = new;
        }
    }

    fn find_root<A: Array<OptionVertex<CompleteGraph>>>() {
        let n = 30;
        let (g, tree) = graph_tree(n);
        let r = g.source(tree[0]);
        let tree: PredecessorTree<CompleteGraph, A> = PredecessorTree::from_iter(g.clone(), tree);
        for v in g.vertices() {
            assert_eq!(r == v, tree.is_root(v));
            assert_eq!(r, tree.find_root(v));
        }
    }

    fn paths<A: Array<OptionVertex<CompleteGraph>>>() {
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

        let tree: PredecessorTree<CompleteGraph, A> = PredecessorTree::from_iter(
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

        assert_eq!(vec![e(1, 0)], vec(tree.edges_to_root(1)));
        assert_eq!(vec![e(2, 1), e(1, 0)], vec(tree.edges_to_root(2)));
        assert_eq!(vec![e(3, 1), e(1, 0)], vec(tree.edges_to_root(3)));
        assert_eq!(vec![e(4, 0)], vec(tree.edges_to_root(4)));
        assert_eq!(vec![e(5, 4), e(4, 0)], vec(tree.edges_to_root(5)));
        assert_eq!(vec![e(6, 2), e(2, 1), e(1, 0)], vec(tree.edges_to_root(6)));
        assert_eq!(vec![e(7, 3), e(3, 1), e(1, 0)], vec(tree.edges_to_root(7)));

        let mut path = vec![];
        tree.find_path(1, 0, &mut path);
        assert_eq!(vec![1, 0], path);

        tree.find_path(0, 1, &mut path);
        assert_eq!(vec![0, 1], path);

        tree.find_path(6, 1, &mut path);
        assert_eq!(vec![6, 2, 1], path);

        tree.find_path(1, 6, &mut path);
        assert_eq!(vec![1, 2, 6], path);

        tree.find_path(1, 4, &mut path);
        assert_eq!(vec![1, 0, 4], path);

        tree.find_path(4, 1, &mut path);
        assert_eq!(vec![4, 0, 1], path);

        tree.find_path(6, 8, &mut path);
        assert_eq!(vec![6, 2, 1, 3, 7, 8], path);

        tree.find_path(7, 6, &mut path);
        assert_eq!(vec![7, 3, 1, 2, 6], path);

        for u in 0..n {
            for v in 0..n {
                tree.find_path(u, v, &mut path);
                if u == v {
                    assert!(path.is_empty());
                } else {
                    assert!(
                        g.is_path(path.windows(2).map(|s| g.edge_by_ends(s[0], s[1]))),
                        "{:?} -> {:?} = {:?}",
                        u,
                        v,
                        path
                    );
                    assert_eq!(u, *path.first().unwrap());
                    assert_eq!(v, *path.last().unwrap());
                }
            }
        }
    }
}
