use std::cell::RefCell;
use std::fmt::Debug;
use std::hash::Hash;
use std::ops::Deref;

// See nddr module documentation for references.
// TODO: make op1 and op2 receive &mut ?

#[derive(Clone, Copy, Eq, PartialEq, Debug)]
pub struct Ndd<V: Copy + Eq + PartialEq + Debug> {
    vertex: V,
    dep: usize,
    deg: usize,
}

impl<V: Copy + Eq + PartialEq + Debug> Ndd<V> {
    pub fn new(vertex: V, dep: usize, deg: usize) -> Ndd<V> {
        Ndd { vertex, dep, deg }
    }

    pub fn deg(&self) -> usize {
        self.deg
    }

    pub fn vertex(&self) -> V {
        self.vertex
    }
}

// TODO: implement Clone::clone_from
#[derive(Clone, Debug)]
pub struct NddTree<V: Copy + Hash + Eq + PartialEq + Debug> {
    ndds: Vec<Ndd<V>>,
    // parents is used to cache the parent of each node.
    // parent[i] is the parent of ndds[i-1].vertex
    parents: RefCell<Vec<V>>,
    deg: usize,
    deg_in_g: usize,
}

pub fn op1<V>(from: &NddTree<V>, p: usize, to: &NddTree<V>, a: usize) -> (NddTree<V>, NddTree<V>)
where
    V: Copy + Hash + Eq + PartialEq + Debug,
{
    (from.without_subtree(p), to.with_subtree(a, from.subtree(p)))
}

pub fn one_tree_op1<V>(t: &NddTree<V>, p: usize, a: usize) -> NddTree<V>
where
    V: Copy + Hash + Eq + PartialEq + Debug,
{
    assert!(!t.is_ancestor(p, a));
    let new = t.without_subtree(p);
    let sub = t.subtree(p);
    if a < p {
        new.with_subtree(a, sub)
    } else {
        new.with_subtree(a - sub.len(), sub)
    }
}

pub fn one_tree_op2<V>(t: &NddTree<V>, p: usize, r: usize, a: usize) -> NddTree<V>
where
    V: Copy + Hash + Eq + PartialEq + Debug,
{
    assert!(t.is_ancestor(p, r));
    assert!(!t.is_ancestor(r, a));
    // It is not clear in page 833 - C) Operations on One-Tree Forests - how op2 can be executed in
    // one tree if p is ancestor of a, so for while we do not accept such case. Restricting the
    // selection does not change the running time for complete graphs, see Forest::find_vertices_op2
    assert!(!t.is_ancestor(p, a));
    let new = t.without_subtree(p);
    // TODO: avoid allocations
    // See the comments of op2
    let mut sub = NddTree::new(t.subtree(p).to_vec());
    sub.ndds[0].deg -= 1;
    sub.change_root(r - p);
    sub.ndds[0].deg += 1;
    if a < p {
        new.with_subtree(a, &*sub)
    } else {
        new.with_subtree(a - sub.len(), &*sub)
    }
}

pub fn op2<V>(
    from: &NddTree<V>,
    p: usize,
    r: usize,
    to: &NddTree<V>,
    a: usize,
) -> (NddTree<V>, NddTree<V>)
where
    V: Copy + Hash + Eq + PartialEq + Debug,
{
    assert!(from.is_ancestor(p, r));
    // subtree returns a slice, so we make a new tree of it and decrement the root deg
    let mut t = NddTree::new(from.subtree(p).to_vec());
    t.ndds[0].deg -= 1;
    t.change_root(r - p);
    // t will be inserted as a subtree, so we must increment the root deg
    // This is not necessary in other places because with_subtree receive a slice as parameter, and
    // the root element of the slice already have the correct deg. In this case, t is a new tree,
    // not a slice.
    t.ndds[0].deg += 1;
    (from.without_subtree(p), to.with_subtree(a, &t))
}

impl<V> Deref for NddTree<V>
where
    V: Copy + Hash + Eq + PartialEq + Debug,
{
    type Target = Vec<Ndd<V>>;

    fn deref(&self) -> &Self::Target {
        &self.ndds
    }
}

impl<V> NddTree<V>
where
    V: Copy + Hash + Eq + PartialEq + Debug,
{
    pub fn from_vecs(vertices: &[V], dep: &[usize], deg: &[usize]) -> Self {
        // TODO: Validade input or make private
        NddTree::new(
            vertices
                .into_iter()
                .zip(dep)
                .zip(deg)
                .map(|((&v, &dp), &dg)| Ndd::new(v, dp, dg))
                .collect(),
        )
    }

    pub fn new(ndds: Vec<Ndd<V>>) -> Self {
        // TODO: Validade input or make private
        NddTree {
            ndds: ndds,
            parents: RefCell::new(vec![]),
            deg: 0,
            deg_in_g: 0,
        }
    }

    pub fn without_subtree(&self, i: usize) -> Self {
        assert!(i != 0, "Cannot remove root");

        let end = self.subtree_end(i);
        let mut new = Vec::with_capacity(self.len() - (end - i));
        new.extend_from_slice(&self[..i]);
        new.extend_from_slice(&self[end..]);
        new[self.parent(i).unwrap()].deg -= 1;

        NddTree::new(new)
    }

    pub fn with_subtree(&self, i: usize, t: &[Ndd<V>]) -> Self {
        let mut new = Vec::with_capacity(self.len() + t.len());
        new.extend_from_slice(&self[..(i + 1)]);
        new.extend_from_slice(t);
        new.extend_from_slice(&self[(i + 1)..]);

        for ndd in &mut new[(i + 1)..(i + 1 + t.len())] {
            ndd.dep = ndd.dep - t[0].dep + self[i].dep + 1;
        }

        new[i].deg += 1;

        NddTree::new(new)
    }

    pub fn change_root(&mut self, r: usize) {
        // TODO: how to do it without allocating?
        let mut new = Vec::with_capacity(self.len());
        new.extend_from_slice(self.subtree(r));
        for ndd in &mut new {
            ndd.dep -= self[r].dep;
        }

        // TODO: Pre-compute subtree_end. Benchmark
        let mut r = r;
        let mut depth = 1;
        while let Some(p) = self.parent(r) {
            // Start of p subtree on new
            let s = new.len();
            // Copy p subtree skipping r subtree
            new.extend_from_slice(&self[p..r]);
            new.extend_from_slice(&self[self.subtree_end(r)..self.subtree_end(p)]);
            // Update depth
            for ndd in &mut new[s..] {
                ndd.dep = (ndd.dep - self[p].dep) + depth;
            }
            depth += 1;
            r = p;
        }
        self.ndds = new;
    }

    pub fn subtree_end(&self, i: usize) -> usize {
        let mut end = i + 1;
        while end < self.len() && self[end].dep > self[i].dep {
            end += 1;
        }
        end
    }

    pub fn is_ancestor(&self, a: usize, i: usize) -> bool {
        i >= a && i < self.subtree_end(a)
    }

    pub fn subtree(&self, i: usize) -> &[Ndd<V>] {
        &self[i..self.subtree_end(i)]
    }

    pub fn find_vertex(&self, v: V) -> Option<usize> {
        self.iter().position(|ndd| ndd.vertex == v)
    }

    pub fn contains_edge(&self, u: V, v: V) -> bool {
        match (self.find_vertex(u), self.find_vertex(v)) {
            (Some(pu), Some(pv)) => self.parent(pu) == Some(pv) || self.parent(pv) == Some(pu),
            _ => false,
        }
    }

    pub fn calc_degs<F>(&mut self, degree: F)
    where
        F: Fn(V) -> usize,
    {
        self.deg = 0;
        self.deg_in_g = 0;
        for ndd in &self.ndds {
            self.deg += ndd.deg;
            self.deg_in_g += degree(ndd.vertex);
        }
    }

    pub fn deg(&self) -> usize {
        self.deg
    }

    pub fn deg_in_g(&self) -> usize {
        self.deg_in_g
    }

    pub fn parent_vertex(&self, i: usize) -> Option<V> {
        self.parent(i).map(|p| self[p].vertex)
    }

    pub fn parent(&self, i: usize) -> Option<usize> {
        self[0..i].iter().rposition(|ndd| ndd.dep < self[i].dep)
    }

    pub fn edges(&self) -> Vec<(V, V)> {
        let mut edges = Vec::with_capacity(self.len() - 1);
        self.for_each_edge(|u, v| edges.push((u, v)));
        edges
    }

    pub fn for_each_edge<F>(&self, mut fun: F)
    where
        F: FnMut(V, V),
    {
        if self.parents.borrow().is_empty() {
            // Its not clear on page 836 - a) Traverse F_from ...
            // how this can be done.
            let mut parents = self.parents.borrow_mut();
            let mut s = vec![&self[0]];
            for ndd in &self[1..] {
                let mut p = *s.last().unwrap();
                while p.dep >= ndd.dep {
                    s.pop();
                    p = *s.last().unwrap();
                }
                parents.push(p.vertex);
                s.push(ndd);
            }
        }
        let parents = self.parents.borrow();
        for i in 0..self.len() - 1 {
            fun(self[i + 1].vertex, parents[i]);
        }
    }
}

impl<V> PartialEq for NddTree<V>
where
    V: Copy + Hash + Eq + PartialEq + Debug,
{
    fn eq(&self, other: &Self) -> bool {
        // TODO: Write test
        self.len() == other.len() && self.edges().iter().all(|e| other.contains_edge(e.0, e.1))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fera::fun::set;

    // t1 is T_from Fig 3 (a) - pg 832
    fn t1() -> NddTree<usize> {
        NddTree::from_vecs(
            &[3, 1, 2, 7, 8, 9, 4, 5, 6, 0],
            &[0, 1, 2, 2, 3, 3, 3, 4, 4, 3],
            &[1, 3, 1, 5, 1, 1, 3, 1, 1, 1],
        )
    }

    fn t1_without_subtree_1() -> NddTree<usize> {
        NddTree::from_vecs(&[3], &[0], &[0])
    }

    fn t1_without_subtree_2() -> NddTree<usize> {
        NddTree::from_vecs(
            &[3, 1, 7, 8, 9, 4, 5, 6, 0],
            &[0, 1, 2, 3, 3, 3, 4, 4, 3],
            &[1, 2, 5, 1, 1, 3, 1, 1, 1],
        )
    }

    fn t1_without_subtree_3() -> NddTree<usize> {
        NddTree::from_vecs(&[3, 1, 2], &[0, 1, 2], &[1, 2, 1])
    }

    fn t1_without_subtree_6() -> NddTree<usize> {
        NddTree::from_vecs(
            &[3, 1, 2, 7, 8, 9, 0],
            &[0, 1, 2, 2, 3, 3, 3],
            &[1, 3, 1, 4, 1, 1, 1],
        )
    }

    #[test]
    fn subtree_last() {
        let t = t1();
        assert_eq!(10, t.subtree_end(0));
        assert_eq!(10, t.subtree_end(1));
        assert_eq!(3, t.subtree_end(2));
        assert_eq!(10, t.subtree_end(3));
        assert_eq!(5, t.subtree_end(4));
        assert_eq!(6, t.subtree_end(5));
        assert_eq!(9, t.subtree_end(6));
        assert_eq!(8, t.subtree_end(7));
        assert_eq!(9, t.subtree_end(8));
        assert_eq!(10, t.subtree_end(9));
    }

    #[test]
    fn parent() {
        let t = t1();
        assert_eq!(None, t.parent(0));
        assert_eq!(Some(0), t.parent(1));
        assert_eq!(Some(1), t.parent(2));
        assert_eq!(Some(1), t.parent(3));
        assert_eq!(Some(3), t.parent(4));
        assert_eq!(Some(3), t.parent(5));
        assert_eq!(Some(3), t.parent(6));
        assert_eq!(Some(6), t.parent(7));
        assert_eq!(Some(6), t.parent(8));
        assert_eq!(Some(3), t.parent(9));
    }

    #[test]
    fn is_ancestor() {
        let t = t1();
        assert!(t.is_ancestor(0, 6)); // 3, 4
        assert!(t.is_ancestor(1, 6)); // 1, 4
        assert!(t.is_ancestor(3, 6)); // 7, 4
        assert!(t.is_ancestor(6, 6)); // 4, 4
        assert!(!t.is_ancestor(2, 6)); // 2, 4
        assert!(!t.is_ancestor(4, 6)); // 8, 4
        assert!(!t.is_ancestor(5, 6)); // 9, 4
        assert!(!t.is_ancestor(7, 6)); // 5, 4
        assert!(!t.is_ancestor(8, 6)); // 6, 4
        assert!(!t.is_ancestor(9, 6)); // 0, 4
    }

    #[test]
    fn test_edges() {
        let exp = set(&[
            (1, 3),
            (2, 1),
            (7, 1),
            (8, 7),
            (9, 7),
            (4, 7),
            (5, 4),
            (6, 4),
            (0, 7),
        ]);
        let t = t1();
        assert_eq!(exp, set(t.edges().iter()));
    }

    #[test]
    fn without_subtree() {
        assert_eq!(t1_without_subtree_1(), t1().without_subtree(1));
        assert_eq!(t1_without_subtree_2(), t1().without_subtree(2));
        assert_eq!(t1_without_subtree_3(), t1().without_subtree(3));
        assert_eq!(t1_without_subtree_6(), t1().without_subtree(6));
    }

    #[test]
    fn with_subtree() {
        assert_eq!(
            t1(),
            t1_without_subtree_1().with_subtree(0, t1().subtree(1))
        );
        assert_eq!(
            t1(),
            t1_without_subtree_2().with_subtree(1, t1().subtree(2))
        );
        assert_eq!(
            t1(),
            t1_without_subtree_3().with_subtree(1, t1().subtree(3))
        );
        assert_eq!(
            t1(),
            t1_without_subtree_6().with_subtree(3, t1().subtree(6))
        );
    }

    #[test]
    fn change_root() {
        // Fig 4 (c) - pg 834
        let exp = NddTree::from_vecs(
            &[4, 5, 6, 7, 8, 9, 0, 1, 2],
            &[0, 1, 1, 1, 2, 2, 2, 2, 3],
            &[3, 1, 1, 5, 1, 1, 1, 2, 1],
        );

        // Subtree with root 1 - Fig 4 (a) - pg 834
        let mut tree = NddTree::from_vecs(
            &[1, 2, 7, 8, 9, 4, 5, 6, 0],
            &[0, 1, 1, 2, 2, 2, 3, 3, 2],
            &[2, 1, 5, 1, 1, 3, 1, 1, 1],
        );
        tree.change_root(5);
        assert_eq!(exp, tree);
    }

    #[test]
    fn test_op1() {
        // Example on pg 832
        let from = t1();
        let to = NddTree::from_vecs(&[11], &[0], &[0]);

        let exp_from = NddTree::from_vecs(
            &[3, 1, 2, 7, 8, 9, 0],
            &[0, 1, 2, 2, 3, 3, 3],
            &[1, 3, 1, 4, 1, 1, 1],
        );

        let exp_to = NddTree::from_vecs(&[11, 4, 5, 6], &[0, 1, 2, 2], &[1, 3, 1, 1]);

        let (from, to) = op1(&from, 6, &to, 0);

        assert_eq!(exp_from, from);
        assert_eq!(exp_to, to);
    }

    #[test]
    fn test_one_tree_op1() {
        let t = t1();
        let exp = NddTree::from_vecs(
            &[3, 1, 2, 4, 5, 6, 7, 8, 9, 0],
            &[0, 1, 2, 3, 4, 4, 2, 3, 3, 3],
            &[1, 3, 2, 2, 1, 1, 4, 1, 1, 1],
        );
        let new = one_tree_op1(&t, 6, 2); // vertex 4 and 2
        assert_eq!(exp, new);
    }

    #[test]
    fn test_op2() {
        // Example on pg 834
        let from = t1();
        let to = NddTree::from_vecs(&[11], &[0], &[0]);

        let exp_from = NddTree::from_vecs(&[3], &[0], &[0]);

        let exp_to = NddTree::from_vecs(
            &[11, 4, 5, 6, 7, 8, 9, 0, 1, 2],
            &[0, 1, 2, 2, 2, 3, 3, 3, 3, 4],
            &[1, 4, 1, 1, 5, 1, 1, 1, 2, 1],
        );

        let (from, to) = op2(&from, 1, 6, &to, 0);

        assert_eq!(exp_from, from);
        assert_eq!(exp_to, to);
    }

    #[test]
    fn test_one_tree_op2() {
        let t = t1();
        let exp = NddTree::from_vecs(
            &[3, 1, 2, 6, 4, 5, 7, 8, 9, 0],
            &[0, 1, 2, 3, 4, 5, 5, 6, 6, 6],
            &[1, 2, 2, 2, 3, 1, 4, 1, 1, 1],
        );
        let new = one_tree_op2(&t, 3, 8, 2); // vertex 7, 6 and 2
        assert_eq!(exp, new);
    }
}
