// TODO: remove
#![allow(dead_code)]

use croaring::Bitmap;
use fera::graph::prelude::*;
use fera::graph::traverse::{Dfs, OnTraverseEvent, TraverseEvent};
use rand::Rng;

use std::rc::Rc;

#[derive(Clone)]
pub struct EulerTourTree<G>
where
    G: WithEdgeIndexProp,
    EdgeIndexProp<G>: Clone,
{
    g: Rc<G>,
    segs: Vec<Rc<Segment<G>>>,
}

impl<G> EulerTourTree<G>
where
    G: IncidenceGraph + WithEdgeIndexProp,
    EdgeIndexProp<G>: Clone,
{
    pub fn new(g: G, edges: &[Edge<G>]) -> Self {
        // TODO: avoid using this intermediary vector
        let mut tree = Vec::with_capacity(2 * (g.num_vertices() - 1));
        g.spanning_subgraph(edges)
            .dfs(OnTraverseEvent(|e| match e {
                TraverseEvent::DiscoverEdge(e) => tree.push(e),
                TraverseEvent::FinishEdge(e) => tree.push(g.reverse(e)),
                _ => (),
            }))
            .run();
        let nsqrt = (2.0 * (g.num_vertices() - 1) as f64).sqrt().ceil() as usize;
        let segs = tree.chunks(nsqrt)
            .map(|t| Rc::new(Segment::new(&g, t)))
            .collect();
        let g = Rc::new(g);
        Self { g, segs }
    }

    pub fn len(&self) -> usize {
        2 * (self.g.num_vertices() - 1)
    }

    pub fn change_parent<R: Rng>(&mut self, mut rng: R) -> (Edge<G>, Edge<G>) {
        loop {
            let e = self.choose_edge(&mut rng);
            let (start, end) = self.range(e).unwrap();
            let to = self.lfind(self.choose_edge_out_range(start, end + 1, &mut rng))
                .unwrap();
            let rem = self.get(start);
            let _to = self.get(to);
            if let Some(ins) = self.g
                .get_edge_by_ends(self.g.target(_to), self.g.target(rem))
            {
                if ins == rem {
                    continue;
                }
                self.set(start, ins);
                let rins = self.g.reverse(ins);
                self.set(end, rins);
                self.move_range(start, end, to);
                return (ins, rem);
            }
        }
    }

    fn move_range(&mut self, start: usize, end: usize, to: usize) {
        assert!(to <= start || end <= to);
        if to == start {
            return;
        }
        self.split(to + 1);
        self.split(start);
        self.split(end + 1);
        assert_eq!(0, self.split_index(start).1);
        assert_eq!(0, self.split_index(to + 1).1);
        let to = self.seg_of(to) + 1;
        let start = self.seg_of(start);
        let end = self.seg_of(end) + 1;

        if to < start {
            self.segs[to..end].rotate(start - to);
        } else {
            self.segs[start..to].rotate(end - start)
        }
    }

    fn split_index(&self, index: usize) -> (usize, usize) {
        assert!(index < self.len());
        let mut count = 0;
        for (i, seg) in self.segs.iter().enumerate() {
            if count + seg.len() <= index {
                count += seg.len()
            } else {
                return (i, index - count);
            }
        }
        unreachable!()
    }

    fn get(&self, index: usize) -> Edge<G> {
        let (i, j) = self.split_index(index);
        self.segs[i].tour[j]
    }

    fn set(&mut self, index: usize, e: Edge<G>) {
        let (i, j) = self.split_index(index);
        let seg = Rc::make_mut(&mut self.segs[i]);
        seg.tour[j] = e;
        seg.update_map();
    }

    fn seg_of(&self, index: usize) -> usize {
        self.split_index(index).0
    }

    fn split(&mut self, index: usize) {
        if index >= self.len() {
            return;
        }
        let (i, j) = self.split_index(index);
        if let Some(new) = Rc::make_mut(&mut self.segs[i]).split(j) {
            self.segs.insert(i, Rc::new(new))
        }
    }

    fn choose_edge<R: Rng>(&self, rng: R) -> Edge<G> {
        // Ignore the first edge
        self.choose_edge_in_range(1, self.len() - 1, rng)
    }

    fn choose_edge_in_range<R: Rng>(&self, start: usize, end: usize, mut rng: R) -> Edge<G> {
        self.get(rng.gen_range(start, end))
    }

    fn choose_edge_out_range<R: Rng>(&self, start: usize, end: usize, mut rng: R) -> Edge<G> {
        let n = (0..start).len() + (end..self.len()).len();
        let i = rng.gen_range(0, n);
        let e = if i < start {
            self.get(i)
        } else {
            self.get(end + i - start)
        };
        debug_assert!({
            let i = self.lfind(e).unwrap();
            i < start || end <= i
        });
        e
    }

    fn range(&self, e: Edge<G>) -> Option<(usize, usize)> {
        self.lfind(e).map(|s| (s, self.rfind(e).unwrap()))
    }

    fn contains(&self, e: Edge<G>) -> bool {
        self.segs.iter().any(|s| s.contains(e))
    }

    fn lfind(&self, e: Edge<G>) -> Option<usize> {
        let mut count = 0;
        for seg in &self.segs {
            if let Some(j) = seg.lfind(e) {
                return Some(count + j);
            }
            count += seg.len();
        }
        None
    }

    fn rfind(&self, e: Edge<G>) -> Option<usize> {
        let mut count = self.len();
        for seg in self.segs.iter().rev() {
            if let Some(j) = seg.rfind(e) {
                return Some(count - seg.len() + j);
            }
            count -= seg.len();
        }
        None
    }

    fn to_vec(&self) -> Vec<Edge<G>> {
        self.segs
            .iter()
            .flat_map(|seg| seg.tour.iter())
            .cloned()
            .collect()
    }

    fn check(&self) {
        use fera::fun::set;
        use fera::graph::algs::{Paths, Trees};
        let edges = || self.segs.iter().flat_map(|seg| seg.tour.iter()).cloned();
        assert!(self.g.spanning_subgraph(set(edges())).is_tree());
        assert!(self.g.is_walk(edges()));
        for (i, e) in edges().enumerate() {
            assert!(self.contains(e));
            assert_eq!(self.get(i), e);
            let (s, t) = self.range(e).unwrap();
            assert!(s < t);
            assert_eq!(self.get(s), e);
            assert_eq!(self.get(t), e);
        }
    }
}

struct Segment<G>
where
    G: WithEdgeIndexProp,
    EdgeIndexProp<G>: Clone,
{
    index: EdgeIndexProp<G>,
    tour: Vec<Edge<G>>,
    map: Bitmap,
}

impl<G> Segment<G>
where
    G: WithEdgeIndexProp,
    EdgeIndexProp<G>: Clone,
{
    fn new(g: &G, edges: &[Edge<G>]) -> Self {
        let index = g.edge_index();
        let map = Bitmap::create();
        let tour = edges.to_vec();
        let mut new = Self { index, tour, map };
        new.update_map();
        new
    }

    fn len(&self) -> usize {
        self.tour.len()
    }

    fn update_map(&mut self) {
        self.map.clear();
        for &e in &self.tour {
            self.map.add(self.index.get(e) as u32)
        }
    }

    fn contains(&self, e: Edge<G>) -> bool {
        self.map.contains(self.index.get(e) as u32)
    }

    fn lfind(&self, e: Edge<G>) -> Option<usize> {
        if self.contains(e) {
            Some(self.tour.iter().position(|f| *f == e).unwrap())
        } else {
            None
        }
    }

    fn rfind(&self, e: Edge<G>) -> Option<usize> {
        if self.contains(e) {
            Some(self.tour.iter().rposition(|f| *f == e).unwrap())
        } else {
            None
        }
    }

    // [0..i) [i..
    fn split(&mut self, i: usize) -> Option<Segment<G>> {
        if i == 0 || i == self.len() {
            None
        } else {
            let mut new = Self {
                index: self.index.clone(),
                tour: self.tour.drain(0..i).collect(),
                map: Bitmap::create(),
            };
            new.update_map();
            self.update_map();
            Some(new)
        }
    }
}

impl<G> Clone for Segment<G>
where
    G: WithEdgeIndexProp,
    EdgeIndexProp<G>: Clone,
{
    fn clone(&self) -> Self {
        Self {
            index: self.index.clone(),
            tour: self.tour.clone(),
            map: self.map.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand;
    use random_sp;

    #[test]
    fn new() {
        let g = CompleteGraph::new(5);
        let e = |u: u32, v: u32| g.edge_by_ends(u, v);
        let edges = [e(0, 1), e(1, 2), e(1, 3), e(3, 4)];
        let tree = EulerTourTree::new(g.clone(), &edges);

        assert_eq!(
            vec![
                e(0, 1),
                e(1, 2),
                e(2, 1),
                e(1, 3),
                e(3, 4),
                e(4, 3),
                e(3, 1),
                e(1, 0),
            ],
            tree.to_vec()
        );
    }

    fn graph_tree(n: u32) -> (CompleteGraph, Vec<Edge<CompleteGraph>>) {
        let g = CompleteGraph::new(n);
        let tree = random_sp(&g, rand::XorShiftRng::new_unseeded());
        (g, tree)
    }

    #[test]
    fn basic() {
        let n = 30;
        for _ in 0..100 {
            let (g, tree) = graph_tree(n);
            EulerTourTree::new(g.clone(), &tree).check();
        }
    }

    fn ptree(g: &CompleteGraph, tour: Vec<Edge<CompleteGraph>>) {
        for e in tour {
            print!("{:?} ", g.end_vertices(e));
        }
        println!("");
    }

    fn _change_parent(n: u32) {
        let mut rng = rand::XorShiftRng::new_unseeded();
        for _ in 0..100 {
            let (g, tree) = graph_tree(n);
            let mut tree = EulerTourTree::new(g.clone(), &tree);
            let (_ins, _rem) = tree.change_parent(&mut rng);
            tree.check();
        }
    }

    #[test]
    fn change_parent() {
        for &n in &[3, 4, 5, 6, 10, 30] {
            _change_parent(n);
        }
    }
}
