#![allow(dead_code)]

use fera::graph::prelude::*;
use fera::graph::choose::Choose;
use fera::graph::traverse::{Dfs, OnTraverseEvent, TraverseEvent};
use fera_array::{CowNestedArray, DynamicArray};
use rand::Rng;
use rpds::HashTrieMap;

use std::fmt::{self, Debug};
use std::rc::Rc;

#[derive(Clone)]
pub struct Tour<G: WithEdge> {
    g: Rc<G>,
    // Map from TourEdge.index() to Edge<G>
    edges: CowNestedArray<Edge<G>>,
    // Map from Edge<G> to TourEdge
    tour_edges: HashTrieMap<Edge<G>, TourEdge>,
    segs: Rc<Vec<Rc<Segment>>>,
    len: usize,
}

impl<G> Tour<G>
where
    G: AdjacencyGraph + WithVertexIndexProp + Choose,
{
    #[inline(never)]
    pub fn new(g: Rc<G>, edges: &[Edge<G>]) -> Self {
        // TODO: avoid using this intermediary vector
        let mut tour = Vec::with_capacity(2 * (g.num_vertices() - 1));
        let mut eds = CowNestedArray::with_capacity(g.num_vertices() - 1);
        let mut stack = vec![];
        let mut tour_edges = HashTrieMap::new();
        let mut id = 0;
        g.spanning_subgraph(edges)
            .dfs(OnTraverseEvent(|e| match e {
                TraverseEvent::DiscoverEdge(e) => {
                    eds.push(e);
                    let te = TourEdge::new(id);
                    tour_edges = tour_edges.insert(e, te);
                    tour.push(te);
                    stack.push(id);
                    id += 1;
                }
                TraverseEvent::FinishEdge(e) => {
                    let id = stack.pop().unwrap();
                    assert_eq!(eds[id], e);
                    tour.push(TourEdge::new_reversed(id));
                }
                _ => (),
            }))
            .run();
        Self::new_(g, &tour, eds, tour_edges)
    }

    #[inline(never)]
    fn new_(
        g: Rc<G>,
        values: &[TourEdge],
        edges: CowNestedArray<Edge<G>>,
        tour_edges: HashTrieMap<Edge<G>, TourEdge>,
    ) -> Self {
        let nsqrt = (values.len() as f64).sqrt().ceil() as usize;
        let mut tour = Self {
            g,
            edges,
            tour_edges,
            segs: Rc::new(vec![]),
            len: values.len(),
        };
        let segs = values
            .chunks(nsqrt)
            .map(|t| Rc::new(tour.new_segment(t.into())))
            .collect();
        tour.segs = Rc::new(segs);
        tour
    }

    pub fn change_parent<R: Rng>(&mut self, mut rng: R) -> TourEdge {
        let e = self.choose_non_tree_edge(&mut rng);
        let (a, b) = self.g.ends(e);
        let a_sub = self.subtree(a);
        let b_sub = self.subtree(b);
        println!("change-parent");
        println!("before: {:?}", self.to_vec());
        println!("before: {:?}", self.to_path());
        println!("a: {:?}, {:?}", a, a_sub);
        println!("b: {:?}, {:?}", b, b_sub);
        // println!("segs: {:?}", self.segs);
        if a_sub.contains(&b_sub) {
            self.move_(e, a_sub.end, b_sub);
        } else if b_sub.contains(&a_sub) {
            let e = self.g.reverse(e);
            self.move_(e, b_sub.end, a_sub);
        } else if rng.gen() {
            self.move_(e, a_sub.end, b_sub);
        } else {
            let e = self.g.reverse(e);
            self.move_(e, b_sub.end, a_sub);
        }
        println!("after: {:?}", self.to_vec());
        println!("after: {:?}", self.to_path());
        self.get((0, 0))
    }

    fn move_(&mut self, new: Edge<G>, to: (usize, usize), sub: Subtree) {
        let x = self.prev_pos(sub.start);
        let y = self.next_pos(sub.end);
        let te = self.get(x);
        self.set_tour_edge(te, new);
        let (a, b) = self.ends(te);
        self.segment_update_ends(x, (a, b));
        self.segment_update_ends(y, (a, b));
        if y <= to {
            self.move_after(to, x, y);
        } else {
            assert!(to <= x);
            self.move_before(to, x, y);
        }
    }

    fn set_tour_edge(&mut self, tour_edge: TourEdge, edge: Edge<G>) {
        let i = tour_edge.index();
        self.tour_edges = self.tour_edges.remove(&self.edges[i]);
        self.tour_edges = self.tour_edges.insert(edge, tour_edge);
        self.edges[i] = edge;
    }

    /*
    pub fn change_parent<R: Rng>(&mut self, mut rng: R) -> TourEdge {
        let (start, end) = self.choose_subtree(&mut rng);
        let e = self.get(start);
        assert!(start < end);
        assert_eq!(self.get(start), self.get(end));
        loop {
            // TODO: direct selection
            let to = self.choose_pos(&mut rng);
            if start != (0, 0) && to < self.prev_pos(start) {
                self.move_before(to, start, end);
                break;
            }
            if to > end {
                self.move_after(to, start, end);
                break;
            }
        }
        e
    }
    */

    fn choose_non_tree_edge<R: Rng>(&self, rng: R) -> Edge<G> {
        self.g
            .choose_edge_iter(rng)
            .filter(|e| !self.tour_edges.contains_key(e))
            .next()
            .unwrap()
    }

    fn ends(&self, e: TourEdge) -> (usize, usize) {
        let i = e.index();
        let (a, b) = self.g.ends(self.edges[i]);
        let index = self.g.vertex_index();
        if e.is_reverse() {
            (index.get(b), index.get(a))
        } else {
            (index.get(a), index.get(b))
        }
    }

    fn subtree(&self, v: Vertex<G>) -> Subtree {
        let vi = self.g.vertex_index().get(v);
        let mut start = None;
        println!("start");
        for (a, seg) in self.segs.iter().enumerate() {
            println!("a: {}", a);
            if let Some(b) = self.segment_position_source(seg, vi) {
                start = Some((a, b));
                break;
            }
        }

        println!("end");
        for (a, seg) in self.segs.iter().enumerate().rev() {
            println!("a: {}", a);
            if let Some(b) = self.segment_rposition_target(seg, vi) {
                return Subtree::new(start.unwrap(), (a, b));
            }
        }

        panic!()
    }

    fn segment_position_source(&self, seg: &Segment, v: usize) -> Option<usize> {
        println!("position");
        if self.segment_contains_vertex(seg, v) {
            for (i, edge) in seg.edges.iter().enumerate() {
                println!("i = {}", i);
                if self.ends(*edge).0 == v {
                    return Some(i);
                }
            }
        }
        None
    }

    fn segment_rposition_target(&self, seg: &Segment, v: usize) -> Option<usize> {
        println!("rposition");
        if self.segment_contains_vertex(seg, v) {
            println!("yes: {:?}", seg);
            for (i, edge) in seg.edges.iter().enumerate().rev() {
                println!("i = {}", i);
                if self.ends(*edge).1 == v {
                    return Some(i);
                }
            }
        } else {
            println!("no: {:?}", seg);
        }
        None
    }

    fn segment_update_ends(&mut self, (i, j): (usize, usize), (a, b): (usize, usize)) {
        let new = self.new_segment(self.segs[i].edges.clone());
        let segs = Rc::make_mut(&mut self.segs);
        let seg = Rc::make_mut(&mut segs[i]);
        *seg = new;
    }

    fn segment_contains_vertex(&self, seg: &Segment, v: usize) -> bool {
        seg.vertex_pos
            .get(v)
            .map(|i| {
                if let Some((a, b)) = seg.edges.get(*i).map(|edge| self.ends(*edge)) {
                    a == v || b == v
                } else {
                    false
                }
            })
            .unwrap_or(false)
    }

    pub fn range(&self, e: TourEdge) -> Option<((usize, usize), (usize, usize))> {
        self.position(e)
            .and_then(|start| self.rposition(e).map(|end| (start, end)))
    }

    pub fn get(&self, (i, j): (usize, usize)) -> TourEdge {
        self.segs[i].edges[j]
    }

    pub fn get_next(&self, p: (usize, usize)) -> TourEdge {
        self.get(self.next_pos(p))
    }

    pub fn get_prev(&self, p: (usize, usize)) -> TourEdge {
        self.get(self.prev_pos(p))
    }

    fn move_before(&mut self, to: (usize, usize), start: (usize, usize), end: (usize, usize)) {
        let mut segs = Vec::with_capacity(self.segs.len());
        {
            let segs = &mut segs;
            let mut iter = self.seg_iter(self.first_pos(), self.next_pos(to));
            let mut last = iter.next().unwrap();
            last = Self::extend(self, segs, last, iter);
            last = Self::extend(self, segs, last, self.seg_iter(start, self.next_pos(end)));
            last = Self::extend(self, segs, last, self.seg_iter(self.next_pos(to), start));
            last = Self::extend(
                self,
                segs,
                last,
                self.seg_iter(self.next_pos(end), self.end_pos()),
            );
            Self::push(self, segs, last);
        }
        self.segs = segs.into();
    }

    fn move_after(&mut self, to: (usize, usize), start: (usize, usize), end: (usize, usize)) {
        let mut segs = Vec::with_capacity(self.segs.len());
        {
            let segs = &mut segs;
            let mut iter = self.seg_iter(self.first_pos(), start);
            let mut last = if let Some(mut last) = iter.next() {
                last = Self::extend(self, segs, last, iter);
                Self::extend(
                    self,
                    segs,
                    last,
                    self.seg_iter(self.next_pos(end), self.next_pos(to)),
                )
            } else {
                let mut iter = self.seg_iter(self.next_pos(end), self.next_pos(to));
                let mut last = iter.next().unwrap();
                Self::extend(self, segs, last, iter)
            };
            last = Self::extend(self, segs, last, self.seg_iter(start, self.next_pos(end)));
            last = Self::extend(
                self,
                segs,
                last,
                self.seg_iter(self.next_pos(to), self.end_pos()),
            );
            Self::push(self, segs, last);
        }
        self.segs = segs.into();
    }

    fn extend<'a>(
        &self,
        segs: &mut Vec<Rc<Segment>>,
        mut last: Seg<'a>,
        iter: SegIter<'a>,
    ) -> Seg<'a> {
        for seg in iter {
            self.push(segs, last);
            last = seg;
        }
        last
    }

    fn push<'a>(&self, segs: &mut Vec<Rc<Segment>>, last: Seg<'a>) {
        match last {
            Seg::Complete(seg) => segs.push(Rc::clone(seg)),
            Seg::Partial(values) => segs.push(Rc::new(self.new_segment(values.into()))),
        }
    }

    fn choose_subtree<R: Rng>(&mut self, mut rng: R) -> ((usize, usize), (usize, usize)) {
        loop {
            let (start, end) = self.range(self.choose(&mut rng)).unwrap();
            if start != self.first_pos() || end != self.last_pos() {
                return (start, end);
            }
        }
    }

    fn choose_pos<R: Rng>(&self, mut rng: R) -> (usize, usize) {
        self.split_index(rng.gen_range(0, self.len()))
    }

    fn choose<R: Rng>(&self, rng: R) -> TourEdge {
        self.get(self.choose_pos(rng))
    }

    fn position(&self, e: TourEdge) -> Option<(usize, usize)> {
        for (i, seg) in self.segs.iter().enumerate() {
            if let Some(j) = seg.position(e) {
                return Some((i, j));
            }
        }
        None
    }

    fn rposition(&self, e: TourEdge) -> Option<(usize, usize)> {
        for (i, seg) in self.segs.iter().enumerate().rev() {
            if let Some(j) = seg.rposition(e) {
                return Some((i, j));
            }
        }
        None
    }

    fn next_pos(&self, (i, j): (usize, usize)) -> (usize, usize) {
        if j == self.segs[i].len() - 1 {
            (i + 1, 0)
        } else {
            (i, j + 1)
        }
    }

    fn prev_pos(&self, (i, j): (usize, usize)) -> (usize, usize) {
        if j == 0 {
            if i == 0 {
                panic!()
            }
            (i - 1, self.segs[i - 1].len() - 1)
        } else {
            (i, j - 1)
        }
    }

    fn first_pos(&self) -> (usize, usize) {
        (0, 0)
    }

    fn last_pos(&self) -> (usize, usize) {
        (self.segs.len() - 1, self.segs.last().unwrap().len() - 1)
    }

    fn end_pos(&self) -> (usize, usize) {
        (self.segs.len() - 1, self.segs.last().unwrap().len())
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

    fn len(&self) -> usize {
        self.len
    }

    fn seg_iter(&self, start: (usize, usize), end: (usize, usize)) -> SegIter {
        SegIter::new(&self.segs, start, end)
    }

    fn new_segment(&self, edges: Vec<TourEdge>) -> Segment {
        println!("new_segment");

        let m = edges.iter().map(|e| e.index()).max().unwrap();
        let mut edge_pos = unsafe { vec_new_uninitialized(m + 1) };
        for (i, edge) in edges.iter().enumerate() {
            edge_pos[edge.index()] = i;
        }

        let m = edges
            .iter()
            .map(|e| {
                let (a, b) = self.ends(*e);
                a.max(b)
            })
            .max()
            .unwrap();
        let mut vertex_pos = unsafe { vec_new_uninitialized(m + 1) };
        for (i, edge) in edges.iter().enumerate() {
            let (a, b) = self.ends(*edge);
            vertex_pos[a] = i;
            vertex_pos[b] = i;
        }

        Segment {
            edges,
            edge_pos,
            vertex_pos,
        }
    }

    // #[cfg(test)]
    pub fn to_vec(&self) -> Vec<TourEdge> {
        self.segs
            .iter()
            .flat_map(|seg| seg.edges.iter().cloned())
            .collect()
    }

    pub fn to_path(&self) -> Vec<(usize, usize)> {
        self.to_vec().into_iter().map(|t| self.ends(t)).collect()
    }

    #[cfg(test)]
    pub fn check(&self) {
        let mut last = self.ends(self.get((0, 0))).0;
        for edge in self.to_vec() {
            let v = self.ends(edge);
            assert_eq!(last, v.0);
            last = v.1;
        }
        let mut seen = vec![0u8; self.len()];
        let mut stack = vec![];
        for value in self.to_vec() {
            seen[value.0] += 1;
            if stack.last().cloned() == Some(value) {
                stack.pop();
            } else {
                stack.push(value);
            }

            let (start, end) = self.range(value).unwrap();
            assert!(start < end);
            assert_eq!(value, self.get(start));
            assert_eq!(value, self.get(end));
        }
        println!("check: {:?}", self.to_vec());
        assert!(stack.is_empty(), "{:?}", self.segs);
        for (i, count) in seen.into_iter().enumerate() {
            assert_eq!(1, count, "id: {}, {:?}", i, self.segs);
        }
        assert_eq!(self.segs.iter().map(|s| s.len()).sum::<usize>(), self.len());
    }
}

#[derive(Copy, Clone, PartialEq, Debug)]
struct Subtree {
    start: (usize, usize),
    end: (usize, usize),
}

impl Subtree {
    fn new(start: (usize, usize), end: (usize, usize)) -> Self {
        Self { start, end }
    }

    fn contains(&self, other: &Subtree) -> bool {
        self.start <= other.start && other.end <= self.end
    }
}

#[derive(Clone, Debug)]
struct Segment {
    edges: Vec<TourEdge>,
    // TODO: use a bit vec
    edge_pos: Vec<usize>, // TODO: u32?
    vertex_pos: Vec<usize>,
}

unsafe fn vec_new_uninitialized<T>(n: usize) -> Vec<T> {
    use fera::ext::VecExt;

    let mut vec = Vec::new_uninitialized(n);
    vec.set_len(n);
    vec
}

impl Segment {
    fn len(&self) -> usize {
        self.edges.len()
    }

    fn contains_edge(&self, e: TourEdge) -> bool {
        let index = e.index();
        self.edge_pos
            .get(index)
            .map(|&i| self.edges.get(i) == Some(&e))
            .unwrap_or(false)
    }

    fn position(&self, e: TourEdge) -> Option<usize> {
        if self.contains_edge(e) {
            let p = self.edges.iter().position(|v| *v == e);
            debug_assert!(p.is_some());
            p
        } else {
            None
        }
    }

    fn rposition(&self, e: TourEdge) -> Option<usize> {
        if self.contains_edge(e) {
            let p = self.edges.iter().rposition(|v| *v == e);
            debug_assert!(p.is_some());
            p
        } else {
            None
        }
    }
}

#[derive(Copy, Clone, Debug)]
enum Seg<'a> {
    Partial(&'a [TourEdge]),
    Complete(&'a Rc<Segment>),
}

struct SegIter<'a> {
    segs: &'a [Rc<Segment>],
    cur: (usize, usize),
    end: (usize, usize),
}

impl<'a> SegIter<'a> {
    fn new(segs: &'a [Rc<Segment>], cur: (usize, usize), end: (usize, usize)) -> Self {
        SegIter { segs, cur, end }
    }
}

impl<'a> Iterator for SegIter<'a> {
    type Item = Seg<'a>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.cur.0 < self.end.0 {
            let (i, j) = self.cur;
            self.cur = (i + 1, 0);
            if j == 0 {
                Some(Seg::Complete(&self.segs[i]))
            } else {
                Some(Seg::Partial(&self.segs[i].edges[j..]))
            }
        } else if self.cur.0 == self.end.0 {
            // same segment
            let (i, a) = self.cur;
            let b = self.end.1;
            self.cur = (i + 1, 0);
            if a == b {
                None
            } else if a == 0 && b == self.segs[i].len() {
                Some(Seg::Complete(&self.segs[i]))
            } else {
                Some(Seg::Partial(&self.segs[i].edges[a..b]))
            }
        } else {
            None
        }
    }
}

// TODO: use u32
#[derive(Copy, Clone)]
pub struct TourEdge(usize);

impl TourEdge {
    pub fn new(val: usize) -> Self {
        TourEdge(val << 1)
    }

    pub fn new_reversed(val: usize) -> Self {
        TourEdge((val << 1) + 1)
    }

    pub fn index(self) -> usize {
        self.0 >> 1
    }

    pub fn is_reverse(self) -> bool {
        self.0 & 1 == 1
    }
}

impl PartialEq for TourEdge {
    fn eq(&self, other: &Self) -> bool {
        self.index() == other.index()
    }
}

impl Debug for TourEdge {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let id = if self.is_reverse() {
            -(self.index() as isize)
        } else {
            self.index() as isize
        };
        write!(f, "{}", id)
    }
}

#[cfg(test)]
mod tests {
    use fera_array::{Array, DynamicArray};
    use super::*;
    use rand;
    use random::random_sp;

    fn segment_contains_source(
        tour: &Tour<CompleteGraph>,
        seg: &Segment,
        in_: &[usize],
        not_in: &[usize],
    ) {
        for v in in_ {
            assert!(tour.segment_contains_vertex(seg, *v), "v = {}", v);
        }
        for v in not_in {
            assert!(!tour.segment_contains_vertex(seg, *v), "v = {}", v);
        }
    }

    #[test]
    fn subtree() {
        let g = Rc::new(CompleteGraph::new(9));
        let e = |u: u32, v: u32| g.edge_by_ends(u, v);
        let mut edges = CowNestedArray::with_capacity(g.num_vertices() - 1);
        edges.push(e(0, 1));
        edges.push(e(1, 2));
        edges.push(e(2, 3));
        edges.push(e(1, 4));
        edges.push(e(0, 5));
        edges.push(e(5, 6));
        edges.push(e(5, 7));
        edges.push(e(5, 8));

        let mut tour_edges = HashTrieMap::new();
        for i in 0..edges.len() {
            tour_edges = tour_edges.insert(edges[i], TourEdge::new(i));
        }

        let tour = vec![
            TourEdge::new(0),
            TourEdge::new(1),
            TourEdge::new(2),
            TourEdge::new_reversed(2),
            TourEdge::new_reversed(1),
            TourEdge::new(3),
            TourEdge::new_reversed(3),
            TourEdge::new_reversed(0),
            TourEdge::new(4),
            TourEdge::new(5),
            TourEdge::new_reversed(5),
            TourEdge::new(6),
            TourEdge::new_reversed(6),
            TourEdge::new(7),
            TourEdge::new_reversed(7),
            TourEdge::new_reversed(4),
        ];

        let tour = Tour::new_(g.clone(), &*tour, edges, tour_edges);

        segment_contains_source(&tour, &tour.segs[0], &[0, 1, 2, 3], &[4, 5, 6, 7, 8]);

        assert_eq!(Subtree::new((0, 0), (3, 3)), tour.subtree(0));
        assert_eq!(Subtree::new((0, 1), (1, 2)), tour.subtree(1));
        assert_eq!(Subtree::new((0, 2), (0, 3)), tour.subtree(2));
        assert_eq!(Subtree::new((0, 3), (0, 2)), tour.subtree(3));
        assert_eq!(Subtree::new((1, 2), (1, 1)), tour.subtree(4));
        assert_eq!(Subtree::new((2, 1), (3, 2)), tour.subtree(5));
        assert_eq!(Subtree::new((2, 2), (2, 1)), tour.subtree(6));
        assert_eq!(Subtree::new((3, 0), (2, 3)), tour.subtree(7));
        assert_eq!(Subtree::new((3, 2), (3, 1)), tour.subtree(8));
    }

    #[test]
    fn check() {
        let mut rng = rand::XorShiftRng::new_unseeded();
        for n in 5..30 {
            let g = Rc::new(CompleteGraph::new(n));
            let mut tour = Tour::new(g.clone(), &*random_sp(&*g, &mut rng));
            tour.check();
            for _ in 0..100 {
                let old = tour.clone();
                tour.change_parent(&mut rng);
                tour.check();
                assert_ne!(old.to_vec(), tour.to_vec());
            }
        }
    }

    #[test]
    fn check_simple() {
        let mut rng = rand::weak_rng();
        for n in 5..30 {
            let mut tour = SimpleTour::new((0..n).chain((0..n).rev()).collect());
            tour.check();
            for _ in 0..100 {
                tour.change_parent(&mut rng);
                tour.check();
            }
        }
    }

    // An simple implementation of EulerTour tree to help test EulerTourTree
    struct SimpleTour<T: PartialEq> {
        values: Vec<T>,
    }

    impl<T: PartialEq + Debug> SimpleTour<T> {
        pub fn new(values: Vec<T>) -> Self {
            Self { values }
        }

        pub fn change_parent<R: Rng>(&mut self, mut rng: R) -> usize {
            let (start, end) = self.choose_subtree(&mut rng);
            assert!(start < end);
            assert_eq!(self.values[start], self.values[end]);
            loop {
                let to = self.choose_pos(&mut rng);
                if to < start {
                    self.values[to..end + 1].rotate_left(start - to);
                    return 0;
                }
                if to > end + 1 {
                    self.values[start..to].rotate_left(end + 1 - start);
                    return 0;
                }
            }
        }

        fn choose_subtree<R: Rng>(&mut self, mut rng: R) -> (usize, usize) {
            loop {
                let (start, end) = self.range(self.choose(&mut rng)).unwrap();
                if start != 0 || end != self.len() - 1 {
                    return (start, end);
                }
            }
        }

        fn range(&self, value: &T) -> Option<(usize, usize)> {
            self.position(value)
                .and_then(|start| self.rposition(value).map(|end| (start, end)))
        }

        fn position(&self, value: &T) -> Option<usize> {
            self.values.iter().position(|v| v == value)
        }

        fn rposition(&self, value: &T) -> Option<usize> {
            self.values.iter().rposition(|v| v == value)
        }

        fn choose<R: Rng>(&self, rng: R) -> &T {
            &self.values[self.choose_pos(rng)]
        }

        fn choose_pos<R: Rng>(&self, mut rng: R) -> usize {
            rng.gen_range(0, self.len())
        }

        fn len(&self) -> usize {
            self.values.len()
        }

        fn check(&self) {
            let mut stack = vec![];
            for value in &self.values {
                if stack.last().cloned() == Some(value) {
                    stack.pop();
                } else {
                    stack.push(value);
                }
            }
            assert!(stack.is_empty(), "{:?}", self.values);
        }
    }
}
