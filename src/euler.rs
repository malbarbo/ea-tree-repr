use fera::ext::VecExt;
use fera::graph::choose::Choose;
use fera::graph::prelude::*;
use fera::graph::traverse::{Dfs, OnTraverseEvent, TraverseEvent};

use rand::Rng;

use std::rc::Rc;

#[derive(Clone)]
pub struct EulerTourTree<G: WithEdge> {
    g: Rc<G>,
    // TODO: remove vertices field
    vertices: Rc<[Vertex<G>]>,
    segs: Rc<Vec<Rc<Segment>>>,
    len: usize,
}

impl<G> EulerTourTree<G>
where
    G: AdjacencyGraph + WithVertexIndexProp + Choose,
{
    #[inline(never)]
    pub fn new(g: Rc<G>, edges: &[Edge<G>]) -> Self {
        let prop = g.vertex_index();
        let ends = |e| {
            let (a, b) = g.ends(e);
            (prop.get(a) as u32, prop.get(b) as u32)
        };
        // TODO: avoid using this intermediary vector
        let mut tour = Vec::with_capacity(2 * (g.num_vertices() - 1));
        let mut stack = vec![];
        g.spanning_subgraph(edges)
            .dfs(OnTraverseEvent(|evt| match evt {
                TraverseEvent::DiscoverEdge(e) => {
                    tour.push(ends(e));
                    stack.push(e);
                }
                TraverseEvent::FinishEdge(e) => {
                    assert_eq!(e, stack.pop().unwrap());
                    tour.push(ends(g.reverse(e)));
                }
                _ => (),
            }))
            .run();
        Self::new_(g.clone(), &tour)
    }

    #[inline(never)]
    fn new_(g: Rc<G>, tour: &[(u32, u32)]) -> Self {
        let mut vertices = vec![g.vertices().next().unwrap(); g.num_vertices()];
        let index = g.vertex_index();
        for v in g.vertices() {
            vertices[index.get(v)] = v;
        }

        let nsqrt = (tour.len() as f64).sqrt().ceil() as usize;
        let mut tt = Self {
            g,
            segs: Rc::new(vec![]),
            vertices: vertices.into(),
            len: tour.len(),
        };
        let segs = tour.chunks(nsqrt)
            .map(|t| {
                let source = t.iter().map(|st| st.0);
                let target = t.iter().map(|st| st.1);
                tt.new_segment(source.collect(), target.collect())
            })
            .collect();
        tt.segs = Rc::new(segs);
        tt
    }

    #[inline(never)]
    pub fn change_parent<R: Rng>(&mut self, mut rng: R) -> (Edge<G>, Edge<G>) {
        let (ins, a_sub, b_sub) = self.choose_non_tree_edge(&mut rng);
        let rem = if a_sub.contains(&b_sub) {
            // change b parent
            self.move_(ins, a_sub.end, b_sub)
        } else if b_sub.contains(&a_sub) {
            // change a parent
            let ins = self.g.reverse(ins);
            self.move_(ins, b_sub.end, a_sub)
        } else if rng.gen() {
            // change b parent
            self.move_(ins, a_sub.end, b_sub)
        } else {
            // change a parent
            let ins = self.g.reverse(ins);
            self.move_(ins, b_sub.end, a_sub)
        };
        (ins, rem)
    }

    pub fn graph(&self) -> &Rc<G> {
        &self.g
    }

    #[inline(never)]
    fn move_(&mut self, new: Edge<G>, to: (usize, usize), sub: Subtree) -> Edge<G> {
        let x = self.prev_pos(sub.start);
        let y = self.next_pos(sub.end);
        let rem = self.get_edge(x);
        self.set_edge(x, new);
        let new = self.g.reverse(new);
        self.set_edge(y, new);
        if y <= to {
            self.move_after(to, x, y);
        } else {
            assert!(to <= x);
            self.move_before(to, x, y);
        }
        rem
    }

    #[inline(never)]
    fn get_edge(&self, (i, j): (usize, usize)) -> Edge<G> {
        let (a, b) = self.segs[i].get(j);
        self.g.edge_by_ends(self.vertices[a], self.vertices[b])
    }

    #[inline(never)]
    fn set_edge(&mut self, (i, j): (usize, usize), e: Edge<G>) {
        let g = self.g.clone();
        let prop = g.vertex_index();
        let ends = |e| {
            let (a, b) = g.ends(e);
            (prop.get(a) as u32, prop.get(b) as u32)
        };
        let mut source = self.segs[i].source.clone();
        let mut target = self.segs[i].target.clone();
        let (a, b) = ends(e);
        source[j] = a;
        target[j] = b;
        Rc::make_mut(&mut self.segs)[i] = self.new_segment(source, target);
    }

    #[inline(never)]
    fn choose_non_tree_edge<R: Rng>(&self, rng: R) -> (Edge<G>, Subtree, Subtree) {
        self.g
            .choose_edge_iter(rng)
            .filter_map(|e| {
                let (a, b) = self.g.ends(e);
                let a = self.g.vertex_index().get(a) as u32;
                let a_start = self.subtree_start(a);
                if a_start != (0, 0) && self.get_edge(self.prev_pos(a_start)) == e {
                    None
                } else {
                    let b = self.g.vertex_index().get(b) as u32;
                    let b_start = self.subtree_start(b);
                    if b_start != (0, 0) && self.get_edge(self.prev_pos(b_start)) == e {
                        None
                    } else {
                        let a_sub = Subtree::new(a_start, self.subtree_end(a));
                        let b_sub = Subtree::new(b_start, self.subtree_end(b));
                        Some((e, a_sub, b_sub))
                    }
                }
            })
            .next()
            .unwrap()
    }

    #[inline(never)]
    pub fn contains(&self, e: Edge<G>) -> bool {
        let (a, b) = self.g.ends(e);
        let a_sub = self.subtree(a);
        if a_sub.start != (0, 0) && self.get_edge(self.prev_pos(a_sub.start)) == e {
            return true;
        } else {
            let b_sub = self.subtree(b);
            if b_sub.start != (0, 0) && self.get_edge(self.prev_pos(b_sub.start)) == e {
                return true;
            }
        }
        false
    }

    #[inline(never)]
    fn subtree(&self, v: Vertex<G>) -> Subtree {
        let v = self.g.vertex_index().get(v) as u32;
        Subtree::new(self.subtree_start(v), self.subtree_end(v))
    }

    #[inline(never)]
    fn subtree_start(&self, v: u32) -> (usize, usize) {
        for (a, seg) in self.segs.iter().enumerate() {
            if let Some(b) = self.segment_position_source(seg, v) {
                return (a, b);
            }
        }
        unreachable!()
    }

    #[inline(never)]
    fn subtree_end(&self, v: u32) -> (usize, usize) {
        if self.segs[0].source[0] == v {
            return self.prev_pos(self.end_pos());
        }

        for (a, seg) in self.segs.iter().enumerate().rev() {
            if let Some(b) = self.segment_rposition_source(seg, v) {
                return self.prev_pos((a, b));
            }
        }
        unreachable!()
    }

    #[inline(never)]
    fn segment_position_source(&self, seg: &Segment, v: u32) -> Option<usize> {
        if let Some(i) = seg.pos.get(v as usize) {
            let i = *i as usize;
            if Some(&v) == seg.source.get(i) {
                return Some(i);
            }
        }
        None
    }

    #[inline(never)]
    fn segment_rposition_source(&self, seg: &Segment, v: u32) -> Option<usize> {
        if let Some(i) = seg.pos.get(v as usize) {
            let i = *i as usize;
            if Some(&v) == seg.source.get(i) {
                return seg.source.iter().rposition(|&x| x == v);
            }
        }
        None
    }

    #[inline(never)]
    fn move_before(&mut self, to: (usize, usize), start: (usize, usize), end: (usize, usize)) {
        let mut segs = Vec::with_capacity(self.segs.len());
        {
            let segs = &mut segs;
            let to_next = self.next_pos(to);
            let end_next = self.next_pos(end);
            let mut iter = self.seg_iter(self.first_pos(), to_next);
            let mut last = iter.next().unwrap();
            last = self.extend(segs, last, iter);
            last = self.extend(segs, last, self.seg_iter(start, end_next));
            last = self.extend(segs, last, self.seg_iter(to_next, start));
            last = self.extend(segs, last, self.seg_iter(end_next, self.end_pos()));
            self.push(segs, last);
        }
        self.segs = Rc::new(segs);
    }

    #[inline(never)]
    fn move_after(&mut self, to: (usize, usize), start: (usize, usize), end: (usize, usize)) {
        let mut segs = Vec::with_capacity(self.segs.len());
        {
            let segs = &mut segs;
            let to_next = self.next_pos(to);
            let end_next = self.next_pos(end);
            let mut iter = self.seg_iter(self.first_pos(), start);
            let mut last = if let Some(mut last) = iter.next() {
                last = self.extend(segs, last, iter);
                self.extend(segs, last, self.seg_iter(end_next, to_next))
            } else {
                let mut iter = self.seg_iter(end_next, to_next);
                let mut last = iter.next().unwrap();
                self.extend(segs, last, iter)
            };
            last = self.extend(segs, last, self.seg_iter(start, end_next));
            last = self.extend(segs, last, self.seg_iter(to_next, self.end_pos()));
            self.push(segs, last);
        }
        self.segs = Rc::new(segs);
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
            Seg::Partial(source, target) => segs.push(self.new_segment(source, target)),
        }
    }

    fn seg_iter(&self, start: (usize, usize), end: (usize, usize)) -> SegIter {
        SegIter::new(&self.segs, start, end)
    }

    #[inline(never)]
    fn new_segment(&self, source: Vec<u32>, target: Vec<u32>) -> Rc<Segment> {
        let m_source = *source.iter().max().unwrap();
        let mut pos = unsafe { Vec::new_uninitialized(m_source as usize + 1) };
        for (i, &v) in source.iter().enumerate().rev() {
            unsafe {
                *pos.get_unchecked_mut(v as usize) = i as u32;
            }
        }
        Rc::new(Segment {
            source,
            target,
            pos,
        })
    }

    fn first_pos(&self) -> (usize, usize) {
        (0, 0)
    }

    fn end_pos(&self) -> (usize, usize) {
        (self.segs.len() - 1, self.segs.last().unwrap().len())
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
    // TODO: use a bit vec?
    source: Vec<u32>,
    target: Vec<u32>,
    pos: Vec<u32>,
}

impl Segment {
    fn len(&self) -> usize {
        self.source.len()
    }

    fn get(&self, i: usize) -> (usize, usize) {
        (self.source[i] as usize, self.target[i] as usize)
    }
}

#[derive(Clone, Debug)]
enum Seg<'a> {
    Partial(Vec<u32>, Vec<u32>),
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

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        if self.cur.0 < self.end.0 {
            let (i, j) = self.cur;
            self.cur = (i + 1, 0);
            if j == 0 {
                Some(Seg::Complete(&self.segs[i]))
            } else {
                Some(Seg::Partial(
                    self.segs[i].source[j..].into(),
                    self.segs[i].target[j..].into(),
                ))
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
                Some(Seg::Partial(
                    self.segs[i].source[a..b].into(),
                    self.segs[i].target[a..b].into(),
                ))
            }
        } else {
            None
        }
    }
}

// tests

#[cfg(test)]
impl<G> EulerTourTree<G>
where
    G: IncidenceGraph + WithVertexIndexProp + Choose,
{
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn get(&self, (i, j): (usize, usize)) -> (usize, usize) {
        self.segs[i].get(j)
    }

    pub fn tour_edges(&self) -> Vec<(usize, usize)> {
        let segs = &self.segs;
        (0..segs.len())
            .flat_map(|i| (0..segs[i].len()).map(move |j| segs[i].get(j)))
            .collect()
    }

    pub fn edges(&self) -> Vec<Edge<G>> {
        self.tour_edges()
            .into_iter()
            .map(|(a, b)| self.g.edge_by_ends(self.vertices[a], self.vertices[b]))
            .collect()
    }

    pub fn check(&self) {
        use std::collections::HashSet;
        use fera::graph::algs::{Paths, Trees};

        // check if tour is a path
        let mut last = self.get((0, 0)).0;
        for edge in self.tour_edges() {
            assert_eq!(last, edge.0);
            last = edge.1;
        }

        let edges_set: HashSet<_> = self.edges().into_iter().collect();
        assert!(self.g.spanning_subgraph(edges_set).is_tree());

        // check if the tour is an euler tour
        assert!(self.g.is_walk(self.edges()));
        let mut stack = vec![];
        for (a, b) in self.tour_edges() {
            let last = stack.last().cloned();
            if Some((a, b)) == last || Some((b, a)) == last {
                stack.pop();
            } else {
                stack.push((a, b));
            }
        }
        assert!(stack.is_empty(), "{:?}", self.segs);

        // check that there is no repeated edge
        let set: HashSet<_> = self.tour_edges().into_iter().collect();
        assert_eq!((self.g.num_vertices() - 1) * 2, set.len());

        assert_eq!(self.segs.iter().map(|s| s.len()).sum::<usize>(), self.len());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand;
    use random::random_sp;

    #[test]
    fn subtree() {
        let g = Rc::new(CompleteGraph::new(9));
        let e = |u: u32, v: u32| g.edge_by_ends(u, v);
        let mut edges = vec![];
        edges.push(e(0, 1));
        edges.push(e(1, 2));
        edges.push(e(2, 3));
        edges.push(e(1, 4));
        edges.push(e(0, 5));
        edges.push(e(5, 6));
        edges.push(e(5, 7));
        edges.push(e(5, 8));

        let tour = vec![
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 2),
            (2, 1),
            (1, 4),
            (4, 1),
            (1, 0),
            (0, 5),
            (5, 6),
            (6, 5),
            (5, 7),
            (7, 5),
            (5, 8),
            (8, 5),
            (5, 0),
        ];

        let tour = EulerTourTree::new_(g.clone(), &*tour);

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
            let mut tour = EulerTourTree::new(g.clone(), &*random_sp(&*g, &mut rng));
            tour.check();
            for _ in 0..100 {
                let old = tour.clone();
                let (ins, rem) = tour.change_parent(&mut rng);
                assert_ne!(ins, rem);
                tour.check();
                assert_ne!(old.tour_edges(), tour.tour_edges());
                let edges = tour.edges();
                assert!(edges.contains(&ins));
                assert!(!edges.contains(&rem));
                assert_ne!(old.edges(), tour.edges());
            }
        }
    }
}
