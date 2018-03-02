use fera::graph::prelude::*;
use fera::graph::choose::Choose;
use fera::graph::traverse::{Dfs, OnTraverseEvent, TraverseEvent};
use rand::Rng;
use rpds::HashTrieSet;

use std::rc::Rc;

pub type TourEdge = (u32, u32);

#[derive(Clone)]
pub struct EulerTourTree<G: WithEdge> {
    pub(crate) g: Rc<G>,
    pub(crate) tour_edges: HashTrieSet<Edge<G>>,
    pub(crate) vertices: Rc<Vec<Vertex<G>>>,
    segs: Rc<Vec<Rc<Segment>>>,
    len: usize,
}

impl<G> EulerTourTree<G>
where
    G: AdjacencyGraph + WithVertexIndexProp + Choose,
{
    #[inline(never)]
    pub fn new(g: Rc<G>, edges: &[Edge<G>]) -> Self {
        // TODO: avoid using this intermediary vector
        let mut tour = Vec::with_capacity(2 * (g.num_vertices() - 1));
        let mut stack = vec![];
        let mut tour_edges = HashTrieSet::new();
        let ends = |e| {
            let prop = g.vertex_index();
            let (a, b) = g.ends(e);
            (prop.get(a) as u32, prop.get(b) as u32)
        };
        g.spanning_subgraph(edges)
            .dfs(OnTraverseEvent(|evt| match evt {
                TraverseEvent::DiscoverEdge(e) => {
                    tour_edges = tour_edges.insert(e);
                    tour.push(ends(e));
                    stack.push(e);
                }
                TraverseEvent::FinishEdge(e) => {
                    let f = stack.pop().unwrap();
                    assert_eq!(e, f);
                    assert!(tour_edges.contains(&e));
                    tour.push(ends(g.reverse(e)));
                }
                _ => (),
            }))
            .run();
        Self::new_(g.clone(), &tour, tour_edges)
    }

    #[inline(never)]
    fn new_(g: Rc<G>, tour: &[TourEdge], tour_edges: HashTrieSet<Edge<G>>) -> Self {
        // TODO: remove vertices field
        let mut vertices = vec![g.vertices().next().unwrap(); g.num_vertices()];
        let index = g.vertex_index();
        for v in g.vertices() {
            vertices[index.get(v)] = v;
        }

        let nsqrt = (tour.len() as f64).sqrt().ceil() as usize;
        let mut tt = Self {
            g,
            tour_edges,
            segs: Rc::new(vec![]),
            vertices: Rc::new(vertices),
            len: tour.len(),
        };
        let segs = tour.chunks(nsqrt)
            .map(|t| Rc::new(tt.new_segment(t.into())))
            .collect();
        tt.segs = Rc::new(segs);
        tt
    }

    #[inline(never)]
    pub fn change_parent<R: Rng>(&mut self, mut rng: R) -> (Edge<G>, Edge<G>) {
        let ins = self.choose_non_tree_edge(&mut rng);
        let (a, b) = self.g.ends(ins);
        let a_sub = self.subtree(a);
        let b_sub = self.subtree(b);
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
        self.tour_edges = self.tour_edges.remove(&rem);
        self.tour_edges = self.tour_edges.insert(new);
        if y <= to {
            self.move_after(to, x, y);
        } else {
            assert!(to <= x);
            self.move_before(to, x, y);
        }
        rem
    }

    #[inline(never)]
    fn get_edge(&mut self, (i, j): (usize, usize)) -> Edge<G> {
        let (a, b) = self.segs[i].edges[j];
        self.g
            .edge_by_ends(self.vertices[a as usize], self.vertices[b as usize])
    }

    #[inline(never)]
    fn set_edge(&mut self, (i, j): (usize, usize), e: Edge<G>) {
        let g = self.g.clone();
        let ends = |e| {
            let prop = g.vertex_index();
            let (a, b) = g.ends(e);
            (prop.get(a) as u32, prop.get(b) as u32)
        };
        let mut edges = self.segs[i].edges.clone();
        edges[j] = ends(e);
        Rc::make_mut(&mut self.segs)[i] = Rc::new(self.new_segment(edges));
    }

    #[inline(never)]
    fn choose_non_tree_edge<R: Rng>(&self, rng: R) -> Edge<G> {
        self.g
            .choose_edge_iter(rng)
            .filter(|e| !self.contains(*e))
            .next()
            .unwrap()
    }

    #[inline(never)]
    pub fn contains(&self, e: Edge<G>) -> bool {
        self.tour_edges.contains(&e)
    }

    #[inline(never)]
    fn subtree(&self, v: Vertex<G>) -> Subtree {
        let v = self.g.vertex_index().get(v) as u32;
        let mut start = None;
        for (a, seg) in self.segs.iter().enumerate() {
            if let Some(b) = self.segment_position_source(seg, v) {
                start = Some((a, b));
                break;
            }
        }

        for (a, seg) in self.segs.iter().enumerate().rev() {
            if let Some(b) = self.segment_rposition_target(seg, v) {
                return Subtree::new(start.unwrap(), (a, b));
            }
        }

        unreachable!()
    }

    #[inline(never)]
    fn segment_position_source(&self, seg: &Segment, v: u32) -> Option<usize> {
        if self.segment_contains_vertex(seg, v) {
            for (i, edge) in seg.edges.iter().enumerate() {
                if edge.0 == v {
                    return Some(i);
                }
            }
        }
        None
    }

    #[inline(never)]
    fn segment_rposition_target(&self, seg: &Segment, v: u32) -> Option<usize> {
        if self.segment_contains_vertex(seg, v) {
            for (i, edge) in seg.edges.iter().enumerate().rev() {
                if edge.1 == v {
                    return Some(i);
                }
            }
        }
        None
    }

    #[inline(never)]
    fn segment_contains_vertex(&self, seg: &Segment, v: u32) -> bool {
        seg.pos
            .get(v as usize)
            .cloned()
            .map(|i| {
                if let Some(&(a, b)) = seg.edges.get(i) {
                    a == v || b == v
                } else {
                    false
                }
            })
            .unwrap_or(false)
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

    #[inline(never)]
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

    #[inline(never)]
    fn push<'a>(&self, segs: &mut Vec<Rc<Segment>>, last: Seg<'a>) {
        match last {
            Seg::Complete(seg) => segs.push(Rc::clone(seg)),
            Seg::Partial(values) => segs.push(Rc::new(self.new_segment(values.into()))),
        }
    }

    fn seg_iter(&self, start: (usize, usize), end: (usize, usize)) -> SegIter {
        SegIter::new(&self.segs, start, end)
    }

    #[inline(never)]
    fn new_segment(&self, edges: Vec<TourEdge>) -> Segment {
        let m = edges.iter().map(|&(a, b)| a.max(b)).max().unwrap();
        let mut pos = unsafe { vec_new_uninitialized(m as usize + 1) };
        for (i, &(a, b)) in edges.iter().enumerate() {
            pos[a as usize] = i;
            pos[b as usize] = i;
        }
        Segment { edges, pos }
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
    edges: Vec<TourEdge>,
    // TODO: use a bit vec?
    pos: Vec<usize>,
}

impl Segment {
    fn len(&self) -> usize {
        self.edges.len()
    }
}

unsafe fn vec_new_uninitialized<T>(n: usize) -> Vec<T> {
    use fera::ext::VecExt;

    let mut vec = Vec::new_uninitialized(n);
    vec.set_len(n);
    vec
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

// tests

#[cfg(test)]
impl<G> EulerTourTree<G>
where
    G: IncidenceGraph + WithVertexIndexProp + Choose,
{
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn get(&self, (i, j): (usize, usize)) -> TourEdge {
        self.segs[i].edges[j]
    }

    pub fn tour_edges(&self) -> Vec<TourEdge> {
        self.segs
            .iter()
            .flat_map(|seg| seg.edges.iter().cloned())
            .collect()
    }

    pub fn edges(&self) -> Vec<Edge<G>> {
        self.tour_edges()
            .into_iter()
            .map(|(a, b)| {
                self.g
                    .edge_by_ends(self.vertices[a as usize], self.vertices[b as usize])
            })
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

    fn segment_contains_source(
        tour: &EulerTourTree<CompleteGraph>,
        seg: &Segment,
        in_: &[u32],
        not_in: &[u32],
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
        let mut edges = vec![];
        edges.push(e(0, 1));
        edges.push(e(1, 2));
        edges.push(e(2, 3));
        edges.push(e(1, 4));
        edges.push(e(0, 5));
        edges.push(e(5, 6));
        edges.push(e(5, 7));
        edges.push(e(5, 8));

        let mut tour_edges = HashTrieSet::new();
        for i in 0..edges.len() {
            tour_edges = tour_edges.insert(edges[i]);
        }

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

        let tour = EulerTourTree::new_(g.clone(), &*tour, tour_edges);

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
            let mut tour = EulerTourTree::new(g.clone(), &*random_sp(&*g, &mut rng));
            tour.check();
            for _ in 0..100 {
                let old = tour.clone();
                let (ins, rem) = tour.change_parent(&mut rng);
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
