use fera::graph::choose::Choose;
use fera::graph::prelude::*;
use fera::graph::traverse::{Dfs, OnTraverseEvent, TraverseEvent};
use fera::fun::vec;

use rand::Rng;

use std::rc::Rc;

// TODO: use a shared pool, like we use buffer in ParentTree
use {bitset_acquire, bitset_release, Bitset};

const MAX_TRIES_CHANGE_ANY: usize = 5;

#[derive(Clone)]
pub struct EulerTourTree<G: WithEdge + WithVertexIndexProp> {
    g: Rc<G>,
    segs: Rc<Vec<Rc<Segment<G>>>>,
    nsqrt: usize,
    len: usize,
}

impl<G> EulerTourTree<G>
where
    G: IncidenceGraph + WithVertexIndexProp + Choose,
{
    pub fn new(g: Rc<G>, edges: &[Edge<G>]) -> Self {
        let len = 2 * (g.num_vertices() - 1);
        let nsqrt = ((len as f64).sqrt()).round() as usize;
        let mut tour = Self {
            g,
            segs: Rc::default(),
            nsqrt,
            len,
        };
        tour.set_edges(edges);
        tour.check();
        tour
    }

    fn edges_to_tour(&self, edges: &[Edge<G>]) -> Vec<(Vertex<G>, Vertex<G>)> {
        let mut tour = Vec::with_capacity(2 * (self.g.num_vertices() - 1));
        let mut stack = vec![];
        self.g
            .spanning_subgraph(edges)
            .dfs(OnTraverseEvent(|evt| match evt {
                TraverseEvent::DiscoverEdge(e) => {
                    tour.push(self.ends(e));
                    stack.push(e);
                }
                TraverseEvent::FinishEdge(e) => {
                    assert_eq!(Some(e), stack.pop());
                    tour.push(self.ends(self.g.reverse(e)));
                }
                _ => (),
            }))
            .run();
        assert!(stack.is_empty());
        tour
    }

    pub fn set_edges(&mut self, edges: &[Edge<G>]) {
        let edges = self.edges_to_tour(edges);
        let mut last = 0;
        let step = edges.len() as f64 / self.nsqrt as f64;
        let segs = vec(Linspace::new(self.nsqrt, step)
            .map(|to| {
                let seg = &edges[last..to];
                let source = seg.iter().map(|t| t.0).collect();
                let target = seg.iter().map(|t| t.1).collect();
                last = to;
                self.new_segment(source, target)
            }));
        self.segs = segs.into();
    }

    fn tour_edges(&self) -> Vec<(Vertex<G>, Vertex<G>)> {
        let segs = &self.segs;
        (0..segs.len())
            .flat_map(|i| (0..segs[i].len()).map(move |j| segs[i].get(j)))
            .collect()
    }

    pub fn edges(&self) -> Vec<Edge<G>> {
        self.tour_edges()
            .into_iter()
            .map(|(a, b)| self.g.edge_by_ends(a, b))
            .collect()
    }

    pub fn graph(&self) -> &Rc<G> {
        &self.g
    }

    pub fn contains(&self, e: Edge<G>) -> bool {
        let (a, b) = self.ends(e);
        let start = self.subtree_start(a);
        start != (0, 0) && self.get_edge(self.prev_pos(start)) == e || {
            let start = self.subtree_start(b);
            start != (0, 0) && self.get_edge(self.prev_pos(start)) == e
        }
    }

    pub fn change_pred<R: Rng>(&mut self, mut rng: R) -> (Edge<G>, Edge<G>) {
        let (ins, a_sub, b_sub) = self.choose_non_tree_edge(&mut rng);
        let (ins, to, sub) = if a_sub.contains(&b_sub) {
            // change b parent
            (ins, a_sub.end, b_sub)
        } else if b_sub.contains(&a_sub) {
            // change a parent
            (self.g.reverse(ins), b_sub.end, a_sub)
        } else if rng.gen() {
            // change b parent
            (ins, a_sub.end, b_sub)
        } else {
            // change a parent
            (self.g.reverse(ins), b_sub.end, a_sub)
        };

        let rem = self.get_edge(self.prev_pos(sub.start));

        if sub.start <= to {
            self.move_after(ins, to, sub.start, sub.start, sub.end);
        } else {
            self.move_before(ins, to, sub.start, sub.start, sub.end);
        }

        self.check();

        (ins, rem)
    }

    #[cfg_attr(feature = "cargo-clippy", allow(never_loop))]
    pub fn change_any<R: Rng>(&mut self, mut rng: R) -> (Edge<G>, Edge<G>) {
        let sub = self.choose_subtree(&mut rng);
        assert_ne!((0, 0), sub.start);
        let rem = self.get_edge(self.prev_pos(sub.start));
        let ins;
        let to;
        let sub_new_root;
        if self.subtree_len(sub) <= self.len / 2 {
            // TODO: extract to a function
            'out1: loop {
                let x = self.choose_subtree_vertex(sub, &mut rng);
                for e in self
                    .g
                    .choose_out_edge_iter(x, &mut rng)
                    .take(MAX_TRIES_CHANGE_ANY)
                {
                    if e == rem {
                        continue;
                    }
                    let y = self.g.target(e);
                    let yi = self.subtree_end(y);
                    if sub.contains_pos(yi) {
                        continue;
                    }
                    ins = self.g.reverse(e);
                    to = yi;
                    sub_new_root = self.subtree_start(x);
                    break 'out1;
                }
                return self.change_pred(rng);
            }
        } else {
            // TODO: extract to a function
            'out2: loop {
                let x = self.choose_non_subtree_vertex(sub, &mut rng);
                for e in self
                    .g
                    .choose_out_edge_iter(x, &mut rng)
                    .take(MAX_TRIES_CHANGE_ANY)
                {
                    if e == rem || x == self.source(sub.start) {
                        continue;
                    }
                    let y = self.g.target(e);
                    let yi = self.subtree_start(y);
                    if !sub.contains_pos(yi) {
                        continue;
                    }
                    ins = e;
                    to = self.subtree_end(x);
                    sub_new_root = yi;
                    break 'out2;
                }
                return self.change_pred(rng);
            }
        }

        if to <= sub.start {
            self.move_before(ins, to, sub.start, sub_new_root, sub.end);
        } else {
            self.move_after(ins, to, sub.start, sub_new_root, sub.end);
        }

        self.check();

        (ins, rem)
    }

    fn move_before(
        &mut self,
        new: Edge<G>,
        to: (usize, usize),
        start: (usize, usize),
        root: (usize, usize),
        end: (usize, usize),
    ) {
        // Input:
        //    1     2       3     4       5
        // +---------------------------------+
        // |     |     |a|     |     |b|     |
        // +---------------------------------+
        //       ^       ^     ^     ^
        //      to      start root  end
        //
        // Output:
        //   1        4     3       2     5
        // +---------------------------------+
        // |     |a|     |     |b|     |     |
        // +---------------------------------+
        //
        // where x = removed edge
        //       y = reverse(removed edge)
        //       a = inserted edge
        //       b = reverse(inserted edge)
        assert!(to <= start);
        assert!(start <= root);
        assert!(root <= self.next_pos(end));
        let (u, v) = self.ends(new);
        let mut segs = Vec::with_capacity(self.segs.len());
        {
            let segs = &mut segs;
            let to_next = self.next_pos(to);
            let end_next = self.next_pos(end);
            let end_next_next = self.next_pos(end_next);
            let start_prev = self.prev_pos(start);
            let mut last = Seg::default();
            // 1
            last = self.extend(segs, last, self.seg_iter(self.first_pos(), to_next));
            // new
            last = self.push_edge(segs, last, &u, &v);
            // 4
            last = self.extend(segs, last, self.seg_iter(root, end_next));
            // 3
            last = self.extend(segs, last, self.seg_iter(start, root));
            // new reversed
            last = self.push_edge(segs, last, &v, &u);
            // 2
            last = self.extend(segs, last, self.seg_iter(to_next, start_prev));
            // 5
            last = self.extend(segs, last, self.seg_iter(end_next_next, self.end_pos()));
            self.push_last(segs, last);
        }
        self.segs = segs.into();
    }

    fn move_after(
        &mut self,
        new: Edge<G>,
        to: (usize, usize),
        start: (usize, usize),
        root: (usize, usize),
        end: (usize, usize),
    ) {
        // Input:
        //    1       2     3       4     5
        // +---------------------------------+
        // |     |x|     |     |y|     |     |
        // +---------------------------------+
        //         ^     ^     ^       ^
        //       start  root  end      to
        //
        // Output:
        //    1     4       3     2       5
        // +---------------------------------+
        // |     |     |a|     |     |b|     |
        // +---------------------------------+
        //
        // where x = removed edge
        //       y = reverse(removed edge)
        //       a = inserted edge
        //       b = reverse(inserted edge)
        assert!(end <= to);
        assert!(start <= root);
        assert!(root <= self.next_pos(end));
        let (u, v) = self.ends(new);
        let mut segs = Vec::with_capacity(self.segs.len());
        {
            let segs = &mut segs;
            let to_next = self.next_pos(to);
            let end_next = self.next_pos(end);
            let end_next_next = self.next_pos(end_next);
            let start_prev = self.prev_pos(start);
            let mut last = Seg::default();
            // 1
            last = self.extend(segs, last, self.seg_iter(self.first_pos(), start_prev));
            // 4
            last = self.extend(segs, last, self.seg_iter(end_next_next, to_next));
            // new
            last = self.push_edge(segs, last, &u, &v);
            // 3
            last = self.extend(segs, last, self.seg_iter(root, end_next));
            // 2
            last = self.extend(segs, last, self.seg_iter(start, root));
            // new reversed
            last = self.push_edge(segs, last, &v, &u);
            // 5
            last = self.extend(segs, last, self.seg_iter(to_next, self.end_pos()));
            self.push_last(segs, last);
        }
        self.segs = segs.into();
    }

    fn choose_non_tree_edge<R: Rng>(&self, rng: R) -> (Edge<G>, Subtree, Subtree) {
        for e in self.g.choose_edge_iter(rng) {
            let (a, b) = self.ends(e);
            let a_start = self.subtree_start(a);
            if a_start != (0, 0) && self.source(self.prev_pos(a_start)) == b {
                continue;
            }
            let b_start = self.subtree_start(b);
            if b_start != (0, 0) && self.source(self.prev_pos(b_start)) == a {
                continue;
            }
            let a_sub = Subtree::new(a_start, self.subtree_end(a));
            let b_sub = Subtree::new(b_start, self.subtree_end(b));
            return (e, a_sub, b_sub);
        }
        unreachable!()
    }

    fn choose_subtree<R: Rng>(&self, mut rng: R) -> Subtree {
        let root = self.source((0, 0));
        let v = self
            .g
            .choose_vertex_iter(&mut rng)
            .find(|v| *v != root)
            .unwrap();
        self.subtree(v)
    }

    fn choose_subtree_vertex<R: Rng>(&self, tree: Subtree, mut rng: R) -> Vertex<G> {
        // TODO: check if this method has some tendency. Every subtree should have the same
        // probability of being chosen
        if tree.start > tree.end {
            self.source(tree.start)
        } else {
            let start = self.pos_to_index(tree.start);
            let end = self.pos_to_index(tree.end) + 1;
            let i = rng.gen_range(start, end);
            self.target(self.index_to_pos(i))
        }
    }

    fn choose_non_subtree_vertex<R: Rng>(&self, tree: Subtree, mut rng: R) -> Vertex<G> {
        // TODO: check if this method has some tendency. Every subtree should have the same
        // probability of being chosen
        let start = self.pos_to_index(tree.start);
        let end = self.pos_to_index(tree.end) + 1;
        let mut i = rng.gen_range(0, self.len - (end - start));
        if i >= start {
            i += end - start;
        }
        assert!(!tree.contains_pos(self.index_to_pos(i)));
        self.source(self.index_to_pos(i))
    }

    fn subtree_len(&self, tree: Subtree) -> usize {
        if self.next_pos(tree.end) == tree.start {
            0
        } else if tree.start.0 == tree.end.0 {
            tree.end.1 + 1 - tree.start.1
        } else {
            let mut len = (self.segs[tree.start.0].len() - tree.start.1) + tree.end.1 + 1;
            for i in (tree.start.0 + 1)..tree.end.0 {
                len += self.segs[i].len();
            }
            len
        }
    }

    fn subtree(&self, v: Vertex<G>) -> Subtree {
        Subtree::new(self.subtree_start(v), self.subtree_end(v))
    }

    fn subtree_start(&self, v: Vertex<G>) -> (usize, usize) {
        for (a, seg) in self.segs.iter().enumerate() {
            if let Some(b) = self.segment_position_source(seg, v) {
                return (a, b);
            }
        }
        unreachable!()
    }

    fn subtree_end(&self, v: Vertex<G>) -> (usize, usize) {
        // special case for the root
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

    fn segment_position_source(&self, seg: &Segment<G>, v: Vertex<G>) -> Option<usize> {
        if seg.contains(v) {
            seg.source.iter().position(|x| *x == v)
        } else {
            None
        }
    }

    fn segment_rposition_source(&self, seg: &Segment<G>, v: Vertex<G>) -> Option<usize> {
        if seg.contains(v) {
            seg.source.iter().rposition(|x| *x == v)
        } else {
            None
        }
    }

    fn extend<'a>(
        &self,
        segs: &mut Vec<Rc<Segment<G>>>,
        mut last: Seg<'a, G>,
        iter: SegIter<'a, G>,
    ) -> Seg<'a, G> {
        for seg in iter {
            match last {
                Seg::Complete(x) => {
                    segs.push(Rc::clone(x));
                    last = seg;
                }
                Seg::Partial(mut source, mut target) => match seg {
                    Seg::Partial(n_source, n_target) => {
                        source.extend(n_source);
                        target.extend(n_target);
                        last = Seg::Partial(source, target);
                    }
                    Seg::Complete(x) => {
                        let len: usize = source.iter().map(|s| s.len()).sum();
                        if len < self.min_seg_len() {
                            source.push(&x.source[..]);
                            target.push(&x.target[..]);
                            last = Seg::Partial(source, target);
                        } else {
                            self.push_source_target(segs, source, target);
                            last = Seg::Complete(x);
                        }
                    }
                },
            }
        }
        last
    }

    fn push_last<'a>(&self, segs: &mut Vec<Rc<Segment<G>>>, last: Seg<'a, G>) {
        match last {
            Seg::Complete(seg) => segs.push(Rc::clone(seg)),
            Seg::Partial(source, target) => {
                if source.is_empty() {
                    return;
                }
                let len: usize = source.iter().map(|s| s.len()).sum();
                if len < self.min_seg_len() {
                    let seg = Rc::try_unwrap(segs.pop().unwrap());
                    let (mut ss, mut tt) = match &seg {
                        &Ok(ref seg) => (vec![&seg.source[..]], vec![&seg.target[..]]),
                        &Err(ref seg) => (vec![&seg.source[..]], vec![&seg.target[..]]),
                    };
                    ss.extend(source);
                    tt.extend(target);
                    self.push_source_target(segs, ss, tt);
                } else {
                    self.push_source_target(segs, source, target);
                }
            }
        }
    }

    fn push_source_target(
        &self,
        segs: &mut Vec<Rc<Segment<G>>>,
        source: Vec<&[Vertex<G>]>,
        target: Vec<&[Vertex<G>]>,
    ) {
        fn flatten<T: Copy>(x: Vec<&[T]>) -> Vec<T> {
            let len = x.iter().map(|s| s.len()).sum();
            let mut vec = Vec::with_capacity(len);
            for slice in x {
                vec.extend_from_slice(slice);
            }
            vec
        }
        let len: usize = source.iter().map(|s| s.len()).sum();
        if len <= self.max_seg_len() {
            segs.push(self.new_segment(flatten(source), flatten(target)));
        } else {
            let count = len / self.nsqrt;
            let step = len as f64 / count as f64;
            let mut i = 0;
            let mut j = 0;
            let mut s = 0;
            for t in Linspace::new(count, step) {
                let new_len = t - s;
                let mut new_source = Vec::with_capacity(new_len);
                let mut new_target = Vec::with_capacity(new_len);
                loop {
                    if source[i][j..].len() + new_source.len() <= new_len {
                        new_source.extend_from_slice(&source[i][j..]);
                        new_target.extend_from_slice(&target[i][j..]);
                        i += 1;
                        j = 0;
                    } else {
                        let k = j + new_len - new_source.len();
                        new_source.extend_from_slice(&source[i][j..k]);
                        new_target.extend_from_slice(&target[i][j..k]);
                        assert_eq!(new_len, new_source.len());
                        j = k;
                    }
                    if new_source.len() == new_len {
                        break;
                    }
                }
                segs.push(self.new_segment(new_source, new_target));
                s = t;
            }
        }
    }

    fn push_edge<'a>(
        &self,
        segs: &mut Vec<Rc<Segment<G>>>,
        last: Seg<'a, G>,
        s: &'a Vertex<G>,
        t: &'a Vertex<G>,
    ) -> Seg<'a, G> {
        match last {
            Seg::Complete(seg) => {
                segs.push(Rc::clone(seg));
                Seg::Partial(vec![ref_slice(s)], vec![ref_slice(t)])
            }
            Seg::Partial(mut source, mut target) => {
                source.push(ref_slice(s));
                target.push(ref_slice(t));
                Seg::Partial(source, target)
            }
        }
    }

    fn seg_iter(&self, start: (usize, usize), end: (usize, usize)) -> SegIter<G> {
        SegIter::new(&self.segs, start, end)
    }

    fn new_segment(&self, source: Vec<Vertex<G>>, target: Vec<Vertex<G>>) -> Rc<Segment<G>> {
        assert_ne!(0, source.len());
        assert_eq!(source.len(), target.len());
        let mut bitset = bitset_acquire(self.g.num_vertices() + 1);
        for &v in &source {
            bitset.set(self.g.vertex_index().get(v), true);
        }
        Rc::new(Segment {
            index: self.g.vertex_index(),
            source,
            target,
            bitset,
        })
    }

    fn get_edge(&self, (i, j): (usize, usize)) -> Edge<G> {
        let (a, b) = self.segs[i].get(j);
        self.g.edge_by_ends(a, b)
    }

    fn ends(&self, e: Edge<G>) -> (Vertex<G>, Vertex<G>) {
        self.g.ends(e)
    }

    fn pos_to_index(&self, (i, j): (usize, usize)) -> usize {
        assert!(j < self.segs[i].len());
        self.segs[..i].iter().map(|s| s.len()).sum::<usize>() + j
    }

    fn index_to_pos(&self, mut index: usize) -> (usize, usize) {
        assert!(index < self.len);
        let mut i = 0;
        while index >= self.segs[i].len() {
            index -= self.segs[i].len();
            i += 1;
        }
        (i, index)
    }

    fn source(&self, (i, j): (usize, usize)) -> Vertex<G> {
        self.segs[i].source[j]
    }

    fn target(&self, (i, j): (usize, usize)) -> Vertex<G> {
        self.segs[i].target[j]
    }

    fn max_seg_len(&self) -> usize {
        2 * self.nsqrt
    }

    fn min_seg_len(&self) -> usize {
        self.nsqrt / 2
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

    #[cfg(not(test))]
    fn check(&self) {}
}

pub fn ref_slice<A>(s: &A) -> &[A] {
    unsafe { ::std::slice::from_raw_parts(s, 1) }
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

    fn contains_pos(&self, pos: (usize, usize)) -> bool {
        self.start <= pos && pos <= self.end
    }
}

struct Segment<G: WithVertexIndexProp> {
    index: VertexIndexProp<G>,
    source: Vec<Vertex<G>>,
    target: Vec<Vertex<G>>,
    bitset: Bitset,
}

impl<G: WithVertexIndexProp> Segment<G> {
    #[inline]
    fn len(&self) -> usize {
        self.source.len()
    }

    #[inline]
    fn get(&self, i: usize) -> (Vertex<G>, Vertex<G>) {
        (self.source[i], self.target[i])
    }

    #[inline]
    fn contains(&self, v: Vertex<G>) -> bool {
        self.bitset[self.index.get(v)]
    }

    fn reset_bitset(&mut self) {
        for &v in &self.source {
            self.bitset.set(self.index.get(v), false);
        }
    }
}

impl<G: WithVertexIndexProp> Drop for Segment<G> {
    fn drop(&mut self) {
        self.reset_bitset();
        bitset_release(::std::mem::replace(&mut self.bitset, Bitset::default()));
    }
}

#[derive(Clone)]
enum Seg<'a, G: 'a + WithVertexIndexProp> {
    Partial(Vec<&'a [Vertex<G>]>, Vec<&'a [Vertex<G>]>),
    Complete(&'a Rc<Segment<G>>),
}

impl<'a, G: 'a + WithVertexIndexProp> Default for Seg<'a, G> {
    fn default() -> Self {
        Seg::Partial(vec![], vec![])
    }
}

struct SegIter<'a, G: 'a + WithVertexIndexProp> {
    segs: &'a [Rc<Segment<G>>],
    cur: (usize, usize),
    end: (usize, usize),
}

impl<'a, G: 'a + WithVertexIndexProp> SegIter<'a, G> {
    fn new(segs: &'a [Rc<Segment<G>>], cur: (usize, usize), end: (usize, usize)) -> Self {
        SegIter { segs, cur, end }
    }
}

impl<'a, G: WithVertexIndexProp> Iterator for SegIter<'a, G> {
    type Item = Seg<'a, G>;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        if self.cur.0 < self.end.0 {
            let (i, j) = self.cur;
            self.cur = (i + 1, 0);
            if j == 0 {
                Some(Seg::Complete(&self.segs[i]))
            } else {
                Some(Seg::Partial(
                    vec![&self.segs[i].source[j..]],
                    vec![&self.segs[i].target[j..]],
                ))
            }
        } else if self.cur.0 == self.end.0 {
            // same segment
            let (i, a) = self.cur;
            let b = self.end.1;
            self.cur = (i + 1, 0);
            if a >= b {
                None
            } else if a == 0 && b == self.segs[i].len() {
                Some(Seg::Complete(&self.segs[i]))
            } else {
                Some(Seg::Partial(
                    vec![&self.segs[i].source[a..b]],
                    vec![&self.segs[i].target[a..b]],
                ))
            }
        } else {
            None
        }
    }
}

pub struct Linspace {
    last: usize,
    cur: usize,
    step: f64,
}

impl Linspace {
    fn new(last: usize, step: f64) -> Self {
        Self { last, cur: 0, step }
    }
}

impl Iterator for Linspace {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        if self.cur < self.last {
            self.cur += 1;
            let value = (self.cur as f64) * self.step;
            Some(value.round() as usize)
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

    pub fn get(&self, (i, j): (usize, usize)) -> (Vertex<G>, Vertex<G>) {
        self.segs[i].get(j)
    }

    pub fn check(&self) {
        use fera::graph::algs::{Paths, Trees};
        use std::collections::HashSet;

        // check seg lens
        for seg in self.segs.iter() {
            assert!(seg.len() >= self.min_seg_len());
            assert!(seg.len() <= self.max_seg_len());
        }

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
        assert!(stack.is_empty());

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
        //       0
        //     /   \
        //    1     5
        //  / |   / | \
        // 2  4  6  7  8
        // |
        // 3
        edges.push(e(0, 1));
        edges.push(e(1, 2));
        edges.push(e(2, 3));
        edges.push(e(1, 4));
        edges.push(e(0, 5));
        edges.push(e(5, 6));
        edges.push(e(5, 7));
        edges.push(e(5, 8));

        let tour = EulerTourTree::new(g.clone(), &edges);

        assert_eq!(16, tour.subtree_len(tour.subtree(0)));
        assert_eq!(6, tour.subtree_len(tour.subtree(1)));
        assert_eq!(2, tour.subtree_len(tour.subtree(2)));
        assert_eq!(0, tour.subtree_len(tour.subtree(3)));
        assert_eq!(0, tour.subtree_len(tour.subtree(4)));
        assert_eq!(6, tour.subtree_len(tour.subtree(5)));
        assert_eq!(0, tour.subtree_len(tour.subtree(6)));
        assert_eq!(0, tour.subtree_len(tour.subtree(7)));
        assert_eq!(0, tour.subtree_len(tour.subtree(8)));

        assert_eq!(Subtree::new((0, 0), (3, 3)), tour.subtree(0));
        assert_eq!(Subtree::new((0, 1), (1, 2)), tour.subtree(1));
        assert_eq!(Subtree::new((0, 2), (0, 3)), tour.subtree(2));
        assert_eq!(Subtree::new((0, 3), (0, 2)), tour.subtree(3));
        assert_eq!(Subtree::new((1, 2), (1, 1)), tour.subtree(4));
        assert_eq!(Subtree::new((2, 1), (3, 2)), tour.subtree(5));
        assert_eq!(Subtree::new((2, 2), (2, 1)), tour.subtree(6));
        assert_eq!(Subtree::new((3, 0), (2, 3)), tour.subtree(7));
        assert_eq!(Subtree::new((3, 2), (3, 1)), tour.subtree(8));

        for &(index, pos) in &[
            (0, (0, 0)),
            (1, (0, 1)),
            (2, (0, 2)),
            (3, (0, 3)),
            (4, (1, 0)),
            (5, (1, 1)),
            (6, (1, 2)),
            (7, (1, 3)),
            (8, (2, 0)),
            (9, (2, 1)),
            (10, (2, 2)),
            (11, (2, 3)),
            (12, (3, 0)),
            (13, (3, 1)),
            (14, (3, 2)),
            (15, (3, 3)),
        ] {
            assert_eq!(index, tour.pos_to_index(pos));
            assert_eq!(pos, tour.index_to_pos(index));
        }
    }

    #[test]
    fn change_pred() {
        let mut rng = rand::XorShiftRng::new_unseeded();
        for n in 5..30 {
            let g = Rc::new(CompleteGraph::new(n));
            let mut tour = EulerTourTree::new(g.clone(), &*random_sp(&*g, &mut rng));
            tour.check();
            for _ in 0..100 {
                let old = tour.clone();
                let (ins, rem) = tour.change_pred(&mut rng);
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

    #[test]
    fn change_any() {
        let mut rng = rand::XorShiftRng::new_unseeded();
        for n in 5..30 {
            let g = Rc::new(CompleteGraph::new(n));
            let mut tour = EulerTourTree::new(g.clone(), &*random_sp(&*g, &mut rng));
            tour.check();
            for _ in 0..100 {
                let old = tour.clone();
                let (ins, rem) = tour.change_any(&mut rng);
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
