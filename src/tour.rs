use rand::Rng;
use std::fmt::{self, Debug};
use std::rc::Rc;

#[derive(Clone, Debug)]
pub struct Tour {
    segs: Rc<Vec<Rc<Segment>>>,
    len: usize,
}

impl Tour {
    pub fn new(values: &[TourEdge]) -> Self {
        let nsqrt = (values.len() as f64).sqrt().ceil() as usize;
        let segs = values
            .chunks(nsqrt)
            .map(|t| Rc::new(Segment::new(t.into())))
            .collect();
        Self {
            segs: Rc::new(segs),
            len: values.len(),
        }
    }

    pub fn change_parent<R: Rng>(&mut self, mut rng: R) -> TourEdge {
        let (start, end) = self.choose_subtree(&mut rng);
        let e = self.get(start);
        assert!(start < end);
        assert_eq!(self.get(start), self.get(end));
        loop {
            // TODO: direct selection
            let to = self.choose_pos(&mut rng);
            // TODO: create a function
            if start != (0, 0) && to < self.prev_pos(start) {
                let mut new = Self {
                    segs: Rc::new(Vec::with_capacity(self.segs.len())),
                    len: self.len(),
                };
                {
                    let mut iter = self.seg_iter(self.first_pos(), self.next_pos(to));
                    let mut last = iter.next().unwrap();
                    last = new.extend(last, iter);
                    last = new.extend(last, self.seg_iter(start, self.next_pos(end)));
                    last = new.extend(last, self.seg_iter(self.next_pos(to), start));
                    last = new.extend(last, self.seg_iter(self.next_pos(end), self.nlast_pos()));
                    new.push(last);
                }
                *self = new;
                break;
            }
            // TODO: create a function
            if to > end {
                let mut new = Self {
                    segs: Rc::new(Vec::with_capacity(self.segs.len())),
                    len: self.len(),
                };
                {
                    let mut iter = self.seg_iter(self.first_pos(), start);
                    let mut last = if let Some(mut last) = iter.next() {
                        last = new.extend(last, iter);
                        new.extend(last, self.seg_iter(self.next_pos(end), self.next_pos(to)))
                    } else {
                        let mut iter = self.seg_iter(self.next_pos(end), self.next_pos(to));
                        let mut last = iter.next().unwrap();
                        new.extend(last, iter)
                    };
                    last = new.extend(last, self.seg_iter(start, self.next_pos(end)));
                    last = new.extend(last, self.seg_iter(self.next_pos(to), self.nlast_pos()));
                    new.push(last);
                }
                *self = new;
                break;
            }
        }
        e
    }

    pub fn range(&self, e: TourEdge) -> Option<((usize, usize), (usize, usize))> {
        self.position(e)
            .and_then(|start| self.rposition(e).map(|end| (start, end)))
    }

    pub fn get(&self, (i, j): (usize, usize)) -> TourEdge {
        self.segs[i].values[j]
    }

    pub fn get_next(&self, p: (usize, usize)) -> TourEdge {
        self.get(self.next_pos(p))
    }

    pub fn get_prev(&self, p: (usize, usize)) -> TourEdge {
        self.get(self.prev_pos(p))
    }

    fn extend<'a>(&mut self, mut last: Seg<'a>, iter: SegIter<'a>) -> Seg<'a> {
        for seg in iter {
            self.push(last);
            last = seg;
        }
        last
    }

    fn push<'a>(&mut self, last: Seg<'a>) {
        match last {
            Seg::Complete(seg) => Rc::make_mut(&mut self.segs).push(seg),
            Seg::Partial(values) => {
                Rc::make_mut(&mut self.segs).push(Rc::new(Segment::new(values.into())))
            }
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

    fn nlast_pos(&self) -> (usize, usize) {
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

    #[cfg(test)]
    pub fn to_vec(&self) -> Vec<TourEdge> {
        self.segs
            .iter()
            .flat_map(|seg| seg.values.iter().cloned())
            .collect()
    }

    #[cfg(test)]
    pub fn check(&self) {
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
        assert!(stack.is_empty(), "{:?}", self.segs);
        for (i, count) in seen.into_iter().enumerate() {
            assert_eq!(1, count, "id: {}, {:?}", i, self.segs);
        }
        assert_eq!(self.segs.iter().map(|s| s.len()).sum::<usize>(), self.len());
    }
}

#[derive(Clone, Debug)]
struct Segment {
    values: Vec<TourEdge>,
    pos: Vec<usize>,
}

impl Segment {
    fn new(values: Vec<TourEdge>) -> Self {
        use fera::ext::VecExt;
        let m = values.iter().map(|e| e.index()).max().unwrap();
        let mut pos = unsafe { Vec::new_uninitialized(m + 1) };
        unsafe {
            pos.set_len(m + 1);
        }
        for (i, &value) in values.iter().enumerate() {
            pos[value.index()] = i;
        }
        Self { values, pos }
    }

    fn len(&self) -> usize {
        self.values.len()
    }

    fn contains(&self, e: TourEdge) -> bool {
        let index = e.index();
        self.pos
            .get(index)
            .map(|&i| self.values.get(i) == Some(&e))
            .unwrap_or(false)
    }

    fn position(&self, e: TourEdge) -> Option<usize> {
        if self.contains(e) {
            let p = self.values.iter().position(|v| *v == e);
            debug_assert!(p.is_some());
            p
        } else {
            None
        }
    }

    fn rposition(&self, e: TourEdge) -> Option<usize> {
        if self.contains(e) {
            let p = self.values.iter().rposition(|v| *v == e);
            debug_assert!(p.is_some());
            p
        } else {
            None
        }
    }
}

#[derive(Debug)]
enum Seg<'a> {
    Partial(&'a [TourEdge]),
    Complete(Rc<Segment>),
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
                Some(Seg::Complete(self.segs[i].clone()))
            } else {
                Some(Seg::Partial(&self.segs[i].values[j..]))
            }
        } else if self.cur.0 == self.end.0 {
            // same segment
            let (i, a) = self.cur;
            let b = self.end.1;
            self.cur = (i + 1, 0);
            if a == b {
                None
            } else if a == 0 && b == self.segs[i].len() {
                Some(Seg::Complete(self.segs[i].clone()))
            } else {
                Some(Seg::Partial(&self.segs[i].values[a..b]))
            }
        } else {
            None
        }
    }
}

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
    use super::*;
    use rand;

    #[test]
    fn check() {
        let mut rng = rand::XorShiftRng::new_unseeded();
        for n in 5..30 {
            let v: Vec<_> = (0..n)
                .map(TourEdge::new)
                .chain((0..n).rev().map(TourEdge::new_reversed))
                .collect();
            let mut tour = Tour::new(&v);
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
                    self.values[to..end + 1].rotate(start - to);
                    return 0;
                }
                if to > end + 1 {
                    self.values[start..to].rotate(end + 1 - start);
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
