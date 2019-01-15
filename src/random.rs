use fera::ext::VecExt;
use fera::fun::vec;
use fera::graph::algs::Kruskal;
use fera::graph::choose::Choose;
use fera::graph::prelude::*;
use rand::prelude::*;
use rand::XorShiftRng;

use std::collections::HashSet;

pub fn new_rng_with_seed(x: u32) -> XorShiftRng {
    let b1: u8 = ((x >> 24) & 0xff) as u8;
    let b2: u8 = ((x >> 16) & 0xff) as u8;
    let b3: u8 = ((x >> 8) & 0xff) as u8;
    let b4: u8 = (x & 0xff) as u8;
    XorShiftRng::from_seed([
        b1, b2, b3, b4, b1, b2, b3, b4, b1, b2, b3, b4, b1, b2, b3, b4,
    ])
}

pub fn new_rng() -> XorShiftRng {
    new_rng_with_seed(random())
}

pub fn random_sp<G: Graph + Choose, R: Rng>(g: &G, rng: R) -> Vec<Edge<G>> {
    g.kruskal().edges(g.random_walk(rng)).into_iter().collect()
}

pub fn random_sp_with_diameter<R: Rng>(
    g: &CompleteGraph,
    d: usize,
    rng: R,
) -> Vec<Edge<CompleteGraph>> {
    StaticGraph::new_random_tree_with_diameter(g.num_vertices() as u32, d as u32, rng)
        .unwrap()
        .edges_ends()
        .map(|(u, v)| g.edge_by_ends(u, v))
        .collect()
}


// FIXME: This does not work as expected
pub fn random_carrabs<R: Rng>(n: usize, i: usize, mut rng: R) -> StaticGraph {
    fn sort(a: usize, b: usize) -> (usize, usize) {
        if a > b {
            (a, b)
        } else {
            (b, a)
        }
    }
    assert!(n > 1);
    assert!(i > 0);
    let m = n - 1 + f64::floor(1.5 * (i as f64) * f64::ceil(f64::sqrt(n as f64))) as usize;

    let vertices = vec(0..n).shuffled_with(&mut rng);
    let mut builder = StaticGraph::builder(n, m);
    let mut edges = HashSet::new();
    for i in 1..vertices.len() {
        let u = *rng.choose(&vertices[0..i]).unwrap();
        let v = vertices[i];
        builder.add_edge(u, v);
        edges.insert(sort(u, v));
    }

    while edges.len() < m {
        let u = *rng.choose(&vertices).unwrap();
        let v = *rng.choose(&vertices).unwrap();
        if u == v || edges.contains(&sort(u, v)) {
            continue
        }
        builder.add_edge(u, v);
        edges.insert(sort(u, v));
    }

    builder.finalize()

    // StaticGraph::new_gnm_connected(n, m, rng).unwrap()
}


// FIXME: This does not work as expected
pub fn random_cerrone<R: Rng>(n: usize, mut rng: R) -> StaticGraph {
    fn sort(a: usize, b: usize) -> (usize, usize) {
        if a > b {
            (a, b)
        } else {
            (b, a)
        }
    }
    assert!(n >= 4);
    assert!(n % 2 == 0);
    let vertices = vec(0..n).shuffled_with(&mut rng);
    let mut builder = StaticGraph::builder(n, 4 * n);
    let mut edges = HashSet::new();
    let mut tours: Vec<[usize; 4]> = vec![];
    let mut a = vertices[0];
    let mut b = vertices[1];
    let mut c = vertices[2];
    let mut d = vertices[3];
    builder.add_edge(a, b);
    builder.add_edge(b, c);
    builder.add_edge(c, d);
    builder.add_edge(d, a);
    edges.insert(sort(a, b));
    edges.insert(sort(b, c));
    edges.insert(sort(c, d));
    edges.insert(sort(d, a));
    tours.push([a, b, c, d]);
    for i in (4..vertices.len()).step_by(2) {
        {
            let tour = rng.choose(&tours).unwrap();
            a = *rng.choose(tour).unwrap();
            b = a;
            while b == a {
                b = *rng.choose(tour).unwrap();
            }
        }
        c = vertices[i];
        d = vertices[i + 1];
        if !edges.contains(&sort(a, b)) {
            builder.add_edge(a, b);
            edges.insert(sort(a, b));
        }
        builder.add_edge(b, c);
        builder.add_edge(c, d);
        builder.add_edge(d, a);
        edges.insert(sort(b, c));
        edges.insert(sort(c, d));
        edges.insert(sort(d, a));
        tours.push([a, b, c, d]);
    }
    let m = edges.len();
    while edges.len() < (m + n / 2) {
        let a = *rng.choose(&vertices).unwrap();
        let b = *rng.choose(&vertices).unwrap();
        if a != b && !edges.contains(&sort(a, b)) {
            builder.add_edge(a, b);
            edges.insert(sort(a, b));
        }
    }
    builder.finalize()
}

#[derive(Default)]
pub struct Ranges {
    perms: Vec<Vec<u32>>,
}

impl Ranges {
    fn prepare(&mut self, n: usize) {
        if self.perms.len() <= n + 1 {
            self.perms.resize(n + 1, vec![]);
        }
        if self.perms[n].is_empty() {
            self.perms[n] = (0..n as u32).collect();
        }
    }

    pub fn sample_without_replacement<'a, R: 'a + Rng>(
        &'a mut self,
        n: usize,
        rng: R,
    ) -> SampleIter<R> {
        self.prepare(n);
        SampleIter {
            rng: rng,
            i: 0,
            perm: self.perms[n].as_mut_slice(),
        }
    }
}

pub struct SampleIter<'a, R> {
    rng: R,
    i: usize,
    perm: &'a mut [u32],
}

impl<'a, R: Rng> Iterator for SampleIter<'a, R> {
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.i < self.perm.len() {
            let j = self.rng.gen_range(self.i, self.perm.len());
            self.perm.swap(self.i, j);
            self.i += 1;
            Some(self.perm[self.i - 1] as usize)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fera::fun::vec;

    fn assert_perm(n: usize, mut values: Vec<usize>) {
        values.sort();
        for i in 0..n {
            assert_eq!(values[i], i);
        }
    }

    #[test]
    fn sample() {
        let mut rng = new_rng();
        let mut ranges = Ranges::default();
        assert_eq!(vec![0], vec(ranges.sample_without_replacement(1, &mut rng)));
        for n in 1..10 {
            for _ in 0..100 {
                assert_perm(n, vec(ranges.sample_without_replacement(n, &mut rng)));
            }
        }
    }
}
