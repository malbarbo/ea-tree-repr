use fera::graph::algs::Kruskal;
use fera::graph::choose::Choose;
use fera::graph::prelude::*;
use rand::prelude::*;
use rand::XorShiftRng;

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
