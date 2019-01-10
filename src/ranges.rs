use rand::distributions::{IndependentSample, Range};
use rand::Rng;

use std::rc::Rc;

#[derive(Clone)]
pub struct Ranges(Rc<[Range<usize>]>);

impl Ranges {
    pub fn new(n: usize) -> Self {
        let mut r = Vec::with_capacity(n);
        r.push(Range::new(0, 1));
        for i in 1..n {
            r.push(Range::new(0, i));
        }
        Ranges(r.into())
    }

    #[inline]
    pub fn gen<R>(&self, mut rng: R, low: usize, high: usize) -> usize
    where
        R: Rng,
    {
        debug_assert!(low < high);
        low + self.0[high - low].ind_sample(&mut rng)
    }

    pub fn choose<T, R>(&self, rng: R, values: &[T]) -> T
    where
        T: Copy,
        R: Rng,
    {
        let i = self.gen(rng, 0, values.len());
        values[i]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use new_rng;

    #[test]
    fn test() {
        let mut rng = new_rng(1);
        let ranges = Ranges::new(10);
        for i in 0..10 {
            for _ in 0..100 {
                assert_eq!(i, ranges.gen(&mut rng, i, i + 1));
            }
        }
    }
}
