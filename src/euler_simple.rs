use rand::Rng;

use std::fmt::Debug;

// An simple implementation of EulerTour to help understand the change operations
pub struct EulerTourSimple<T: PartialEq> {
    values: Vec<T>,
}

impl<T: PartialEq + Debug> EulerTourSimple<T> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use rand;

    #[test]
    fn check_simple() {
        let mut rng = rand::weak_rng();
        for n in 5..30 {
            let mut tree = EulerTourSimple::new((0..n).chain((0..n).rev()).collect());
            tree.check();
            for _ in 0..100 {
                tree.change_parent(&mut rng);
                tree.check();
            }
        }
    }
}
