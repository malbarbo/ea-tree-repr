use fera::graph::prelude::*;
use fera::graph::choose::Choose;
use rand::Rng;

use std::rc::Rc;

use tour::Tour;

// TODO: remove this wrapper
#[derive(Clone)]
pub struct EulerTourTree<G: Graph> {
    tour: Tour<G>,
}

impl<G> EulerTourTree<G>
where
    G: IncidenceGraph + Choose + WithVertexIndexProp,
{
    #[inline(never)]
    pub fn new(g: Rc<G>, edges: &[Edge<G>]) -> Self {
        Self {
            tour: Tour::new(g, edges),
        }
    }

    pub fn graph(&self) -> &Rc<G> {
        &self.tour.g
    }

    pub fn change_parent<R: Rng>(&mut self, rng: R) -> (Edge<G>, Edge<G>) {
        self.tour.change_parent(rng)
    }

    // TODO: add change_any

    #[cfg(test)]
    fn contains(&self, e: Edge<G>) -> bool {
        use fera_array::Array;
        self.tour.edges.contains(&e)
    }

    #[cfg(test)]
    fn to_vec(&self) -> Vec<Edge<G>> {
        self.tour
            .to_vec()
            .into_iter()
            .map(|e| {
                let index = e.index();
                let ed = self.tour.edges[index];
                if e.is_reverse() {
                    self.graph().reverse(ed)
                } else {
                    ed
                }
            })
            .collect()
    }

    #[cfg(test)]
    fn check(&self) {
        use fera::graph::algs::{Paths, Trees};
        let edges = || (0..self.graph().num_vertices() - 1).map(|i| self.tour.edges[i]);
        self.tour.check();
        assert!(self.graph().spanning_subgraph(edges()).is_tree());
        assert!(edges().all(|e| self.contains(e)));
        assert!(self.graph().is_walk(self.to_vec()));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand;
    use random_sp;

    #[test]
    fn new() {
        let g = Rc::new(CompleteGraph::new(5));
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

    fn graph_tree(n: u32) -> (Rc<CompleteGraph>, Vec<Edge<CompleteGraph>>) {
        let g = CompleteGraph::new(n);
        let tree = random_sp(&g, rand::XorShiftRng::new_unseeded());
        (Rc::new(g), tree)
    }

    #[test]
    fn basic() {
        let n = 30;
        for _ in 0..100 {
            let (g, tree) = graph_tree(n);
            EulerTourTree::new(g.clone(), &tree).check();
        }
    }

    #[test]
    fn change_parent() {
        for &n in &[3, 4, 5, 6, 10, 30] {
            _change_parent(n);
        }
    }

    fn _change_parent(n: u32) {
        let mut rng = rand::XorShiftRng::new_unseeded();
        for _ in 0..100 {
            let (g, tree) = graph_tree(n);
            let mut tree = EulerTourTree::new(g.clone(), &tree);
            let (ins, rem) = tree.change_parent(&mut rng);
            assert!(tree.contains(ins));
            assert!(!tree.contains(rem));
            tree.check();
        }
    }

    fn _ptree(g: &CompleteGraph, tour: Vec<Edge<CompleteGraph>>) {
        for e in tour {
            print!("{:?} ", g.end_vertices(e));
        }
        println!("");
    }
}
