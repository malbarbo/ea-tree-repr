// TODO: remove
#![allow(dead_code)]

use fera::graph::prelude::*;
use fera::graph::traverse::{Dfs, OnTraverseEvent, TraverseEvent};
use rand::Rng;

use std::rc::Rc;

use tour::{Tour, TourEdge};

#[derive(Clone)]
pub struct EulerTourTree<G: Graph>
{
    g: Rc<G>,
    tour: Tour,
    edges: Vec<Edge<G>>,
}

impl<G: Graph> EulerTourTree<G>
{
    #[inline(never)]
    pub fn new(g: G, edges: &[Edge<G>]) -> Self {
        // TODO: avoid using this intermediary vector
        let mut tour = Vec::with_capacity(2 * (g.num_vertices() - 1));
        let mut eds = Vec::with_capacity(g.num_vertices() - 1);
        let mut stack = vec![];
        let mut id = 0;
        g.spanning_subgraph(edges)
            .dfs(OnTraverseEvent(|e| match e {
                TraverseEvent::DiscoverEdge(e) => {
                    eds.push(e);
                    tour.push(TourEdge::new(id));
                    stack.push(id);
                    id += 1;
                },
                TraverseEvent::FinishEdge(e) => {
                    let id = stack.pop().unwrap();
                    assert_eq!(eds[id], e);
                    tour.push(TourEdge::new_reversed(id));
                }
                _ => (),
            }))
            .run();
        Self {
            g: Rc::new(g),
            tour: Tour::new(&tour),
            edges: eds,
        }
    }

    #[inline(never)]
    pub fn change_parent<R: Rng>(&mut self, rng: R) -> (Edge<G>, Edge<G>) {
        let rem = self.tour.change_parent(rng);
        let (start, end) = self.tour.range(rem).unwrap();
        let p = if start != (0, 0) {
            start
        } else {
            end
        };
        // create a function
        let parent = {
            let e = self.tour.get_prev(p);
            let i = e.index();
            if e.is_reverse() {
                self.g.source(self.edges[i])
            } else {
                self.g.target(self.edges[i])
            }
        };
        let subroot = {
            let e = self.tour.get_next(p);
            let i = e.index();
            if e.is_reverse() {
                self.g.target(self.edges[i])
            } else {
                self.g.source(self.edges[i])
            }
        };
        let ins = self.g.edge_by_ends(parent, subroot);
        let re = self.edges[rem.index()];
        if rem.is_reverse() {
            self.edges[rem.index()] = self.g.reverse(ins);
        } else {
            self.edges[rem.index()] = ins;
        }
        // FIXME: make sure that ins != re
        (ins, re)
    }

    fn contains(&self, e: Edge<G>) -> bool {
        self.edges.contains(&e)
    }

    fn to_vec(&self) -> Vec<Edge<G>> {
        self.tour.to_vec().into_iter().map(|e| self.edges[e.index()]).collect()
    }

    fn check(&self) {
        use fera::graph::algs::Trees;
        assert!(self.g.spanning_subgraph(&self.edges).is_tree());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand;
    use random_sp;

    #[test]
    fn new() {
        let g = CompleteGraph::new(5);
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

    fn graph_tree(n: u32) -> (CompleteGraph, Vec<Edge<CompleteGraph>>) {
        let g = CompleteGraph::new(n);
        let tree = random_sp(&g, rand::XorShiftRng::new_unseeded());
        (g, tree)
    }

    #[test]
    fn basic() {
        let n = 30;
        for _ in 0..100 {
            let (g, tree) = graph_tree(n);
            EulerTourTree::new(g.clone(), &tree).check();
        }
    }

    fn ptree(g: &CompleteGraph, tour: Vec<Edge<CompleteGraph>>) {
        for e in tour {
            print!("{:?} ", g.end_vertices(e));
        }
        println!("");
    }

    fn _change_parent(n: u32) {
        let mut rng = rand::XorShiftRng::new_unseeded();
        for _ in 0..100 {
            let (g, tree) = graph_tree(n);
            let mut tree = EulerTourTree::new(g.clone(), &tree);
            let (ins, rem) = tree.change_parent(&mut rng);
            // FIXME: make sure that ins != rem
            if ins != rem {
                assert!(tree.contains(ins));
                assert!(!tree.contains(rem));
            }
            tree.check();
        }
    }

    #[test]
    fn change_parent() {
        for &n in &[3, 4, 5, 6, 10, 30] {
            _change_parent(n);
        }
    }
}
