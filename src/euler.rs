use fera_array::{CowNestedArray, DynamicArray};
use fera::graph::prelude::*;
use fera::graph::traverse::{Dfs, OnTraverseEvent, TraverseEvent};
use fera::graph::choose::Choose;
use rand::Rng;
use rpds::HashTrieMap;

use std::rc::Rc;

use tour::{Tour, TourEdge};

#[derive(Clone)]
pub struct EulerTourTree<G: Graph> {
    g: Rc<G>,
    tour: Tour<G>,
    // Map from TourEdge.index() to Edge<G>
    edges: CowNestedArray<Edge<G>>,
    // Map from Edge<G> to TourEdge
    tour_edges: HashTrieMap<Edge<G>, TourEdge>,
}

impl<G> EulerTourTree<G>
    where G: IncidenceGraph + Choose + WithVertexIndexProp
{
    #[inline(never)]
    pub fn new(g: Rc<G>, edges: &[Edge<G>]) -> Self {
        // TODO: avoid using this intermediary vector
        let mut tour = Vec::with_capacity(2 * (g.num_vertices() - 1));
        let mut eds = CowNestedArray::with_capacity(g.num_vertices() - 1);
        let mut stack = vec![];
        let mut tour_edges = HashTrieMap::new();
        let mut id = 0;
        g.spanning_subgraph(edges)
            .dfs(OnTraverseEvent(|e| match e {
                TraverseEvent::DiscoverEdge(e) => {
                    eds.push(e);
                    let te = TourEdge::new(id);
                    tour_edges = tour_edges.insert(e, te);
                    tour.push(te);
                    stack.push(id);
                    id += 1;
                }
                TraverseEvent::FinishEdge(e) => {
                    let id = stack.pop().unwrap();
                    assert_eq!(eds[id], e);
                    tour.push(TourEdge::new_reversed(id));
                }
                _ => (),
            }))
            .run();
        Self {
            g: g.clone(),
            tour: panic!(), //Tour::new(g.clone(), &tour, eds.clone(), tour_edges.clone()),
            edges: eds,
            tour_edges: tour_edges,
        }
    }

    pub fn graph(&self) -> &Rc<G> {
        &self.g
    }

    pub fn change_parent<R: Rng>(&mut self, rng: R) -> (Edge<G>, Edge<G>) {
        let rem = self.tour.change_parent(rng);
        let (start, end) = self.tour.range(rem).unwrap();
        let p = if start != (0, 0) { start } else { end };
        let parent = self.get_target(self.tour.get_prev(p));
        let subroot = self.get_source(self.tour.get_next(p));
        let ins = self.g.edge_by_ends(parent, subroot);
        let rem = self.replace_edge(rem, ins);
        // FIXME: make sure that ins != re
        (ins, rem)
    }

    // TODO: add change_any

    #[inline(never)]
    fn replace_edge(&mut self, rem: TourEdge, new: Edge<G>) -> Edge<G> {
        let old = self.edges[rem.index()];
        if rem.is_reverse() {
            self.edges[rem.index()] = self.g.reverse(new);
        } else {
            self.edges[rem.index()] = new;
        }
        old
    }

    fn get_source(&self, e: TourEdge) -> Vertex<G> {
        let i = e.index();
        if e.is_reverse() {
            self.g.target(self.edges[i])
        } else {
            self.g.source(self.edges[i])
        }
    }

    fn get_target(&self, e: TourEdge) -> Vertex<G> {
        let i = e.index();
        if e.is_reverse() {
            self.g.source(self.edges[i])
        } else {
            self.g.target(self.edges[i])
        }
    }

    /*
    #[cfg(test)]
    fn contains(&self, e: Edge<G>) -> bool {
        use fera_array::Array;
        self.edges.contains(&e)
    }

    #[cfg(test)]
    fn to_vec(&self) -> Vec<Edge<G>> {
        self.tour
            .to_vec()
            .into_iter()
            .map(|e| {
                if e.is_reverse() {
                    self.g.reverse(self.edges[e.index()])
                } else {
                    self.edges[e.index()]
                }
            })
            .collect()
    }

    #[cfg(test)]
    fn check(&self) {
        use fera::graph::algs::{Paths, Trees};
        let edges = || (0..self.g.num_vertices() - 1).map(|i| self.edges[i]);
        self.tour.check();
        assert!(self.g.spanning_subgraph(edges()).is_tree());
        assert!(edges().all(|e| self.contains(e)));
        assert!(self.g.is_walk(self.to_vec()));
    }
    */
}

/* #[cfg(test)]
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
            // FIXME: make sure that ins != rem
            if ins != rem {
                assert!(tree.contains(ins));
                assert!(!tree.contains(rem));
            }
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
*/
