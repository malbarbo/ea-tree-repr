use fera::graph::choose::Choose;
use fera::graph::prelude::*;
use rand::Rng;

use std::rc::Rc;

use {collect_ndds, one_tree_op1, one_tree_op2, NddTree};

pub struct NddrOneTree<G: WithVertex>
where
    G: WithVertex,
{
    g: Rc<G>,
    tree: NddTree<Vertex<G>>,
}

impl<G: WithVertex> Clone for NddrOneTree<G> {
    fn clone(&self) -> Self {
        Self {
            g: Rc::clone(&self.g),
            tree: self.tree.clone(),
        }
    }

    fn clone_from(&mut self, other: &Self) {
        self.g.clone_from(&other.g);
        self.tree.clone_from(&other.tree);
    }
}

impl<G: AdjacencyGraph + Choose> NddrOneTree<G> {
    pub fn new(g: Rc<G>, edges: &[Edge<G>]) -> Self {
        let tree = {
            let sub = g.spanning_subgraph(edges);
            let r = g.source(edges[0]);
            let mut ndds = collect_ndds(&sub, &[r]);
            assert_eq!(1, ndds.len());
            let mut t = NddTree::new(ndds.pop().unwrap());
            t.comp_degs(&*g);
            t
        };
        Self { g, tree }
    }

    pub fn graph(&self) -> &Rc<G> {
        &self.g
    }

    pub fn contains(&self, e: Edge<G>) -> bool {
        let (u, v) = self.g.ends(e);
        self.tree.contains_edge_by_vertex(u, v)
    }

    pub fn edges(&self) -> Vec<Edge<G>> {
        (1..self.tree.len())
            .map(|i| {
                let v = self.tree[i].vertex();
                self.g.edge_by_ends(v, self.tree.parent_vertex(i).unwrap())
            })
            .collect()
    }

    pub fn op1<R: Rng>(&mut self, rng: R) -> (Edge<G>, Edge<G>) {
        let (ins, p, a) = self.find_op1(rng);
        let rem = self
            .g
            .edge_by_ends(self.tree[p].vertex(), self.tree.parent_vertex(p).unwrap());
        self.tree = one_tree_op1(&self.tree, p, a);
        (ins, rem)
    }

    pub fn op2<R: Rng>(&mut self, rng: R) -> (Edge<G>, Edge<G>) {
        let (ins, p, r, a) = self.find_op2(rng);
        let rem = self
            .g
            .edge_by_ends(self.tree[p].vertex(), self.tree.parent_vertex(p).unwrap());
        self.tree = one_tree_op2(&self.tree, p, r, a);
        (ins, rem)
    }

    fn find_op1<R: Rng>(&self, mut rng: R) -> (Edge<G>, usize, usize) {
        for ins in self.g.choose_edge_iter(&mut rng) {
            if self.contains(ins) {
                continue;
            }
            let (u, v) = self.g.ends(ins);
            let u = self.tree.find_vertex(u).unwrap();
            let v = self.tree.find_vertex(v).unwrap();

            if u != 0 && !self.tree.is_ancestor(u, v) {
                return (ins, u, v);
            }

            if v != 0 && !self.tree.is_ancestor(v, u) {
                return (ins, v, u);
            }
        }
        unreachable!()
    }

    fn find_op2<R: Rng>(&self, mut rng: R) -> (Edge<G>, usize, usize, usize) {
        for _ in 0..1_000_000 {
            let (ins, r, a) = self.find_op1(&mut rng);
            let mut count = 0;
            let mut p = r;
            while let Some(pp) = self.tree.parent(p) {
                count += 1;
                p = pp;
            }

            p = r;
            for _ in 0..rng.gen_range(0, count) {
                p = self.tree.parent(p).unwrap()
            }
            // See NddrOneTreeForest::find_vertices_op2
            if !self.tree.is_ancestor(p, a) {
                return (ins, p, r, a);
            }
        }
        unreachable!("find_op2: could not find operands!")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fera::graph::algs::Trees;
    use new_rng;
    use random_sp;

    fn new(n: u32) -> NddrOneTree<CompleteGraph> {
        let mut rng = new_rng();
        let g = Rc::new(CompleteGraph::new(n));
        let edges = random_sp(&*g, &mut rng);
        NddrOneTree::new(g, &*edges)
    }

    #[test]
    fn op1() {
        let mut rng = new_rng();
        let mut tree = new(100);
        for _ in 0..100 {
            let (ins, rem) = tree.op1(&mut rng);
            assert!(tree.contains(ins));
            assert!(!tree.contains(rem));
            assert!(tree.g.spanning_subgraph(tree.edges()).is_tree());
        }
    }

    #[test]
    fn op2() {
        let mut rng = new_rng();
        let mut tree = new(100);
        for _ in 0..100 {
            let (ins, rem) = tree.op2(&mut rng);
            assert!(tree.contains(ins));
            assert!(!tree.contains(rem));
            assert!(tree.g.spanning_subgraph(tree.edges()).is_tree());
        }
    }
}
