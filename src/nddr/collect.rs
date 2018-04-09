use fera::graph::algs::Degrees;
use fera::graph::choose::Choose;
use fera::graph::prelude::*;
use fera::graph::traverse::{continue_if, Control, Dfs, OnDiscoverTreeEdge, Visitor};

use rand::Rng;

use std::cmp::Ordering;
use std::collections::{BinaryHeap, VecDeque};

use Ndd;

pub fn collect_ndds<G>(g: &G, roots: &[Vertex<G>]) -> Vec<Vec<Ndd<Vertex<G>>>>
where
    G: IncidenceGraph,
{
    struct CompsVisitor<G: IncidenceGraph> {
        trees: Vec<Vec<Ndd<Vertex<G>>>>,
        depth: DefaultVertexPropMut<G, usize>,
    }

    impl<G: IncidenceGraph> Visitor<G> for CompsVisitor<G> {
        fn discover_root_vertex(&mut self, g: &G, v: Vertex<G>) -> Control {
            self.trees.push(vec![Ndd::new(v, 0, g.out_degree(v))]);
            Control::Continue
        }

        fn discover_edge(&mut self, g: &G, e: Edge<G>) -> Control {
            let (u, v) = g.ends(e);
            self.depth[v] = self.depth[u] + 1;
            self.trees
                .last_mut()
                .unwrap()
                .push(Ndd::new(v, self.depth[v], g.out_degree(v)));
            Control::Continue
        }
    }

    let mut vis = CompsVisitor {
        trees: vec![],
        depth: g.vertex_prop(0usize),
    };
    g.dfs(&mut vis).roots(roots.iter().cloned()).run();
    vis.trees
}

pub fn find_star_tree<G, R>(g: &G, n: usize, rng: R) -> Vec<Ndd<Vertex<G>>>
where
    G: IncidenceGraph + Choose,
    R: Rng,
{
    find_star_tree_balanced(g, n, rng)
}

pub fn find_star_tree_dfs<G, R>(g: &G, n: usize, mut rng: R) -> Vec<Ndd<Vertex<G>>>
where
    G: IncidenceGraph + Choose,
    R: Rng,
{
    let r = g.choose_vertex(&mut rng).unwrap();
    let mut edges = vec![];

    g.dfs(OnDiscoverTreeEdge(|e| {
        edges.push(e);
        continue_if(edges.len() + 1 < n)
    })).root(r)
        .run();

    let s = g.edge_induced_subgraph(edges);
    collect_ndds(&s, &[r]).into_iter().next().unwrap()
}

pub fn find_star_tree_balanced<G, R>(g: &G, n: usize, rng: R) -> Vec<Ndd<Vertex<G>>>
where
    G: IncidenceGraph + Choose,
    R: Rng,
{
    // used in binary heap ordered by the minimum sublen value
    #[derive(Copy, Clone, Eq, PartialEq)]
    struct Key<E: Copy + Eq + PartialEq> {
        edge: E,
        sublen: usize,
    }

    impl<E: Copy + Eq + PartialEq> Ord for Key<E> {
        fn cmp(&self, other: &Self) -> Ordering {
            // we want a min heap, so invert the comparison
            other.sublen.cmp(&self.sublen)
        }
    }

    impl<E: Copy + Eq + PartialEq> PartialOrd for Key<E> {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }

    // the vertices that will form the star tree
    let mut star = g.default_vertex_prop(true);
    let mut sublen = g.default_vertex_prop(1usize);
    let mut out: DefaultVertexPropMut<G, Vec<Edge<G>>> = g.vertex_prop(vec![]);
    for e in g.edges() {
        let (u, v) = g.ends(e);
        out[u].push(e);
        out[v].push(g.reverse(e));
    }

    // BinaryHeap does not support removing arbitrary entries, we work around this by adding
    // repeated entries with updated sublen add validating the sublen before using a poped entry
    let mut leafs: BinaryHeap<_> = g.vertices()
        .filter(|&u| out[u].len() == 1)
        .map(|u| {
            let v = g.target(out[u][0]);
            Key {
                edge: out[u][0],
                sublen: sublen[u] + sublen[v],
            }
        })
        .collect();

    for _ in 0..(g.num_vertices() - n) {
        // u is a vertex with only one edge (u, v) such that sublen[u] + sublen[v] is minimum
        let u = (0..)
            .filter_map(|_| {
                let Key { edge, sublen: len } = leafs.pop().unwrap();
                let (u, v) = g.ends(edge);
                // discard entries with incorrect sublen
                if sublen[u] + sublen[v] == len {
                    Some(u)
                } else {
                    None
                }
            })
            .next()
            .unwrap();

        star[u] = false;

        assert!(out[u].len() == 1);
        let e = out[u].pop().unwrap();
        let v = g.target(e);
        sublen[v] += sublen[u];
        out[v].remove_item(&e).unwrap();

        if out[v].len() == 1 {
            let e = out[v][0];
            let w = g.target(e);
            leafs.push(Key {
                edge: e,
                sublen: sublen[v] + sublen[w],
            })
        } else {
            // re add leafs with the corrected sublen
            for &e in &out[v] {
                let e = g.reverse(e);
                let (u, v) = g.ends(e);
                if out[u].len() == 1 {
                    leafs.push(Key {
                        edge: out[u][0],
                        sublen: sublen[u] + sublen[v],
                    })
                }
            }
        }
    }

    // we are left with g.num_vertices() - n star vertices, so we that the edges that
    // connect this vertices
    let edges: Vec<_> = g.edges()
        .filter(|e| {
            let (u, v) = g.ends(*e);
            star[u] && star[v]
        })
        .collect();

    find_star_tree_dfs(&g.edge_induced_subgraph(edges), n, rng)
}

pub fn find_star_tree_random_walk<G, R>(g: &G, n: usize, rng: R) -> Vec<Ndd<Vertex<G>>>
where
    G: IncidenceGraph + Choose,
    R: Rng,
{
    let start = tree_center(g);
    let mut edges = vec![];
    for e in g.random_walk(rng).start(start) {
        if !edges.contains(&e) {
            edges.push(e);
            if edges.len() + 1 == n {
                break;
            }
        }
    }

    let s = g.edge_induced_subgraph(edges);
    collect_ndds(&s, &[start]).into_iter().next().unwrap()
}

fn tree_center<G: AdjacencyGraph>(g: &G) -> Vertex<G> {
    let mut deg = g.degree_spanning_subgraph(g.edges());
    let mut q: VecDeque<_> = g.vertices().filter(|&v| deg[v] == 1).collect();
    for _ in 0..g.num_vertices() - 1 {
        let u = q.pop_front().unwrap();
        for v in g.out_neighbors(u) {
            if let Some(d) = deg[v].checked_sub(1) {
                deg[v] = d;
                if d == 1 {
                    q.push_back(v);
                }
            }
        }
        deg[u] = 0;
    }
    assert_eq!(1, q.len());
    q.pop_back().unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;
    use NddTree;

    #[test]
    fn test_collect_ndds() {
        //      . 0
        //    .   .  \
        //   1    2   3
        //  / \   |   |
        // 4   5  6   7
        //            .
        //            8

        let mut b = StaticGraph::builder(9, 5);
        b.add_edge(1, 4);
        b.add_edge(1, 5);
        b.add_edge(2, 6);
        b.add_edge(0, 3);
        b.add_edge(3, 7);
        let g = b.finalize();

        let trees = collect_ndds(&g, &vec![0, 1, 2, 8]);

        assert_eq!(
            NddTree::new(trees[0].clone()),
            NddTree::from_vecs(&[0, 3, 7], &[0, 1, 2], &[1, 2, 1])
        );

        assert_eq!(
            NddTree::new(trees[1].clone()),
            NddTree::from_vecs(&[1, 4, 5], &[0, 1, 1], &[2, 1, 1])
        );

        assert_eq!(
            NddTree::new(trees[2].clone()),
            NddTree::from_vecs(&[2, 6], &[0, 1], &[1, 1])
        );
    }
}
