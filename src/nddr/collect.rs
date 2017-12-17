use fera::graph::prelude::*;
use fera::graph::traverse::{continue_if, Control, Dfs, Visitor, OnDiscoverTreeEdge};

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

pub fn find_star_tree<G>(g: &G, v: Vertex<G>, n: usize) -> Vec<Ndd<Vertex<G>>>
where
    G: IncidenceGraph,
{
    let mut edges = vec![];
    // TODO: use a randwalk, so star_edges do not have a tendency
    g.dfs(&mut OnDiscoverTreeEdge(|e| {
        edges.push(e);
        continue_if(edges.len() + 1 < n)
    })).root(v)
        .run();

    let s = g.edge_induced_subgraph(edges);
    collect_ndds(&s, &[v]).into_iter().next().unwrap()
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
