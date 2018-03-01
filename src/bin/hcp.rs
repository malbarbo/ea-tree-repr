extern crate ea_tree_repr;
extern crate fera;
extern crate rand;
extern crate tsplib;

#[macro_use]
extern crate clap;

use ea_tree_repr::*;

use fera::fun::{position_max_by_key, position_min_by_key, vec};
use fera::graph::algs::Kruskal;
use fera::graph::prelude::*;
use rand::Rng;

use std::rc::Rc;

pub fn main() {
    let (repr, op, input) = args();
    let instance = tsplib::read(input).expect("Fail to parse input file");

    assert_eq!(
        Some(tsplib::Type::Hcp),
        instance.type_,
        "Only HCP instances are supported"
    );

    let mut builder = StaticGraph::builder(instance.dimension, 0);

    if let Some(tsplib::EdgeData::EdgeList(edges)) = instance.edge_data {
        for (u, v) in edges {
            builder.add_edge(u - 1, v - 1);
        }
    } else {
        panic!("Only EDGE_DATA_FORMAT : EDGE_LIST is supported")
    }

    let g = Rc::new(builder.finalize());

    match repr {
        Repr::EulerTour => {
            run::<EulerTourTree<_>, CowNestedArrayVertexProp<StaticGraph, u32>>(g, op)
        }
        Repr::NddrAdj => run::<NddrAdjTree<_>, CowNestedArrayVertexProp<StaticGraph, u32>>(g, op),
        Repr::NddrBalanced => {
            run::<NddrBalancedTree<_>, CowNestedArrayVertexProp<StaticGraph, u32>>(g, op)
        }
        Repr::Parent => run::<ParentTree<_>, DefaultVertexPropMut<StaticGraph, u32>>(g, op),
        Repr::Parent2 => run::<Parent2Tree<_>, CowNestedArrayVertexProp<StaticGraph, u32>>(g, op),
    }
}

fn run<T, D>(g: Rc<StaticGraph>, op: Op)
where
    T: Tree<StaticGraph>,
    D: VertexPropMutNew<StaticGraph, u32>,
    Ind<T, D>: Clone,
{
    let mut rng = rand::weak_rng();
    let mut edges = vec(g.edges());
    let mut tree = vec![];
    let mut pop = vec![];
    for _ in 0..5 {
        rng.shuffle(&mut edges);
        tree.clear();
        tree.extend(g.kruskal().edges(&edges));
        pop.push(Ind::<T, D>::new(
            g.clone(),
            &tree,
            g.vertex_prop(0),
            &mut rng,
        ));
    }

    println!("n = {}, m = {}", g.num_vertices(), g.num_edges());

    let mut best = position_min_by_key(&pop, |ind| ind.branches).unwrap();
    println!("best = {}", pop[best].branches);
    for it in 0.. {
        let i = rng.gen_range(0, pop.len());
        let mut new = pop[i].clone();
        new.mutate(op, &mut rng);
        if new.branches <= pop[i].branches {
            if new.branches < pop[best].branches {
                best = i;
                println!("it = {}, best = {}", it, new.branches);
                if new.branches == 0 {
                    break;
                }
            }
            pop[i] = new;
        } else {
            let j = position_max_by_key(&pop, |ind| ind.branches).unwrap();
            pop[j] = new;
            best = position_min_by_key(&pop, |ind| ind.branches).unwrap();
        }
    }
}

fn args() -> (Repr, Op, String) {
    let app = clap_app!(
        ("hcp") =>
            (version: crate_version!())
            (about: crate_description!())
            (author: crate_authors!())
            (@arg repr:
                +required
                possible_values(&[
                    "euler-tour",
                    "nddr-adj",
                    "nddr-balanced",
                    "parent",
                    "parent2",
                ])
                "tree representation"
            )
            (@arg op:
                +required
                possible_values(&[
                    "change-parent",
                    "change-any"
                ])
                "operation"
            )
            (@arg input:
                +required
                "input file graph (an HCP tsplib instance)"
            )
    );

    let matches = app.get_matches();
    let repr = match matches.value_of("repr").unwrap() {
        "euler-tour" => Repr::EulerTour,
        "nddr-adj" => Repr::NddrAdj,
        "nddr-balanced" => Repr::NddrBalanced,
        "parent" => Repr::Parent,
        "parent2" => Repr::Parent2,
        _ => unreachable!(),
    };
    let op = match matches.value_of("op").unwrap() {
        "change-parent" => Op::ChangeParent,
        "change-any" => Op::ChangeAny,
        _ => unreachable!(),
    };

    (repr, op, matches.value_of("input").unwrap().to_string())
}

#[derive(Clone)]
struct Ind<T, D>
where
    T: Tree<StaticGraph>,
    D: VertexPropMut<StaticGraph, u32>,
{
    tree: T,
    degree: D,
    branches: u32,
}

impl<T, D> Ind<T, D>
where
    T: Tree<StaticGraph>,
    D: VertexPropMut<StaticGraph, u32>,
{
    fn new<R: Rng>(g: Rc<StaticGraph>, edges: &[Edge<StaticGraph>], mut degree: D, rng: R) -> Self {
        let tree = T::new(g.clone(), edges, rng);
        for &e in edges {
            let (u, v) = g.ends(e);
            degree[u] += 1;
            degree[v] += 1;
        }
        let branches = g.vertices().filter(|v| degree[*v] > 2).count() as u32;
        Ind {
            tree,
            degree,
            branches,
        }
    }

    fn mutate<R: Rng>(&mut self, op: Op, rng: R) {
        let (ins, rem) = match op {
            Op::ChangeAny => self.tree.change_any(rng),
            Op::ChangeParent => self.tree.change_parent(rng),
        };

        let (a, b) = self.tree.graph().ends(rem);
        if self.degree[a] == 3 {
            self.branches -= 1;
        }
        if self.degree[b] == 3 {
            self.branches -= 1;
        }
        self.degree[a] -= 1;
        self.degree[b] -= 1;

        let (a, b) = self.tree.graph().ends(ins);
        if self.degree[a] == 2 {
            self.branches += 1;
        }
        if self.degree[b] == 2 {
            self.branches += 1;
        }
        self.degree[a] += 1;
        self.degree[b] += 1;

        self.check();
    }

    fn check(&self) {
        debug_assert_eq!(
            self.tree
                .graph()
                .vertices()
                .filter(|v| self.degree[*v] > 2)
                .count() as u32,
            self.branches
        );
    }
}

#[derive(Copy, Clone)]
enum Repr {
    EulerTour,
    NddrAdj,
    NddrBalanced,
    Parent,
    Parent2,
}

#[derive(Copy, Clone)]
enum Op {
    ChangeParent,
    ChangeAny,
}
