extern crate ec_tree_repr;
extern crate fera;
extern crate rand;

#[macro_use]
extern crate clap;

use ec_tree_repr::*;
use fera::fun::vec;
use fera::graph::prelude::*;

use std::time::{Duration, Instant};

fn main() {
    let (n, repr, op, samples, times) = args();

    let (d, time) = match repr {
        Repr::EulerTour => run::<EulerTourTree<_>>(n, op, samples, times),
        Repr::NddrAdj => run::<NddrAdjTree>(n, op, samples, times),
        Repr::NddrBalanced => run::<NddrBalancedTree>(n, op, samples, times),
        Repr::Parent => run::<ParentTree<_>>(n, op, samples, times),
        Repr::Parent2 => run::<Parent2Tree<_>>(n, op, samples, times),
    };

    println!("diameter time");
    for (d, t) in d.into_iter().zip(time) {
        println!("{} {}", d, micro_secs(t));
    }
}

pub fn run<T: Tree>(
    n: usize,
    op: usize,
    samples: usize,
    times: usize,
) -> (Vec<usize>, Vec<Duration>) {
    let space = (n - 3) as f64 / (samples - 1) as f64;
    let ds = vec((0..samples).map(|i| 2 + (i as f64 * space).round() as usize));
    let mut time = vec![Duration::default(); samples];
    let mut rng = rand::weak_rng();
    for _ in progress(0..times) {
        for (&d, t) in ds.iter().zip(&mut time) {
            let (g, tree) = graph_tree(n, d);
            let tree = T::new(g, &*tree, &mut rng);
            let start = Instant::now();
            for _ in 0..10_000 {
                let mut changed = tree.clone();
                if op == 1 {
                    changed.op1(&mut rng);
                } else {
                    changed.op2(&mut rng);
                }
            }
            *t += start.elapsed() / 10_000;
        }
    }
    for t in &mut time {
        *t /= times as u32;
    }
    (ds, time)
}

fn args() -> (usize, Repr, usize, usize, usize) {
    let app = clap_app!(
        ("time-diameter") =>
            (version: "0.1")
            (author: "Marco A L Barbosa <https://github.com/malbarbo/ec-tree-repr>")
            (@arg n: +required "number of vertices")
            (@arg repr:
                +required
                possible_values(&[
                    "euler-tour",
                    "nddr-adj",
                    "nddr-balanced",
                    "parent",
                    "parent2",
                    "parent3"
                ])
                "tree representation"
            )
            (@arg op:
                +required
                possible_values(&["1", "2"])
                "operation number"
            )
            (@arg samples:
                +required
                "number of tree diameter samples"
            )
            (@arg times:
                +required
                "number of times to executed the experiment"
            )
    );

    let matches = app.get_matches();
    let n = value_t_or_exit!(matches, "n", usize);
    let repr = match matches.value_of("repr").unwrap() {
        "euler-tour" => Repr::EulerTour,
        "nddr-adj" => Repr::NddrAdj,
        "nddr-balanced" => Repr::NddrBalanced,
        "parent" => Repr::Parent,
        "parent2" => Repr::Parent2,
        _ => unreachable!(),
    };
    let op = value_t_or_exit!(matches, "op", usize);
    let samples = value_t_or_exit!(matches, "samples", usize);
    let times = value_t_or_exit!(matches, "times", usize);
    (n, repr, op, samples, times)
}

fn graph_tree(n: usize, d: usize) -> (CompleteGraph, Vec<Edge<CompleteGraph>>) {
    let g = CompleteGraph::new(n as u32);
    let tree = random_sp_with_diameter(&g, d, rand::weak_rng());
    (g, tree)
}

enum Repr {
    EulerTour,
    NddrAdj,
    NddrBalanced,
    Parent,
    Parent2,
}
