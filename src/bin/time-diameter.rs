extern crate ea_tree_repr;
extern crate fera;
extern crate rand;

#[macro_use]
extern crate clap;

use ea_tree_repr::*;
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
        println!("{} {:.03}", d, micro_secs(t));
    }
}

fn run<T: Tree>(
    n: usize,
    op: Op,
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
            match op {
                Op::ChangeParent => {
                    for _ in 0..10_000 {
                        tree.clone().change_parent(&mut rng);
                    }
                },
                Op::ChangeAny => {
                    for _ in 0..10_000 {
                        tree.clone().change_any(&mut rng);
                    }
                }
            }
            *t += start.elapsed();
        }
    }
    for t in &mut time {
        *t /= 10_000 * times as u32;
    }
    (ds, time)
}

fn args() -> (usize, Repr, Op, usize, usize) {
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
                possible_values(&[
                    "change-parent",
                    "change-any"
                ])
                "operation"
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
    let op = match matches.value_of("op").unwrap() {
        "change-parent" => Op::ChangeParent,
        "change-any" => Op::ChangeAny,
        _ => unreachable!(),
    };
    let samples = value_t_or_exit!(matches, "samples", usize);
    let times = value_t_or_exit!(matches, "times", usize);
    (n, repr, op, samples, times)
}

fn graph_tree(n: usize, d: usize) -> (CompleteGraph, Vec<Edge<CompleteGraph>>) {
    let g = CompleteGraph::new(n as u32);
    let tree = random_sp_with_diameter(&g, d, rand::weak_rng());
    (g, tree)
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
