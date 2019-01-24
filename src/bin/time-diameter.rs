extern crate ea_tree_repr;
extern crate fera;
extern crate rand;
extern crate rayon;

#[macro_use]
extern crate clap;

// external
use fera::fun::vec;
use fera::graph::prelude::*;
use rayon::prelude::*;

// system
use std::rc::Rc;
use std::time::{Duration, Instant};

// local
use ea_tree_repr::{
    micro_secs, new_rng, progress, random_sp_with_diameter, setup_rayon, EulerTourTree,
    NddrAdjTree, NddrBalancedTree, PredecessorTree, PredecessorTree2, Tree,
};

fn main() {
    setup_rayon();
    let args = args();

    if args.ds != Ds::NddrBalanced {
        eprintln!("Ignoring k = {} parameter", args.k);
    }
    ea_tree_repr::set_default_k(args.k);

    let (d, time) = match args.ds {
        Ds::EulerTour => run::<EulerTourTree<_>>(args.n, args.op, args.samples, args.times),
        Ds::NddrAdj => run::<NddrAdjTree<_>>(args.n, args.op, args.samples, args.times),
        Ds::NddrBalanced => run::<NddrBalancedTree<_>>(args.n, args.op, args.samples, args.times),
        Ds::Predecessor => run::<PredecessorTree<_>>(args.n, args.op, args.samples, args.times),
        Ds::Predecessor2 => run::<PredecessorTree2<_>>(args.n, args.op, args.samples, args.times),
    };

    println!("diameter time");
    for (d, t) in d.into_iter().zip(time) {
        println!("{} {:.03}", d, micro_secs(t));
    }
}

fn run<T: Tree<CompleteGraph>>(
    n: usize,
    op: Op,
    samples: usize,
    times: usize,
) -> (Vec<usize>, Vec<Duration>) {
    let space = (n - 3) as f64 / (samples - 1) as f64;
    let ds = vec((0..samples).map(|i| 2 + (i as f64 * space).round() as usize));
    let mut time = vec![Duration::default(); samples];
    for _ in progress(0..times) {
        ds.par_iter().zip(&mut time).for_each(|(&d, t)| {
            let mut rng = new_rng();
            let (g, tree) = graph_tree(n, d);
            let tree = T::new(Rc::new(g), &*tree, &mut rng);
            let start = Instant::now();
            match op {
                Op::ChangePred => {
                    for _ in 0..10_000 {
                        tree.clone().change_pred(&mut rng);
                    }
                }
                Op::ChangeAny => {
                    for _ in 0..10_000 {
                        tree.clone().change_any(&mut rng);
                    }
                }
            }
            *t += start.elapsed();
        })
    }
    for t in &mut time {
        *t /= 10_000 * times as u32;
    }
    (ds, time)
}

fn args() -> Args {
    let app = clap_app!(
        ("time-diameter") =>
            (version: crate_version!())
            (about: crate_description!())
            (author: crate_authors!())
            (@arg ds:
                +required
                possible_values(&[
                    "euler-tour",
                    "nddr-adj",
                    "nddr-balanced",
                    "pred",
                    "pred2",
                ])
                "Data structure")
            (@arg op:
                +required
                possible_values(&[
                    "change-pred",
                    "change-any"
                ])
                "Operator")
            (@arg n:
                +required
                "Number of vertices")
            (@arg k:
                -k
                +takes_value
                default_value("1")
                "The k parameter for NDDR")
            (@arg samples:
                +required
                "Number of tree diameter samples")
            (@arg times:
                +required
                "Number of times to executed the experiment")
    );

    let matches = app.get_matches();
    Args {
        n: value_t_or_exit!(matches, "n", usize),
        ds: match matches.value_of("ds").unwrap() {
            "euler-tour" => Ds::EulerTour,
            "nddr-adj" => Ds::NddrAdj,
            "nddr-balanced" => Ds::NddrBalanced,
            "pred" => Ds::Predecessor,
            "pred2" => Ds::Predecessor2,
            _ => unreachable!(),
        },
        op: match matches.value_of("op").unwrap() {
            "change-pred" => Op::ChangePred,
            "change-any" => Op::ChangeAny,
            _ => unreachable!(),
        },
        k: value_t_or_exit!(matches, "k", usize),
        samples: value_t_or_exit!(matches, "samples", usize),
        times: value_t_or_exit!(matches, "times", usize),
    }
}

#[derive(Debug)]
struct Args {
    n: usize,
    ds: Ds,
    op: Op,
    k: usize,
    samples: usize,
    times: usize,
}

fn graph_tree(n: usize, d: usize) -> (CompleteGraph, Vec<Edge<CompleteGraph>>) {
    let g = CompleteGraph::new(n as u32);
    let tree = random_sp_with_diameter(&g, d, new_rng());
    (g, tree)
}

#[derive(Copy, Clone, Debug, PartialEq)]
enum Ds {
    EulerTour,
    NddrAdj,
    NddrBalanced,
    Predecessor,
    Predecessor2,
}

#[derive(Copy, Clone, Debug)]
enum Op {
    ChangePred,
    ChangeAny,
}
