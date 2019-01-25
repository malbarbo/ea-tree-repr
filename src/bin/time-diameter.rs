extern crate ea_tree_repr;
extern crate fera;
extern crate rand;

#[macro_use]
extern crate clap;

// external
use fera::fun::vec;
use fera::graph::prelude::*;

// system
use std::rc::Rc;
use std::time::{Duration, Instant};

// local
use ea_tree_repr::{
    micro_secs, new_rng, progress, random_sp_with_diameter, EulerTourTree, NddrAdjTree,
    NddrFreeTree, PredecessorTree, PredecessorTree2, Tree,
};

fn main() {
    let args = args();

    if args.ds != Ds::NddrFree {
        eprintln!("Ignoring k = {} parameter", args.k);
    }
    ea_tree_repr::set_default_k(args.k);

    let (d, time) = match args.ds {
        Ds::EulerTour => run::<EulerTourTree<_>>(&args),
        Ds::NddrAdj => run::<NddrAdjTree<_>>(&args),
        Ds::NddrFree => run::<NddrFreeTree<_>>(&args),
        Ds::Predecessor => run::<PredecessorTree<_>>(&args),
        Ds::Predecessor2 => run::<PredecessorTree2<_>>(&args),
    };

    println!("diameter time");
    for (d, t) in d.into_iter().zip(time) {
        println!("{} {:.03}", d, micro_secs(t));
    }
}

fn run<T: Tree<CompleteGraph>>(args: &Args) -> (Vec<usize>, Vec<Duration>) {
    let space = (args.n - 3) as f64 / (args.samples - 1) as f64;
    let ds = vec((0..args.samples).map(|i| 2 + (i as f64 * space).round() as usize));
    let mut time = vec![Duration::default(); args.samples];
    for _ in progress(0..args.times) {
        ds.iter().zip(&mut time).for_each(|(&d, t)| {
            let mut rng = new_rng();
            let (g, tree) = graph_tree(args.n, d);
            let tree = T::new(Rc::new(g), &*tree, &mut rng);
            let start = Instant::now();
            match args.op {
                Op::ChangePred => {
                    for _ in 0..args.iters {
                        tree.clone().change_pred(&mut rng);
                    }
                }
                Op::ChangeAny => {
                    for _ in 0..args.iters {
                        tree.clone().change_any(&mut rng);
                    }
                }
            }
            *t += start.elapsed();
        })
    }
    for t in &mut time {
        *t /= (args.iters * args.times) as u32;
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
                    "nddr-free",
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
            (@arg iters:
                --iters
                +takes_value
                default_value("10000")
                "The number times the mutation operation is applied in a tree")
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
            "nddr-free" => Ds::NddrFree,
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
        iters: value_t_or_exit!(matches, "iters", usize),
        samples: value_t_or_exit!(matches, "samples", usize),
        times: value_t_or_exit!(matches, "times", usize),
    }
}

fn graph_tree(n: usize, d: usize) -> (CompleteGraph, Vec<Edge<CompleteGraph>>) {
    let g = CompleteGraph::new(n as u32);
    let tree = random_sp_with_diameter(&g, d, new_rng());
    (g, tree)
}

#[derive(Debug)]
struct Args {
    n: usize,
    ds: Ds,
    op: Op,
    k: usize,
    iters: usize,
    samples: usize,
    times: usize,
}

#[derive(Copy, Clone, Debug, PartialEq)]
enum Ds {
    EulerTour,
    NddrAdj,
    NddrFree,
    Predecessor,
    Predecessor2,
}

#[derive(Copy, Clone, Debug)]
enum Op {
    ChangePred,
    ChangeAny,
}
