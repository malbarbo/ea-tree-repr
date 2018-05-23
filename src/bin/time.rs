extern crate ea_tree_repr;
extern crate fera;
extern crate rand;
extern crate rayon;

#[macro_use]
extern crate clap;

// external
use fera::graph::prelude::*;
use rand::Rng;
use rayon::prelude::*;

// system
use std::rc::Rc;
use std::time::Duration;

// local
use ea_tree_repr::{
    micro_secs, progress, random_sp, random_sp_with_diameter, setup_rayon, time_it, EulerTourTree,
    NddrAdjTree, NddrBalancedTree, PredecessorTree, PredecessorTree2, Tree,
};

pub fn main() {
    setup_rayon();
    let (sizes, diameter, ds, op, times) = args();

    let time = match ds {
        Ds::EulerTour => run::<EulerTourTree<_>>(&*sizes, diameter, op, times),
        Ds::NddrAdj => run::<NddrAdjTree<_>>(&*sizes, diameter, op, times),
        Ds::NddrBalanced => run::<NddrBalancedTree<_>>(&*sizes, diameter, op, times),
        Ds::Predecessor => run::<PredecessorTree<_>>(&*sizes, diameter, op, times),
        Ds::Predecessor2 => run::<PredecessorTree2<_>>(&*sizes, diameter, op, times),
    };

    println!("size time clone change");
    for (s, t) in sizes.into_iter().zip(time) {
        println!(
            "{} {:.03} {:.03} {:.03}",
            s,
            micro_secs(t.0 + t.1),
            micro_secs(t.0),
            micro_secs(t.1)
        );
    }
}

fn run<T: Tree<CompleteGraph>>(
    sizes: &[usize],
    diameter: Option<f32>,
    op: Op,
    times: usize,
) -> Vec<(Duration, Duration)> {
    const TIMES: usize = 10_000;
    let mut time = vec![(Duration::default(), Duration::default()); sizes.len()];
    for _ in progress(0..times) {
        time.par_iter_mut().zip(sizes).for_each(|(t, &n)| {
            let mut rng = rand::weak_rng();
            let (g, tree) = graph_tree(n, diameter, &mut rng);
            let mut tree = T::new(Rc::new(g), &*tree, &mut rng);
            match op {
                Op::ChangePred => for _ in 0..TIMES {
                    let (t0, mut tt) = time_it(|| tree.clone());
                    let (t1, _) = time_it(|| tt.change_pred(&mut rng));
                    tree = tt;
                    t.0 += t0;
                    t.1 += t1;
                },
                Op::ChangeAny => for _ in 0..TIMES {
                    let (t0, mut tt) = time_it(|| tree.clone());
                    let (t1, _) = time_it(|| tt.change_any(&mut rng));
                    tree = tt;
                    t.0 += t0;
                    t.1 += t1;
                },
            }
        })
    }
    for t in &mut time {
        t.0 /= (TIMES * times) as u32;
        t.1 /= (TIMES * times) as u32;
    }
    time
}

fn args() -> (Vec<usize>, Option<f32>, Ds, Op, usize) {
    let app = clap_app!(
        ("time") =>
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
            (@arg diameter:
                -d
                --diameter
                +takes_value
                "Diameter of the random trees [0, 1] - (0.0 = diameter 2, 1.0 = diamenter n - 1). Default is to generate trees with random diameter.")
            (@arg times:
                +required
                "Number of times to executed the experiment")
            (@arg sizes:
                +required
                multiple(true)
                "List of the number of vertices")
    );

    let matches = app.get_matches();
    let ds = match matches.value_of("ds").unwrap() {
        "euler-tour" => Ds::EulerTour,
        "nddr-adj" => Ds::NddrAdj,
        "nddr-balanced" => Ds::NddrBalanced,
        "pred" => Ds::Predecessor,
        "pred2" => Ds::Predecessor2,
        _ => unreachable!(),
    };
    let op = match matches.value_of("op").unwrap() {
        "change-pred" => Op::ChangePred,
        "change-any" => Op::ChangeAny,
        _ => unreachable!(),
    };
    let diameter = if matches.is_present("diameter") {
        let d = value_t_or_exit!(matches, "diameter", f32);
        if d < 0.0 || d > 1.0 {
            panic!("Invalid value for diameter: {}", d)
        }
        Some(d)
    } else {
        None
    };
    let times = value_t_or_exit!(matches, "times", usize);
    let sizes = matches
        .values_of("sizes")
        .unwrap()
        .map(|x| x.parse::<usize>().unwrap())
        .collect();
    (sizes, diameter, ds, op, times)
}

fn graph_tree<R: Rng>(
    n: usize,
    diameter: Option<f32>,
    rng: R,
) -> (CompleteGraph, Vec<Edge<CompleteGraph>>) {
    let g = CompleteGraph::new(n as u32);
    let tree = if let Some(d) = diameter {
        let d = 2 + (d * (n - 3) as f32) as usize;
        random_sp_with_diameter(&g, d, rng)
    } else {
        random_sp(&g, rng)
    };
    (g, tree)
}

#[derive(Copy, Clone)]
enum Ds {
    EulerTour,
    NddrAdj,
    NddrBalanced,
    Predecessor,
    Predecessor2,
}

#[derive(Copy, Clone)]
enum Op {
    ChangePred,
    ChangeAny,
}
