extern crate ea_tree_repr;
extern crate fera;
extern crate rand;
extern crate rayon;

#[macro_use]
extern crate clap;

use ea_tree_repr::*;
use fera::graph::prelude::*;
use rand::Rng;
use rayon::prelude::*;

use std::rc::Rc;
use std::time::Duration;

pub fn main() {
    setup_rayon();
    let (sizes, diameter, repr, op, times) = args();

    let time = match repr {
        Repr::EulerTour => run::<EulerTourTree<_>>(&*sizes, diameter, op, times),
        Repr::NddrAdj => run::<NddrAdjTree<_>>(&*sizes, diameter, op, times),
        Repr::NddrBalanced => run::<NddrBalancedTree<_>>(&*sizes, diameter, op, times),
        Repr::Predecessor => run::<PredecessorTree<_>>(&*sizes, diameter, op, times),
        Repr::Predecessor2 => run::<PredecessorTree2<_>>(&*sizes, diameter, op, times),
    };

    println!("size time clone change");
    for (s, t) in sizes.into_iter().zip(time) {
        println!("{} {:.03} {:.03} {:.03}", s, micro_secs(t.0 + t.1), micro_secs(t.0), micro_secs(t.1));
    }
}

fn run<T: Tree<CompleteGraph>>(
    sizes: &[usize],
    diameter: Option<f32>,
    op: Op,
    times: usize,
) -> Vec<(Duration, Duration)> {
    const TIMES: usize = 100_000;
    let mut time = vec![(Duration::default(), Duration::default()); sizes.len()];
    for _ in progress(0..times) {
        time.par_iter_mut().zip(sizes).for_each(|(t, &n)| {
            let mut rng = rand::weak_rng();
            let (g, tree) = graph_tree(n, diameter, &mut rng);
            let tree = T::new(Rc::new(g), &*tree, &mut rng);
            match op {
                Op::ChangePred => for _ in 0..TIMES {
                    let (t0, mut tree) = time_it(|| tree.clone());
                    let (t1, _) = time_it(|| tree.change_pred(&mut rng));
                    t.0 += t0;
                    t.1 += t1;
                },
                Op::ChangeAny => for _ in 0..TIMES {
                    let (t0, mut tree) = time_it(|| tree.clone());
                    let (t1, _) = time_it(|| tree.change_any(&mut rng));
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

fn args() -> (Vec<usize>, Option<f32>, Repr, Op, usize) {
    let app = clap_app!(
        ("time") =>
            (version: crate_version!())
            (about: crate_description!())
            (author: crate_authors!())
            (@arg repr:
                +required
                possible_values(&[
                    "euler-tour",
                    "nddr-adj",
                    "nddr-balanced",
                    "pred",
                    "pred2",
                ])
                "tree representation"
            )
            (@arg op:
                +required
                possible_values(&[
                    "change-pred",
                    "change-any"
                ])
                "operation"
            )
            (@arg diameter: -d --diameter +takes_value
                "the diamenter of the random trees [0, 1] - (0.0 = diameter 2, 1.0 = diamenter n - 1). Default is to generate trees with random diameters."
            )
            (@arg times:
                +required
                "number of times to executed the experiment"
            )
            (@arg sizes:
                +required
                multiple(true)
                "list of number of vertices"
            )
    );

    let matches = app.get_matches();
    let repr = match matches.value_of("repr").unwrap() {
        "euler-tour" => Repr::EulerTour,
        "nddr-adj" => Repr::NddrAdj,
        "nddr-balanced" => Repr::NddrBalanced,
        "pred" => Repr::Predecessor,
        "pred2" => Repr::Predecessor2,
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
    (sizes, diameter, repr, op, times)
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
enum Repr {
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
