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
use std::time::{Duration, Instant};

pub fn main() {
    setup_rayon();
    let (sizes, diameter, repr, op, times) = args();

    let time = match repr {
        Repr::EulerTour => run::<EulerTourTree<_>>(&*sizes, diameter, op, times),
        Repr::NddrAdj => run::<NddrAdjTree<_>>(&*sizes, diameter, op, times),
        Repr::NddrBalanced => run::<NddrBalancedTree<_>>(&*sizes, diameter, op, times),
        Repr::Parent => run::<ParentTree<_>>(&*sizes, diameter, op, times),
        Repr::Parent2 => run::<Parent2Tree<_>>(&*sizes, diameter, op, times),
    };

    println!("size time");
    for (s, t) in sizes.into_iter().zip(time) {
        println!("{} {:.03}", s, micro_secs(t));
    }
}

fn run<T: Tree<CompleteGraph>>(
    sizes: &[usize],
    diameter: Option<f32>,
    op: Op,
    times: usize,
) -> Vec<Duration> {
    let mut time = vec![Duration::default(); sizes.len()];
    for _ in progress(0..times) {
        time.par_iter_mut().zip(sizes).for_each(|(t, &n)| {
            let mut rng = rand::weak_rng();
            let (g, tree) = graph_tree(n, diameter, &mut rng);
            let tree = T::new(Rc::new(g), &*tree, &mut rng);
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
        })
    }
    for t in &mut time {
        *t /= 10_000 * times as u32;
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
        "parent" => Repr::Parent,
        "parent2" => Repr::Parent2,
        _ => unreachable!(),
    };
    let op = match matches.value_of("op").unwrap() {
        "change-parent" => Op::ChangeParent,
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
    Parent,
    Parent2,
}

#[derive(Copy, Clone)]
enum Op {
    ChangeParent,
    ChangeAny,
}
