extern crate ea_tree_repr;
extern crate fera;
extern crate rand;

#[macro_use]
extern crate clap;

// external
use fera::graph::prelude::*;
use rand::Rng;

// system
use std::rc::Rc;

// local
use ea_tree_repr::{
    progress, random_sp, setup_rayon, FindOpStrategy, FindVertexStrategy, NddrOneTreeForest,
};

pub fn main() {
    setup_rayon();
    let (n, op, strategy, calls, times) = args();
    let size = run(n, op, strategy, calls, times);
    // take 100 samples to not clutter the graph
    let space = (calls - 1) as f64 / 100.0;
    println!("call size");
    for i in 0..100 {
        let call = 1 + (f64::from(i) * space).ceil() as usize;
        println!("{} {}", call, size[call]);
    }
}

fn run(n: usize, op: Op, find_op: FindOpStrategy, calls: usize, times: usize) -> Vec<f32> {
    // TODO: make it run in parallel
    let mut rng = rand::weak_rng();
    let mut size = vec![0; calls];
    for _ in progress(0..times) {
        let mut tree = new(n, find_op, &mut rng);
        for s in &mut size {
            match op {
                Op::ChangePred => tree.op1(&mut rng),
                Op::ChangeAny => tree.op2(&mut rng),
            };
            *s += tree.last_op_size();
        }
    }
    size.into_iter()
        .map(|s| (s as f32) / (times as f32))
        .collect()
}

fn args() -> (usize, Op, FindOpStrategy, usize, usize) {
    let app = clap_app!(
        ("nddr-subtree-len") =>
            (version: crate_version!())
            (about: crate_description!())
            (author: crate_authors!())
            (@arg strategy:
                +required
                possible_values(&[
                    "adj",
                    "adj-smaller",
                    "balanced",
                    "edge",
                ])
                "Strategy used to find the operands values for the operator")
            (@arg op:
                +required
                possible_values(&[
                    "change-pred",
                    "change-any",
                ])
                "Operator")
            (@arg n:
                +required
                "Number of vertices")
            (@arg calls:
                +required
                "Number of calls to operator in one execution")
            (@arg times:
                +required
                "Number of times to executed the experiment")
    );

    let matches = app.get_matches();
    let n = value_t_or_exit!(matches, "n", usize);
    let op = match matches.value_of("op").unwrap() {
        "change-pred" => Op::ChangePred,
        "change-any" => Op::ChangeAny,
        _ => unreachable!(),
    };
    let strategy = match matches.value_of("strategy").unwrap() {
        "adj" => FindOpStrategy::Adj,
        "adj-smaller" => FindOpStrategy::AdjSmaller,
        "balanced" => FindOpStrategy::Balanced,
        "edge" => FindOpStrategy::Edge,
        _ => unreachable!(),
    };
    let calls = value_t_or_exit!(matches, "calls", usize);
    let times = value_t_or_exit!(matches, "times", usize);
    (n, op, strategy, calls, times)
}

fn new<R: Rng>(n: usize, find_op: FindOpStrategy, mut rng: R) -> NddrOneTreeForest<CompleteGraph> {
    let g = CompleteGraph::new(n as u32);
    let tree = random_sp(&g, &mut rng);
    NddrOneTreeForest::new_with_strategies(Rc::new(g), tree, find_op, FindVertexStrategy::Map, rng)
}

#[derive(Copy, Clone)]
enum Op {
    ChangePred,
    ChangeAny,
}
