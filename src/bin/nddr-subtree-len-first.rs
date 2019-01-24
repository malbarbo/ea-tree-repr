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

// local
use ea_tree_repr::{
    new_rng, progress, random_sp, setup_rayon, FindOpStrategy, FindVertexStrategy,
    NddrOneTreeForest,
};

pub fn main() {
    setup_rayon();
    let (op, strategy, times, sizes) = args();
    let subtree_len = run(op, strategy, times, &sizes);
    println!("n subtree_len expected");
    for (n, len) in sizes.into_iter().zip(subtree_len) {
        let sqrt_n = (n as f64).sqrt();
        println!("{} {} {}", n, len, 2.0 * sqrt_n);
    }
}

fn run(op: Op, find_op: FindOpStrategy, times: usize, sizes: &[usize]) -> Vec<usize> {
    let mut len = vec![0; sizes.len()];
    for _ in progress(0..times) {
        sizes.par_iter().zip(&mut len).for_each(|(&n, ll)| {
            let mut rng = new_rng();
            let mut tree = new(n, find_op, &mut rng);
            match op {
                Op::ChangePred => tree.op1(&mut rng),
                Op::ChangeAny => tree.op2(&mut rng),
            };
            *ll += tree.last_op_size();
        })
    }
    for ll in &mut len {
        *ll /= times;
    }
    len
}

fn args() -> (Op, FindOpStrategy, usize, Vec<usize>) {
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
                    "free",
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
            (@arg times:
                +required
                "Number of times to executed the experiment")
            (@arg sizes:
                +required
                multiple(true)
                "Nist of the number of vertices")
    );

    let matches = app.get_matches();
    let op = match matches.value_of("op").unwrap() {
        "change-pred" => Op::ChangePred,
        "change-any" => Op::ChangeAny,
        _ => unreachable!(),
    };
    let strategy = match matches.value_of("strategy").unwrap() {
        "adj" => FindOpStrategy::Adj,
        "adj-smaller" => FindOpStrategy::AdjSmaller,
        "free" => FindOpStrategy::Free,
        "edge" => FindOpStrategy::Edge,
        _ => unreachable!(),
    };
    let times = value_t_or_exit!(matches, "times", usize);
    let sizes = matches
        .values_of("sizes")
        .unwrap()
        .map(|x| x.parse::<usize>().unwrap())
        .collect();
    (op, strategy, times, sizes)
}

fn new<R: Rng>(n: usize, find_op: FindOpStrategy, mut rng: R) -> NddrOneTreeForest<CompleteGraph> {
    let g = CompleteGraph::new(n as u32);
    let tree = random_sp(&g, &mut rng);
    NddrOneTreeForest::new_with_strategies(
        Rc::new(g),
        None,
        tree,
        find_op,
        FindVertexStrategy::Map,
        rng,
    )
}

#[derive(Copy, Clone)]
enum Op {
    ChangePred,
    ChangeAny,
}
