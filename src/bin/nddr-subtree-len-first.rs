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

pub fn main() {
    setup_rayon();
    let (op, strategy, times, sizes) = args();
    let subtree_len = run(op, strategy, times, &sizes);
    println!("n subtree_len");
    for (n, len) in sizes.into_iter().zip(subtree_len) {
        println!("{} {}", n, len);
    }
}

fn run(op: Op, find_op: FindOpStrategy, times: usize, sizes: &[usize]) -> Vec<usize> {
    let mut len = vec![0; sizes.len()];
    for _ in progress(0..times) {
        sizes.par_iter().zip(&mut len).for_each(|(&n, ll)| {
            let mut rng = rand::weak_rng();
            let mut tree = new(n, find_op, &mut rng);
            match op {
                Op::ChangeAny => tree.op1(&mut rng),
                Op::ChangeParent => tree.op2(&mut rng),
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
                possible_values(&["adj", "adj-smaller", "balanced", "edge"])
                "strategy used to find the operands values for the operation"
            )
            (@arg op:
                +required
                possible_values(&["change-parent", "change-any"])
                "operation"
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
    let op = match matches.value_of("op").unwrap() {
        "change-parent" => Op::ChangeParent,
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
    NddrOneTreeForest::new_with_strategies(Rc::new(g), tree, find_op, FindVertexStrategy::Map, rng)
}

#[derive(Copy, Clone)]
enum Op {
    ChangeParent,
    ChangeAny,
}