extern crate ec_tree_repr;
extern crate fera;
extern crate rand;

#[macro_use]
extern crate clap;

use ec_tree_repr::*;
use fera::graph::prelude::*;
use rand::Rng;

pub fn main() {
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

fn args() -> (usize, usize, FindOpStrategy, usize, usize) {
    let app = clap_app!(
        ("nddr-subtree-len") =>
            (version: "0.1")
            (author: "Marco A L Barbosa <https://github.com/malbarbo/ec-tree-repr>")
            (@arg n:
                +required
                "number of vertices"
            )
            (@arg op:
                +required
                possible_values(&["1", "2"])
                "operation number"
            )
            (@arg strategy:
                +required
                possible_values(&["adj", "adj-smaller", "balanced", "edge"])
                "strategy used to find the operands values for op"
            )
            (@arg calls:
                +required
                "number of calls to op in one execution"
            )
            (@arg times:
                +required
                "number of times to executed the experiment"
            )
    );

    let matches = app.get_matches();
    let n = value_t_or_exit!(matches, "n", usize);
    let op = value_t_or_exit!(matches, "op", usize);
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

fn run(n: usize, op: usize, find_op: FindOpStrategy, calls: usize, times: usize) -> Vec<f32> {
    let mut rng = rand::weak_rng();
    let mut size = vec![0; calls];
    for _ in progress(0..times) {
        let mut tree = new(n, find_op, &mut rng);
        for s in &mut size {
            if op == 1 {
                tree.op1(&mut rng);
            } else {
                tree.op2(&mut rng);
            }
            *s += tree.last_op_size();
        }
    }
    size.into_iter()
        .map(|s| (s as f32) / (times as f32))
        .collect()
}

fn new<R: Rng>(n: usize, find_op: FindOpStrategy, mut rng: R) -> NddrOneTreeForest<CompleteGraph> {
    let g = CompleteGraph::new(n as u32);
    let tree = random_sp(&g, &mut rng);
    NddrOneTreeForest::new_with_strategies(g, tree, find_op, FindVertexStrategy::Map, rng)
}
