extern crate ea_tree_repr;
extern crate fera;
extern crate rand;

#[macro_use]
extern crate clap;

// external
use fera::fun::vec;
use fera::graph::prelude::*;
use rand::Rng;

// system
use std::collections::HashSet;
use std::rc::Rc;

// local
use ea_tree_repr::{new_rng, progress, set_default_change_any_max_tries, EulerTourTree};

pub fn main() {
    let args = args();
    set_default_change_any_max_tries(args.max_tries);
    let n = args.n;
    let min = n;
    let max = 5 * n;
    // let max = n * (n - 3) / 2;
    let step = (max - min) as f64 / (args.samples - 1) as f64;
    let mut ms = vec![];
    let mut ds = vec![];
    for i in 0..args.samples {
        ms.push(n + (i as f64 * step).round() as usize);
        ds.push(i as f64 / (args.samples - 1) as f64);
    }
    let (fails, tries) = run(&args, &ms);
    println!("d fails tries m");
    for i in 0..args.samples {
        println!("{:.3} {:.1} {} {}", ds[i], fails[i], tries[i], ms[i]);
    }
}

fn run(args: &Args, ms: &[usize]) -> (Vec<f64>, Vec<f64>) {
    let mut rng = new_rng();
    let mut fails = vec![0.0; ms.len()];
    let mut tries = vec![0.0; ms.len()];
    for _ in progress(0..args.times) {
        for (i, &m) in ms.iter().enumerate() {
            let (_, mut tree) = new(args.n, m, args.diameter, &mut rng);
            tree.avoid_bridges();
            tree.change_any(&mut rng);
            fails[i] += tree.last_change_any_failed() as usize as f64;
            tries[i] += tree.last_change_any_tries() as f64;
        }
    }
    let times = args.times as f64;
    for f in &mut fails {
        *f /= times;
        *f *= 100.0;
    }
    for t in &mut tries {
        *t /= times;
    }
    (fails, tries)
}

fn new<R: Rng>(
    n: usize,
    m: usize,
    diameter: Option<usize>,
    mut rng: R,
) -> (Rc<StaticGraph>, EulerTourTree<StaticGraph>) {
    let tree = if let Some(d) = diameter {
        StaticGraph::new_random_tree_with_diameter(n as _, d as _, &mut rng).unwrap()
    } else {
        StaticGraph::new_random_tree(n as _, &mut rng)
    };

    let mut b = StaticGraph::builder(n, m);
    let mut set = HashSet::new();
    for (u, v) in tree.edges_ends() {
        set.insert((u as usize, v as usize));
        b.add_edge(u as usize, v as usize)
    }

    while set.len() != m {
        let u = rng.gen_range(0, n);
        let v = rng.gen_range(0, n);
        if u == v || set.contains(&(u, v)) || set.contains(&(v, u)) {
            continue;
        }
        set.insert((u, v));
        b.add_edge(u, v)
    }

    let g = b.finalize();

    assert_eq!(g.num_vertices(), n);
    assert_eq!(g.num_edges(), m);

    let edges = vec(tree.edges_ends().map(|(u, v)| g.edge_by_ends(u, v)));
    let g = Rc::new(g);
    let tree = EulerTourTree::new(g.clone(), &*edges);
    (g, tree)
}

#[derive(Debug)]
struct Args {
    n: usize,
    diameter: Option<usize>,
    samples: usize,
    times: usize,
    max_tries: usize,
}

fn args() -> Args {
    let app = clap_app!(
        ("euler-tour-change-any") =>
            (version: crate_version!())
            (about: crate_description!())
            (author: crate_authors!())
            (@arg n:
                +required
                "Number of vertices")
            (@arg diameter:
                --diameter
                +takes_value
                "The tree diameter")
            (@arg max_tries:
                --max_tries
                +takes_value
                default_value("5")
                "Max number of tries in change-any before falling back to change-pred")
            (@arg samples:
                +required
                "Number of samples for the densinty")
            (@arg times:
                +required
                "Number of times to executed the experiment")
    );

    let matches = app.get_matches();
    Args {
        n: value_t_or_exit!(matches, "n", usize),
        diameter: if matches.is_present("d") {
            Some(value_t_or_exit!(matches, "diameter", usize))
        } else {
            None
        },
        max_tries: value_t_or_exit!(matches, "max_tries", usize),
        samples: value_t_or_exit!(matches, "samples", usize),
        times: value_t_or_exit!(matches, "times", usize),
    }
}
