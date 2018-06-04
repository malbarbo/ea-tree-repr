extern crate ea_tree_repr;
extern crate fera;
extern crate rand;
extern crate rayon;

#[macro_use]
extern crate clap;

// external
use fera::ext::VecExt;
use fera::fun::vec;
use fera::graph::algs::Kruskal;
use fera::graph::prelude::*;
use rand::{Rng, XorShiftRng};
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
    let args = args();

    let time = if args.nddr_adj_good {
        let case = case_nddr_adj_good;
        match args.ds {
            Ds::EulerTour => run::<_, EulerTourTree<_>, _>(&args, case),
            Ds::NddrAdj => run::<_, NddrAdjTree<_>, _>(&args, case),
            Ds::NddrBalanced => run::<_, NddrBalancedTree<_>, _>(&args, case),
            Ds::Predecessor => run::<_, PredecessorTree<_>, _>(&args, case),
            Ds::Predecessor2 => run::<_, PredecessorTree2<_>, _>(&args, case),
        }
    } else {
        match args.ds {
            Ds::EulerTour => run::<_, EulerTourTree<_>, _>(&args, case),
            Ds::NddrAdj => run::<_, NddrAdjTree<_>, _>(&args, case),
            Ds::NddrBalanced => run::<_, NddrBalancedTree<_>, _>(&args, case),
            Ds::Predecessor => run::<_, PredecessorTree<_>, _>(&args, case),
            Ds::Predecessor2 => run::<_, PredecessorTree2<_>, _>(&args, case),
        }
    };

    println!("size time clone change");
    for (s, t) in args.sizes.into_iter().zip(time) {
        println!(
            "{} {:.03} {:.03} {:.03}",
            s,
            micro_secs(t.0 + t.1),
            micro_secs(t.0),
            micro_secs(t.1)
        );
    }
}

fn run<G, T, F>(args: &Args, new: F) -> Vec<(Duration, Duration)>
where
    G: Graph,
    T: Tree<G>,
    F: Sync + Fn(&Args, usize, &mut XorShiftRng) -> (G, Vec<Edge<G>>),
{
    const TIMES: usize = 10_000;
    let mut time = vec![(Duration::default(), Duration::default()); args.sizes.len()];
    for _ in progress(0..args.times) {
        time.par_iter_mut().zip(&args.sizes).for_each(|(t, &n)| {
            let mut rng = rand::weak_rng();
            let (g, tree) = new(args, n, &mut rng);
            let mut tree = if args.nddr_adj_good {
                let r = g.vertices().next().unwrap();
                T::new_with_fake_root(Rc::new(g), Some(r), &*tree, &mut rng)
            } else {
                T::new(Rc::new(g), &*tree, &mut rng)
            };
            match args.op {
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
        t.0 /= (TIMES * args.times) as u32;
        t.1 /= (TIMES * args.times) as u32;
    }
    time
}

fn case(args: &Args, n: usize, rng: &mut XorShiftRng) -> (CompleteGraph, Vec<Edge<CompleteGraph>>) {
    let g = CompleteGraph::new(n as u32);
    let tree = if let Some(d) = args.diameter {
        let d = 2 + (d * (n - 3) as f32) as usize;
        random_sp_with_diameter(&g, d, rng)
    } else {
        random_sp(&g, rng)
    };
    (g, tree)
}

fn case_nddr_adj_good(
    _args: &Args,
    n: usize,
    rng: &mut XorShiftRng,
) -> (StaticGraph, Vec<Edge<StaticGraph>>) {
    let nsqrt = (n as f64).sqrt();
    let mut sub_vertices = vec![vec![]; nsqrt.ceil() as usize - 1];
    let r = 0;
    for (i, v) in vec(1..n).shuffled_with(&mut *rng).into_iter().enumerate() {
        let i = i % sub_vertices.len();
        sub_vertices[i].push(v);
    }
    rng.shuffle(&mut sub_vertices);

    let m: usize = sub_vertices
        .iter()
        .map(|sub| (sub.len() * (sub.len() - 1)) / 2)
        .sum();
    let mut b = StaticGraph::builder(n as usize, m + sub_vertices.len());

    // create each target subtree
    for sub in &sub_vertices {
        for i in 0..sub.len() {
            for j in (i + 1)..sub.len() {
                b.add_edge(sub[i] as usize, sub[j] as usize)
            }
        }
    }

    // add the root edges
    for sub in &sub_vertices {
        let v = sub[0];
        b.add_edge(r as usize, v as usize);
    }

    let g = b.finalize();
    let edges = vec(g.edges()).shuffled_with(rng);
    let tree = vec(g.kruskal().edges(g.out_edges(r).chain(edges)));

    (g, tree)
}

struct Args {
    sizes: Vec<usize>,
    diameter: Option<f32>,
    nddr_adj_good: bool,
    ds: Ds,
    op: Op,
    times: usize,
}

fn args() -> Args {
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
            (@arg nddr:
                --nddr_adj_good
                "Test graph that are good for nddr-adj")
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
    Args {
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
        nddr_adj_good: matches.is_present("nddr"),
        diameter: if matches.is_present("diameter") {
            let d = value_t_or_exit!(matches, "diameter", f32);
            if d < 0.0 || d > 1.0 {
                panic!("Invalid value for diameter: {}", d)
            }
            Some(d)
        } else {
            None
        },
        times: value_t_or_exit!(matches, "times", usize),
        sizes: matches
            .values_of("sizes")
            .unwrap()
            .map(|x| x.parse::<usize>().unwrap())
            .collect(),
    }
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
