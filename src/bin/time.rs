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
    micro_secs, new_rng, progress, random_sp, random_sp_with_diameter, setup_rayon, time_it,
    EulerTourTree, NddrAdjTree, NddrBalancedTree, NddrEdgeTree, PredecessorTree, PredecessorTree2,
    Tree,
};

pub fn main() {
    setup_rayon();
    let args = args();

    if args.ds != Ds::NddrBalanced {
        eprintln!("Ignoring k = {} parameter", args.k);
    }
    ea_tree_repr::set_default_k(args.k);

    let time = if args.forest || args.regular.is_some() {
        if args.diameter.is_some() {
            eprintln!("The diameter can only be specified for complete graphs");
            ::std::process::exit(1);
        }
        let case = if args.forest {
            case_forest
        } else {
            case_regular
        };
        match args.ds {
            Ds::EulerTour => run::<_, EulerTourTree<_>, _>(&args, case),
            Ds::NddrAdj => run::<_, NddrAdjTree<_>, _>(&args, case),
            Ds::NddrBalanced => {
                eprintln!("nddr-balanced cannot be used in non complete graphs");
                ::std::process::exit(1);
            }
            Ds::NddrEdge => run::<_, NddrEdgeTree<_>, _>(&args, case),
            Ds::Predecessor => run::<_, PredecessorTree<_>, _>(&args, case),
            Ds::Predecessor2 => run::<_, PredecessorTree2<_>, _>(&args, case),
        }
    } else {
        let case = case_complete;
        match args.ds {
            Ds::EulerTour => run::<_, EulerTourTree<_>, _>(&args, case),
            Ds::NddrAdj => run::<_, NddrAdjTree<_>, _>(&args, case),
            Ds::NddrBalanced => run::<_, NddrBalancedTree<_>, _>(&args, case),
            Ds::NddrEdge => run::<_, NddrEdgeTree<_>, _>(&args, case),
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
    F: Sync + Fn(&Args, u32, &mut XorShiftRng) -> (G, Vec<Edge<G>>),
{
    let mut time = vec![(Duration::default(), Duration::default()); args.sizes.len()];
    for _ in progress(0..args.times) {
        time.par_iter_mut().zip(&args.sizes).for_each(|(t, &n)| {
            let mut rng = new_rng();
            let (g, tree) = new(args, n, &mut rng);
            let mut tree = if args.forest {
                let r = g.vertices().next().unwrap();
                T::new_with_fake_root(Rc::new(g), Some(r), &*tree, &mut rng)
            } else {
                T::new(Rc::new(g), &*tree, &mut rng)
            };
            match args.op {
                Op::ChangePred => {
                    for _ in 0..args.iters {
                        let (t0, mut tt) = time_it(|| tree.clone());
                        let (t1, _) = time_it(|| tt.change_pred(&mut rng));
                        tree = tt;
                        t.0 += t0;
                        t.1 += t1;
                    }
                }
                Op::ChangeAny => {
                    for _ in 0..args.iters {
                        let (t0, mut tt) = time_it(|| tree.clone());
                        let (t1, _) = time_it(|| tt.change_any(&mut rng));
                        tree = tt;
                        t.0 += t0;
                        t.1 += t1;
                    }
                }
            }
        })
    }
    for t in &mut time {
        t.0 /= (args.iters * args.times) as u32;
        t.1 /= (args.iters * args.times) as u32;
    }
    time
}

fn case_complete(
    args: &Args,
    n: u32,
    rng: &mut XorShiftRng,
) -> (CompleteGraph, Vec<Edge<CompleteGraph>>) {
    let g = CompleteGraph::new(n as u32);
    let tree = if let Some(d) = args.diameter {
        let d = 2 + (d * (n - 3) as f32) as usize;
        random_sp_with_diameter(&g, d, rng)
    } else {
        random_sp(&g, rng)
    };
    (g, tree)
}

fn case_regular(
    args: &Args,
    n: u32,
    rng: &mut XorShiftRng,
) -> (StaticGraph, Vec<Edge<StaticGraph>>) {
    let g =
        StaticGraph::new_regular(args.regular.unwrap() as _, n as _, &mut *rng).expect("a graph");
    let tree = random_sp(&g, rng);
    (g, tree)
}

fn case_forest(
    args: &Args,
    n: u32,
    rng: &mut XorShiftRng,
) -> (StaticGraph, Vec<Edge<StaticGraph>>) {
    // partition the vertices in sqrt(n) components
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

    // create edges for each component
    if let Some(r) = args.regular {
        for sub in &sub_vertices {
            let t = StaticGraph::new_regular(r as _, sub.len(), &mut *rng).expect(&format!(
                "Cannot create an regular graph for r = {} and n = {}. Note that n * r must be even.",
                r,
                sub.len()
            ));
            for (i, j) in t.edges_ends() {
                b.add_edge(sub[i as usize] as usize, sub[j as usize] as usize)
            }
        }
    } else {
        for sub in &sub_vertices {
            for i in 0..sub.len() {
                for j in (i + 1)..sub.len() {
                    b.add_edge(sub[i] as usize, sub[j] as usize)
                }
            }
        }
    }

    // join the root of each component with r
    for sub in &sub_vertices {
        let v = sub[0];
        b.add_edge(r as usize, v as usize);
    }

    // create the graph
    let g = b.finalize();

    // generates a random tree in g
    let edges = vec(g.edges()).shuffled_with(rng);
    let tree = vec(g.kruskal().edges(g.out_edges(r).chain(edges)));

    (g, tree)
}

#[derive(Debug)]
struct Args {
    sizes: Vec<u32>,
    diameter: Option<f32>,
    forest: bool,
    regular: Option<u32>,
    ds: Ds,
    op: Op,
    times: u32,
    iters: u32,
    k: usize,
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
                    "nddr-edge",
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
            (@arg iters:
                --iters
                +takes_value
                default_value("10000")
                "The number of iteratively mutation")
            (@arg k:
                -k
                +takes_value
                default_value("1")
                "The k parameter for NDDR")
            (@arg forest:
                --forest
                "Test graphs that are forests (good for nddr)")
            (@arg regular:
                -r
                --regular
                +takes_value
                "Create r-regular graphs. Default is to generate complete (n-regular) graphs.")
            (@arg diameter:
                -d
                --diameter
                +takes_value
                "Diameter of the random trees [0, 1] - (0.0 = diameter 2, 1.0 = diamenter n - 1). Default is to generate trees with random diameter. Only valid for complete graphs.")
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
            "nddr-edge" => Ds::NddrEdge,
            "pred" => Ds::Predecessor,
            "pred2" => Ds::Predecessor2,
            _ => unreachable!(),
        },
        op: match matches.value_of("op").unwrap() {
            "change-pred" => Op::ChangePred,
            "change-any" => Op::ChangeAny,
            _ => unreachable!(),
        },
        forest: matches.is_present("forest"),
        regular: if matches.is_present("regular") {
            let r = value_t_or_exit!(matches, "regular", u32);
            if r < 3 {
                panic!("Invalid value for regular: {}", r)
            }
            Some(r)
        } else {
            None
        },
        diameter: if matches.is_present("diameter") {
            let d = value_t_or_exit!(matches, "diameter", f32);
            if d < 0.0 || d > 1.0 {
                panic!("Invalid value for diameter: {}", d)
            }
            Some(d)
        } else {
            None
        },
        times: value_t_or_exit!(matches, "times", u32),
        iters: value_t_or_exit!(matches, "iters", u32),
        k: value_t_or_exit!(matches, "k", usize),
        sizes: matches
            .values_of("sizes")
            .unwrap()
            .map(|x| x.parse::<u32>().unwrap())
            .collect(),
    }
}

#[derive(Debug, Copy, Clone, PartialEq)]
enum Ds {
    EulerTour,
    NddrAdj,
    NddrBalanced,
    NddrEdge,
    Predecessor,
    Predecessor2,
}

#[derive(Debug, Copy, Clone)]
enum Op {
    ChangePred,
    ChangeAny,
}
