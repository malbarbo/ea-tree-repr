extern crate ea_tree_repr;
extern crate fera;
extern crate rand;
extern crate tsplib;

#[macro_use]
extern crate clap;
#[macro_use]
extern crate log;

// system
use std::collections::HashSet;
use std::rc::Rc;

// external
use fera::ext::VecExt;
use fera::fun::{position_max_by_key, position_min_by_key, vec};
use fera::graph::algs::Kruskal;
use fera::graph::prelude::*;
use fera::graph::traverse::{Dfs, OnDiscoverTreeEdge};
use rand::Rng;

// local
use ea_tree_repr::{
    init_logger, new_rng_with_seed, progress, EulerTourTree, NddrAdjTree, NddrEdgeTree,
    PredecessorTree, PredecessorTree2, Tree,
};

pub fn main() {
    let args = args();

    init_logger(if args.quiet { "warn" } else { "info" });

    info!("{:#?}", args);

    let results = vec(progress(0..args.times).map(|_| run(&args)));
    let iters: usize = results.iter().map(|r| r.iter_fitness.len()).max().unwrap();
    let space = (iters - 1) as f64 / args.samples as f64;
    println!("iter fitness");
    for i in 0..(args.samples + 1) {
        let iter = ((f64::from(i) * space).ceil() as usize).min(iters - 1);
        let sum: u64 = results.iter().map(|r| r.iter_fitness[iter] as u64).sum();
        let fitness = sum as f64 / args.times as f64;
        let fitness_percentage = fitness / (args.size - 1) as f64;
        println!("{} {}", iter + 1, fitness_percentage);
    }
}

fn run(args: &Args) -> EAResult {
    let n = args.size;
    let m = (args.f * n) as u64;
    let (g, target, root) = if args.balanced {
        new_case_balanced(n, m, new_rng_with_seed(args.seed))
    } else {
        new_case(n, m, args.diameter, new_rng_with_seed(args.seed))
    };

    assert_eq!(n as usize, g.num_vertices());
    assert_eq!(m as usize, g.num_edges());

    let g = &Rc::new(g);

    match args.ds {
        Ds::EulerTour => run_ea::<EulerTourTree<_>>(g, root, &target, &args),
        Ds::NddrAdj => run_ea::<NddrAdjTree<_>>(g, root, &target, &args),
        Ds::NddrEdge => run_ea::<NddrEdgeTree<_>>(g, root, &target, &args),
        Ds::Predecessor => run_ea::<PredecessorTree<_>>(g, root, &target, &args),
        Ds::Predecessor2 => run_ea::<PredecessorTree2<_>>(g, root, &target, &args),
    }
}

fn run_ea<'a, T>(
    g: &Rc<StaticGraph>,
    root: Option<Vertex<StaticGraph>>,
    target: &'a TargetTree,
    args: &Args,
) -> EAResult
where
    T: Tree<StaticGraph>,
    Ind<'a, T>: Clone,
{
    let mut iter_fitness = vec![];
    let mut rng = new_rng_with_seed(args.seed);
    let mut edges = vec(g.edges());
    let mut tree = vec![];
    let mut pop = vec![];
    for _ in 0..args.pop_size {
        rng.shuffle(&mut edges);
        tree.clear();
        if let Some(root) = root {
            // put the root edges first so it will be on the tree
            tree.extend(
                g.kruskal()
                    .edges(g.out_edges(root).chain(edges.iter().cloned())),
            );
        } else {
            tree.extend(g.kruskal().edges(&edges));
        }
        if pop.is_empty() {
            pop.push(Ind::<T>::new(g.clone(), root, target, &tree, &mut rng));
        } else {
            // We create a new individual based on an existing one so tree data structures can
            // share internal state (like nddr::Data)
            let new = pop[0].set_edges(&tree);
            pop.push(new);
        }
    }

    let mut best = position_max_by_key(&pop, |ind| ind.fitness()).unwrap();

    info!("best = {:?}", pop[best].fitness());

    let mut best_iter = 0;
    for it in 0..args.max_num_iters() {
        let i = rng.gen_range(0, pop.len());
        let mut new = pop[i].clone();
        new.mutate(args.op, &mut rng);
        if pop.contains(&new) {
            iter_fitness.push(pop[best].fitness());
            continue;
        }
        if new.fitness() >= pop[i].fitness() {
            if new.fitness() > pop[best].fitness() {
                best_iter = it;
                best = i;
                info!("it = {}, best = {}", it, new.fitness());
                if args.stop_on_optimum && new.is_optimum() {
                    info!("optimum found");
                    pop[i] = new;
                    iter_fitness.push(pop[best].fitness());
                    break;
                }
            }
            pop[i] = new;
        } else {
            let j = position_min_by_key(&pop, |ind| ind.fitness()).unwrap();
            pop[j] = new;
            best = position_max_by_key(&pop, |ind| ind.fitness()).unwrap();
        }

        iter_fitness.push(pop[best].fitness());

        if it - best_iter >= args.max_num_iters_no_impr() {
            info!(
                "max number of iterations without improvement reached: {:?}",
                args.max_num_iters_no_impr
            );
            break;
        }
    }

    EAResult::new(iter_fitness, best_iter)
}

#[derive(Clone, Default)]
struct EAResult {
    iter_fitness: Vec<u32>,
    best_iter: u64,
}

impl EAResult {
    fn new(iter_fitness: Vec<u32>, best_iter: u64) -> EAResult {
        EAResult {
            iter_fitness,
            best_iter,
        }
    }
}

fn new_case<R: Rng>(
    n: u32,
    m: u64,
    diameter: Option<u32>,
    mut rng: R,
) -> (StaticGraph, TargetTree, Option<Vertex<StaticGraph>>) {
    let tree = if let Some(d) = diameter {
        StaticGraph::new_random_tree_with_diameter(n, d, &mut rng).unwrap()
    } else {
        StaticGraph::new_random_tree(n as usize, &mut rng)
    };

    let mut b = StaticGraph::builder(n as usize, m as usize);
    let mut edges = HashSet::new();
    for (u, v) in tree.edges_ends() {
        b.add_edge(u as usize, v as usize);
        edges.insert((u, v));
    }

    while edges.len() != m as usize {
        let u = rng.gen_range(0, n);
        let v = rng.gen_range(0, n);
        if u == v || edges.contains(&(u, v)) || edges.contains(&(v, u)) {
            continue;
        }
        b.add_edge(u as usize, v as usize);
        edges.insert((u, v));
    }

    let g = b.finalize();
    let target = TargetTree::new(&g, tree.edges_ends().map(|(u, v)| g.edge_by_ends(u, v)));

    (g, target, None)
}

fn new_case_balanced<R: Rng>(
    n: u32,
    m: u64,
    mut rng: R,
) -> (StaticGraph, TargetTree, Option<Vertex<StaticGraph>>) {
    let nsqrt = (n as f64).sqrt();
    let mut vertices = vec![vec![]; nsqrt.ceil() as usize - 1];
    let mut comp = vec![0; n as usize];
    let r = 0;
    for (i, v) in vec(1..n).shuffled_with(&mut rng).into_iter().enumerate() {
        let i = i % vertices.len();
        vertices[i].push(v);
        comp[v as usize] = i;
    }
    rng.shuffle(&mut vertices);

    let mut b = StaticGraph::builder(n as usize, m as usize);
    let mut edges = HashSet::new();

    // create each target subtree
    for sub in &vertices {
        let g = StaticGraph::new_random_tree(sub.len(), &mut rng);
        for (u, v) in g.edges_ends() {
            let u = sub[u as usize];
            let v = sub[v as usize];
            b.add_edge(u as usize, v as usize);
            edges.insert((u, v));
        }
    }

    // add the root edges
    for sub in &vertices {
        let v = sub[0];
        b.add_edge(r as usize, v as usize);
        edges.insert((r, v));
    }

    // salve the target tree
    let tree = edges.clone();

    // add the remaining edges
    while edges.len() != m as usize {
        let u = rng.gen_range(0, n);
        let v = rng.gen_range(0, n);
        if u == r
            || v == r
            || u == v
            || comp[u as usize] != comp[v as usize]
            || edges.contains(&(u, v))
            || edges.contains(&(v, u))
        {
            continue;
        }
        b.add_edge(u as usize, v as usize);
        edges.insert((u, v));
    }

    let g = b.finalize();
    let target = TargetTree::new(&g, tree.into_iter().map(|(u, v)| g.edge_by_ends(u, v)));

    (g, target, Some(r))
}

struct TargetTree(DefaultVertexPropMut<StaticGraph, OptionVertex<StaticGraph>>);

impl TargetTree {
    fn new<I>(g: &StaticGraph, edges: I) -> Self
    where
        I: IntoIterator<Item = Edge<StaticGraph>>,
    {
        let mut pred = g.default_vertex_prop(StaticGraph::vertex_none());
        g.spanning_subgraph(edges)
            .dfs(OnDiscoverTreeEdge(|e| {
                let (u, v) = g.ends(e);
                assert!(pred[v].into_option().is_none());
                pred[v] = Some(u).into();
            }))
            .run();
        TargetTree(pred)
    }

    fn contains(&self, u: Vertex<StaticGraph>, v: Vertex<StaticGraph>) -> bool {
        self.0[u].into_option() == Some(v) || self.0[v].into_option() == Some(u)
    }
}

#[derive(Clone)]
struct Ind<'a, T>
where
    T: Tree<StaticGraph>,
{
    target: &'a TargetTree,
    tree: T,
    hash: usize,
    fitness: u32,
}

impl<'a, T> PartialEq for Ind<'a, T>
where
    T: Tree<StaticGraph>,
{
    fn eq(&self, other: &Self) -> bool {
        self.hash == other.hash && self.fitness == other.fitness
    }
}

impl<'a, T> Ind<'a, T>
where
    T: Tree<StaticGraph>,
{
    fn new<R: Rng>(
        g: Rc<StaticGraph>,
        root: Option<Vertex<StaticGraph>>,
        target: &'a TargetTree,
        edges: &[Edge<StaticGraph>],
        rng: R,
    ) -> Self {
        let tree = T::new_with_fake_root(g, root, edges, rng);
        let mut ind = Ind {
            target,
            tree,
            hash: 0,
            fitness: 0,
        };
        ind.fitness = ind.comp_fitness();
        ind.hash = ind.comp_hash();
        ind
    }

    fn set_edges(&self, edges: &[Edge<StaticGraph>]) -> Self {
        let mut new = self.clone();
        new.tree.set_edges(edges);
        new.fitness = new.comp_fitness();
        new.hash = new.comp_hash();
        new
    }

    fn mutate<R: Rng>(&mut self, op: Op, rng: R) {
        let (ins, rem) = match op {
            Op::ChangeAny => self.tree.change_any(rng),
            Op::ChangePred => self.tree.change_pred(rng),
        };

        self.hash -= self.tree.graph().edge_index().get(rem);
        self.hash += self.tree.graph().edge_index().get(ins);

        let (u, v) = self.tree.graph().ends(ins);
        if self.target.contains(u, v) {
            self.fitness += 1;
        }

        let (u, v) = self.tree.graph().ends(rem);
        if self.target.contains(u, v) {
            self.fitness -= 1;
        }

        self.check();
    }

    fn fitness(&self) -> u32 {
        self.fitness
    }

    fn is_optimum(&self) -> bool {
        self.fitness == self.tree.graph().num_vertices() as u32 - 1
    }

    fn comp_fitness(&self) -> u32 {
        let g = self.tree.graph();
        let target = self.target;
        self.tree
            .edges()
            .into_iter()
            .filter(|&e| {
                let (u, v) = g.ends(e);
                target.contains(u, v)
            })
            .count() as u32
    }

    fn comp_hash(&self) -> usize {
        self.tree
            .edges()
            .into_iter()
            .map(|e| self.tree.graph().edge_index().get(e))
            .sum()
    }

    #[cfg(not(debug_assertions))]
    fn check(&self) {}

    #[cfg(debug_assertions)]
    fn check(&self) {
        assert_eq!(self.fitness, self.comp_fitness());
        assert_eq!(self.hash, self.comp_hash());
    }
}

#[derive(Debug)]
struct Args {
    ds: Ds,
    op: Op,
    times: u32,
    size: u32,
    // optional parameters
    samples: u32,
    f: u32,
    diameter: Option<u32>,
    balanced: bool,
    stop_on_optimum: bool,
    seed: u32,
    pop_size: u32,
    max_num_iters: Option<u64>,
    max_num_iters_no_impr: Option<u64>,
    quiet: bool,
}

impl Args {
    fn max_num_iters(&self) -> u64 {
        self.max_num_iters.unwrap_or(u64::max_value())
    }

    fn max_num_iters_no_impr(&self) -> u64 {
        self.max_num_iters_no_impr.unwrap_or(u64::max_value())
    }
}

fn args() -> Args {
    let app = clap_app!(
        ("omt") =>
        (version: crate_version!())
        (about: "Evolutionary algorithm for the one-max tree problem")
        (author: crate_authors!())
        (@arg ds:
             +required
             possible_values(&[
                "euler-tour",
                "nddr-adj",
                "nddr-edge",
                "pred",
                "pred2",
             ])
             "Data structure")
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
        (@arg size:
             +required
             "Number of vertices")
        (@arg seed:
            --seed
             +takes_value
             "Seed used in the random number generator. A random value is used if none is specified")
        (@arg pop_size:
             --pop_size
             +takes_value
             default_value("10")
             "Number of individual in the population")
        (@arg max_num_iters:
             --max_num_iters
             +takes_value
             "Maximum number of iterations")
        (@arg max_num_iters_no_impr:
             --max_num_iters_no_impr
             +takes_value
             "Maximum number of iterations without improvement")
        (@arg quiet:
             --quiet
             "Only prints the final solution")
        (@arg samples:
             --samples
             default_value("100")
             "The number of samples to show")
        (@arg f:
             -f
             +takes_value
             default_value("3")
             "Edge factor. The number of edges is f * n")
        (@arg d:
             -d
             --diameter
             +takes_value
             "The diameter of the target tree")
        (@arg balanced:
             --balanced
             "Create a target tree composed of sqrt(n) subtrees")
        (@arg stop_on_optimum:
             --stop_on_optimum
             "Stop when the optimum solution is found")
    );

    let matches = app.get_matches();

    Args {
        ds: match matches.value_of("ds").unwrap() {
            "euler-tour" => Ds::EulerTour,
            "nddr-adj" => Ds::NddrAdj,
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
        times: value_t_or_exit!(matches, "times", u32),
        size: value_t_or_exit!(matches, "size", u32),
        seed: if matches.is_present("seed") {
            value_t_or_exit!(matches, "seed", u32)
        } else {
            rand::thread_rng().gen()
        },
        pop_size: value_t_or_exit!(matches, "pop_size", u32),
        samples: value_t_or_exit!(matches, "samples", u32),
        max_num_iters: if matches.is_present("max_num_iters") {
            Some(value_t_or_exit!(matches, "max_num_iters", u64))
        } else {
            None
        },
        max_num_iters_no_impr: if matches.is_present("max_num_iters_no_impr") {
            Some(value_t_or_exit!(matches, "max_num_iters_no_impr", u64))
        } else {
            None
        },
        quiet: matches.is_present("quiet"),
        f: value_t_or_exit!(matches, "f", u32),
        diameter: if matches.is_present("d") {
            Some(value_t_or_exit!(matches, "d", u32))
        } else {
            None
        },
        balanced: matches.is_present("balanced"),
        stop_on_optimum: matches.is_present("stop_on_optimum"),
    }
}

#[derive(Copy, Clone, Debug)]
enum Ds {
    EulerTour,
    NddrAdj,
    NddrEdge,
    Predecessor,
    Predecessor2,
}

#[derive(Copy, Clone, Debug)]
enum Op {
    ChangePred,
    ChangeAny,
}
