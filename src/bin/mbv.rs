extern crate ea_tree_repr;
extern crate fera;
extern crate rand;
extern crate tsplib;

#[macro_use]
extern crate clap;
#[macro_use]
extern crate log;

// system
use std::collections::BTreeMap;
use std::rc::Rc;

// external
use fera::fun::{position_max_by_key, position_min_by_key, vec};
use fera::graph::algs::{Kruskal, Paths};
use fera::graph::prelude::*;
use fera::graph::traverse::{Dfs, OnDiscoverTreeEdge};
use rand::Rng;

// local
use ea_tree_repr::{
    init_logger, new_rng, CowNestedArrayVertexProp, EulerTourTree, NddrAdjTree, NddrBalancedTree,
    PredecessorTree, PredecessorTree2, Tree,
};

pub fn main() {
    let args = args();

    init_logger(if args.quiet { "warn" } else { "info" });

    info!("{:#?}", args);

    let instance = tsplib::read(&args.input).expect("Fail to parse input file");

    assert_eq!(
        Some(tsplib::Type::Hcp),
        instance.type_,
        "Only HCP instances are supported"
    );

    let mut builder = StaticGraph::builder(instance.dimension, 0);

    if let Some(tsplib::EdgeData::EdgeList(edges)) = instance.edge_data {
        for (u, v) in edges {
            builder.add_edge(u - 1, v - 1);
        }
    } else {
        panic!("Only EDGE_DATA_FORMAT : EDGE_LIST is supported")
    }

    let g = builder.finalize();
    let g = if args.kind == Kind::Cycle {
        Rc::new(transform(&g))
    } else {
        Rc::new(g)
    };
    let g = &g;

    let mut count = BTreeMap::new();
    for u in g.vertices() {
        *count.entry(g.out_degree(u)).or_insert(0) += 1;
    }

    info!(
        "graph size\n  n = {}\n  m = {}",
        g.num_vertices(),
        g.num_edges()
    );
    info!("degree distribution");
    for (d, c) in count {
        info!("{:2} = {:2}", d, c);
    }

    let (edges, branches) = match args.ds {
        Ds::EulerTour => {
            run::<EulerTourTree<_>, CowNestedArrayVertexProp<StaticGraph, u32>>(g, &args)
        }
        Ds::NddrAdj => run::<NddrAdjTree<_>, CowNestedArrayVertexProp<StaticGraph, u32>>(g, &args),
        Ds::NddrBalanced => {
            run::<NddrBalancedTree<_>, CowNestedArrayVertexProp<StaticGraph, u32>>(g, &args)
        }
        Ds::Predecessor => {
            run::<PredecessorTree<_>, DefaultVertexPropMut<StaticGraph, u32>>(g, &args)
        }
        Ds::Predecessor2 => {
            run::<PredecessorTree2<_>, CowNestedArrayVertexProp<StaticGraph, u32>>(g, &args)
        }
    };

    if args.kind == Kind::Mbv {
        print!("{} {}", args.input, branches);
        for e in edges {
            let (u, v) = g.ends(e);
            print!(" {}-{}", u + 1, v + 1)
        }
        println!();
    } else {
        if branches != 0 {
            panic!("cannot find an {:?}", args.kind);
        }
        let mut path = path(g, &edges);
        if args.kind == Kind::Cycle {
            transform_path(g, &mut path);
        }
        print!("{}", args.input);
        for u in path.iter().map(|&e| g.source(e)) {
            print!(" {}", u + 1);
        }
        println!(" {}", g.target(*path.last().unwrap()) + 1);
    }
}

fn run<T, D>(g: &Rc<StaticGraph>, args: &Args) -> (Vec<Edge<StaticGraph>>, u32)
where
    T: Tree<StaticGraph>,
    D: VertexPropMutNew<StaticGraph, u32>,
    Ind<T, D>: Clone,
{
    let mut rng = new_rng(args.seed);
    let mut edges = vec(g.edges());
    let mut tree = vec![];
    let mut pop = vec![];
    for _ in 0..args.pop_size {
        rng.shuffle(&mut edges);
        tree.clear();
        tree.extend(g.kruskal().edges(&edges));
        if pop.is_empty() {
            pop.push(Ind::<T, D>::new(
                g.clone(),
                &tree,
                g.vertex_prop(0),
                &mut rng,
            ));
        } else {
            // We create a new individual based on an existing one so tree data structures can
            // share internal state
            let new = pop[0].set_edges(&tree, g.vertex_prop(0));
            pop.push(new);
        }
    }

    let mut best = position_min_by_key(&pop, |ind| ind.fitness()).unwrap();

    if pop[best].is_optimum() {
        return (pop[best].tree.edges(), pop[best].branches);
    }

    info!("best = {:?}", pop[best].value());

    let mut last_it_impr = 0;
    for it in 0..=args.max_num_iters.unwrap_or(u64::max_value()) {
        let i = rng.gen_range(0, pop.len());
        let mut new = pop[i].clone();
        new.mutate(args.op, &mut rng);
        if pop.contains(&new) {
            continue;
        }
        if new.fitness() <= pop[i].fitness() {
            if new.fitness() < pop[best].fitness() {
                last_it_impr = it;
                best = i;
                info!("it = {}, best = {}/{}", it, new.value().0, new.value().1);
                if new.is_optimum() {
                    pop[i] = new;
                    break;
                }
            }
            pop[i] = new;
        } else {
            let j = position_max_by_key(&pop, |ind| ind.fitness()).unwrap();
            pop[j] = new;
            best = position_min_by_key(&pop, |ind| ind.fitness()).unwrap();
        }

        if it - last_it_impr >= args.max_num_iters_no_impr.unwrap_or(u64::max_value()) {
            info!(
                "max number of iterations without improvement reached: {:?}",
                args.max_num_iters_no_impr
            );
            break;
        }
    }

    (pop[best].tree.edges(), pop[best].branches)
}

#[derive(Clone)]
struct Ind<T, D>
where
    T: Tree<StaticGraph>,
    D: VertexPropMut<StaticGraph, u32>,
{
    tree: T,
    degree: D,
    hash: usize,
    branches: u32,
    leafs: u32,
}

impl<T, D> PartialEq for Ind<T, D>
where
    T: Tree<StaticGraph>,
    D: VertexPropMut<StaticGraph, u32>,
{
    fn eq(&self, other: &Self) -> bool {
        self.hash == other.hash && self.branches == other.branches && self.leafs == other.leafs
    }
}

impl<T, D> Ind<T, D>
where
    T: Tree<StaticGraph>,
    D: VertexPropMut<StaticGraph, u32>,
{
    fn new_(tree: T, mut degree: D, edges: &[Edge<StaticGraph>]) -> Self {
        let g = tree.graph().clone();
        let mut hash = 0;
        for &e in edges {
            let (u, v) = g.ends(e);
            degree[u] += 1;
            degree[v] += 1;
            hash += g.edge_index().get(e);
        }
        let branches = g.vertices().filter(|v| degree[*v] > 2).count() as _;
        let leafs = g.vertices().filter(|v| degree[*v] == 1).count() as _;
        Ind {
            tree,
            degree,
            hash,
            branches,
            leafs,
        }
    }

    fn new<R: Rng>(g: Rc<StaticGraph>, edges: &[Edge<StaticGraph>], degree: D, rng: R) -> Self {
        Self::new_(T::new(g, edges, rng), degree, edges)
    }

    fn set_edges(&self, edges: &[Edge<StaticGraph>], degree: D) -> Self {
        let mut tree = self.tree.clone();
        tree.set_edges(edges);
        Self::new_(tree, degree, edges)
    }

    fn mutate<R: Rng>(&mut self, op: Op, rng: R) {
        let (ins, rem) = match op {
            Op::ChangeAny => self.tree.change_any(rng),
            Op::ChangePred => self.tree.change_pred(rng),
        };

        self.hash -= self.tree.graph().edge_index().get(rem);
        self.hash += self.tree.graph().edge_index().get(ins);

        let (a, b) = self.tree.graph().ends(rem);
        if self.degree[a] == 3 {
            self.branches -= 1;
        }
        if self.degree[a] == 2 {
            self.leafs += 1;
        }
        if self.degree[b] == 3 {
            self.branches -= 1;
        }
        if self.degree[b] == 2 {
            self.leafs += 1;
        }
        self.degree[a] -= 1;
        self.degree[b] -= 1;

        let (a, b) = self.tree.graph().ends(ins);
        if self.degree[a] == 2 {
            self.branches += 1;
        }
        if self.degree[a] == 1 {
            self.leafs -= 1;
        }
        if self.degree[b] == 2 {
            self.branches += 1;
        }
        if self.degree[b] == 1 {
            self.leafs -= 1;
        }
        self.degree[a] += 1;
        self.degree[b] += 1;

        self.check();
    }

    fn value(&self) -> (u32, u32) {
        (self.branches, self.leafs)
    }

    fn fitness(&self) -> (u32, u32) {
        (self.leafs, self.branches)
    }

    fn is_optimum(&self) -> bool {
        self.branches == 0
    }

    #[cfg(not(debug_assertions))]
    fn check(&self) {}

    #[cfg(debug_assertions)]
    fn check(&self) {
        use fera::graph::algs::Degrees;
        let g = self.tree.graph();
        let edges = self.tree.edges();
        let deg = g.degree_spanning_subgraph(&edges);
        for u in g.vertices() {
            assert_eq!(deg[u], self.degree[u]);
        }
        assert_eq!(
            edges.iter().map(|&e| g.edge_index().get(e)).sum::<usize>(),
            self.hash
        );
        assert_eq!(
            self.tree
                .graph()
                .vertices()
                .filter(|v| self.degree[*v] > 2)
                .count() as u32,
            self.branches
        );
        assert_eq!(
            self.tree
                .graph()
                .vertices()
                .filter(|v| self.degree[*v] == 1)
                .count() as u32,
            self.leafs
        );
    }
}

fn path(g: &StaticGraph, edges: &[Edge<StaticGraph>]) -> Vec<Edge<StaticGraph>> {
    let mut path = vec![];
    let sub = g.edge_induced_subgraph(edges);
    let r = sub.vertices().find(|v| sub.out_degree(*v) == 1).unwrap();
    sub.dfs(OnDiscoverTreeEdge(|e| path.push(e))).root(r).run();
    assert!(g.is_path(&path));
    path
}

// https://math.stackexchange.com/questions/7130/reduction-from-hamiltonian-cycle-to-hamiltonian-path
fn transform(g: &StaticGraph) -> StaticGraph {
    let n = g.num_vertices();
    let v = n - 1;
    let v_ = n;
    let s = n + 1;
    let t = n + 2;
    let mut builder = StaticGraph::builder(n + 3, 0);
    for (u, v) in g.ends(g.edges()) {
        builder.add_edge(u as _, v as _);
    }
    for w in g.out_neighbors(v as _) {
        builder.add_edge(v_, w as _);
    }
    builder.add_edge(s, v);
    builder.add_edge(t, v_);
    builder.add_edge(v, v_);
    builder.finalize()
}

fn transform_path(g: &StaticGraph, path: &mut Vec<Edge<StaticGraph>>) {
    let n = g.num_vertices() as u32;
    let t = n - 1;
    let s = n - 2;
    let v_ = n - 3;
    let v = n - 4;
    let last = path.pop().unwrap();
    let first = path.remove(0);
    assert!(s == g.source(first) || t == g.source(first));
    assert!(s == g.target(last) || t == g.target(last));
    if v_ == g.source(*path.first().unwrap()) {
        let x = g.target(*path.first().unwrap());
        *path.first_mut().unwrap() = g.edge_by_ends(v, x);
    } else {
        assert_eq!(v_, g.target(*path.last().unwrap()));
        let x = g.source(*path.last().unwrap());
        *path.last_mut().unwrap() = g.edge_by_ends(x, v);
    }
}

#[derive(Debug)]
struct Args {
    ds: Ds,
    op: Op,
    input: String,
    seed: u32,
    pop_size: u32,
    max_num_iters: Option<u64>,
    max_num_iters_no_impr: Option<u64>,
    kind: Kind,
    quiet: bool,
}

fn args() -> Args {
    let app = clap_app!(
        ("hcp") =>
            (version: crate_version!())
            (about: crate_description!())
            (author: crate_authors!())
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
                "Maximum number of iterations without improvements")
            (@arg kind:
                --kind
                +takes_value
                default_value("mbv")
                possible_values(&[
                    "mbv",
                    "path",
                    "cycle",
                ])
                "The kind of problem to solve: Minimum branch vertices, Hamiltonian path or Hamiltonian cycle")
            (@arg quiet:
                --quiet
                "Only prints the final solution")
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
            (@arg input:
                +required
                "Input file graph (an HCP tsplib instance)")
    );

    let matches = app.get_matches();

    Args {
        seed: if matches.is_present("seed") {
            value_t_or_exit!(matches, "seed", u32)
        } else {
            rand::thread_rng().gen()
        },
        pop_size: value_t_or_exit!(matches.value_of("pop_size"), u32),
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
        kind: match matches.value_of("kind").unwrap() {
            "mbv" => Kind::Mbv,
            "path" => Kind::Path,
            "cycle" => Kind::Cycle,
            _ => unreachable!(),
        },
        quiet: matches.is_present("quiet"),
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
        input: matches.value_of("input").unwrap().to_string(),
    }
}

#[derive(Copy, Clone, Debug)]
enum Ds {
    EulerTour,
    NddrAdj,
    NddrBalanced,
    Predecessor,
    Predecessor2,
}

#[derive(Copy, Clone, Debug)]
enum Op {
    ChangePred,
    ChangeAny,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum Kind {
    Mbv,
    Path,
    Cycle,
}
