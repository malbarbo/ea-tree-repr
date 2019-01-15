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
use fera::graph::algs::{Components, Kruskal, Paths};
use fera::graph::prelude::*;
use fera::graph::sum_prop;
use fera::graph::traverse::{Dfs, OnDiscoverTreeEdge};
use rand::Rng;

// local
use ea_tree_repr::{
    init_logger, new_rng_with_seed, CowNestedArrayVertexProp, EulerTourTree, NddrAdjTree,
    NddrEdgeTree, PredecessorTree, PredecessorTree2, Tree,
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
    info!("is connected: {}", g.is_connected());

    // run the solver
    let (edges, degree) = match args.ds {
        Ds::EulerTour => {
            run::<EulerTourTree<_>, CowNestedArrayVertexProp<StaticGraph, u32>>(g, &args)
        }
        Ds::NddrAdj => run::<NddrAdjTree<_>, CowNestedArrayVertexProp<StaticGraph, u32>>(g, &args),
        Ds::NddrEdge => {
            run::<NddrEdgeTree<_>, CowNestedArrayVertexProp<StaticGraph, u32>>(g, &args)
        }
        Ds::Predecessor => {
            run::<PredecessorTree<_>, DefaultVertexPropMut<StaticGraph, u32>>(g, &args)
        }
        Ds::Predecessor2 => {
            run::<PredecessorTree2<_>, CowNestedArrayVertexProp<StaticGraph, u32>>(g, &args)
        }
    };

    // show the solution

    // number of leafs
    let v1 = count_vertex_deg(&**g, &degree, |d| d == 1);
    // number of branch vertices
    let branches = count_vertex_deg(&**g, &degree, |d| d > 2);
    // sum of the degree of the branch vertices
    let sum: u32 = sum_prop(&degree, g.vertices().filter(|v| degree[*v] > 2));
    match args.kind {
        Kind::Mbv | Kind::Mds | Kind::Ml => {
            let obj = match args.kind {
                Kind::Mbv => branches,
                Kind::Mds => sum,
                Kind::Ml => v1,
                _ => panic!(),
            };
            print!("{} {}", args.input, obj);
            for e in edges {
                let (u, v) = g.ends(e);
                print!(" {}-{}", u + 1, v + 1)
            }
            println!();
        }
        Kind::Cycle | Kind::Path => {
            if count_vertex_deg(&**g, &degree, |d| d > 2) != 0 {
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
}

fn run<T, D>(
    g: &Rc<StaticGraph>,
    args: &Args,
) -> (
    Vec<Edge<StaticGraph>>,
    DefaultVertexPropMut<StaticGraph, u32>,
)
where
    T: Tree<StaticGraph>,
    D: VertexPropMutNew<StaticGraph, u32>,
    Ind<T, D>: Clone,
{
    let mut rng = new_rng_with_seed(args.seed);
    let mut edges = vec(g.edges());
    let mut tree = vec![];
    let mut pop = vec![];
    for _ in 0..args.pop_size {
        rng.shuffle(&mut edges);
        tree.clear();
        tree.extend(g.kruskal().edges(&edges));
        if pop.is_empty() {
            pop.push(Ind::<T, D>::new(
                args.kind,
                g.clone(),
                &tree,
                g.vertex_prop(0),
                &mut rng,
            ));
        } else {
            // We create a new individual based on an existing one so tree data structures can
            // share internal state (like nddr::Data)
            let new = pop[0].set_edges(&tree, g.vertex_prop(0));
            pop.push(new);
        }
    }

    let mut best = position_max_by_key(&pop, |ind| ind.fitness()).unwrap();

    if pop[best].is_optimum() {
        return (pop[best].tree.edges(), pop[best].degree());
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
        if new.fitness() >= pop[i].fitness() {
            if new.fitness() > pop[best].fitness() {
                last_it_impr = it;
                best = i;
                info!("it = {}, best = {:?}", it, new.value());
                if new.is_optimum() {
                    pop[i] = new;
                    break;
                }
            }
            pop[i] = new;
        } else {
            let j = position_min_by_key(&pop, |ind| ind.fitness()).unwrap();
            pop[j] = new;
            best = position_max_by_key(&pop, |ind| ind.fitness()).unwrap();
        }

        if it - last_it_impr >= args.max_num_iters_no_impr.unwrap_or(u64::max_value()) {
            info!(
                "max number of iterations without improvement reached: {:?}",
                args.max_num_iters_no_impr
            );
            break;
        }
    }

    (pop[best].tree.edges(), pop[best].degree())
}

fn count_vertex_deg<G, D, F>(g: &G, d: &D, fun: F) -> u32
where
    G: Graph,
    D: VertexProp<G, u32>,
    F: Fn(u32) -> bool,
{
    g.vertices().filter(|v| fun(d[*v])).count() as _
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
    v1: u32,
    v2: u32,
    kind: Kind,
}

impl<T, D> PartialEq for Ind<T, D>
where
    T: Tree<StaticGraph>,
    D: VertexPropMut<StaticGraph, u32>,
{
    fn eq(&self, other: &Self) -> bool {
        self.hash == other.hash && self.v1 == other.v1 && self.v2 == other.v2
    }
}

impl<T, D> Ind<T, D>
where
    T: Tree<StaticGraph>,
    D: VertexPropMut<StaticGraph, u32>,
{
    fn new_(kind: Kind, tree: T, mut degree: D, edges: &[Edge<StaticGraph>]) -> Self {
        let g = tree.graph().clone();
        let mut hash = 0;
        for &e in edges {
            let (u, v) = g.ends(e);
            degree[u] += 1;
            degree[v] += 1;
            hash += g.edge_index().get(e);
        }
        let v1 = count_vertex_deg(&*g, &degree, |d| d == 1);
        let v2 = count_vertex_deg(&*g, &degree, |d| d == 2);
        Ind {
            tree,
            degree,
            hash,
            v1,
            v2,
            kind,
        }
    }

    fn new<R: Rng>(
        kind: Kind,
        g: Rc<StaticGraph>,
        edges: &[Edge<StaticGraph>],
        degree: D,
        rng: R,
    ) -> Self {
        Self::new_(kind, T::new(g, edges, rng), degree, edges)
    }

    fn set_edges(&self, edges: &[Edge<StaticGraph>], degree: D) -> Self {
        let mut tree = self.tree.clone();
        tree.set_edges(edges);
        Self::new_(self.kind, tree, degree, edges)
    }

    fn degree(&self) -> DefaultVertexPropMut<StaticGraph, u32> {
        let g = self.tree.graph();
        let mut deg = g.default_vertex_prop(0);
        deg.set_values_from(g.vertices(), &self.degree);
        deg
    }

    fn mutate<R: Rng>(&mut self, op: Op, rng: R) {
        let (ins, rem) = match op {
            Op::ChangeAny => self.tree.change_any(rng),
            Op::ChangePred => self.tree.change_pred(rng),
        };

        self.hash -= self.tree.graph().edge_index().get(rem);
        self.hash += self.tree.graph().edge_index().get(ins);

        let (a, b) = self.tree.graph().ends(rem);
        if self.degree[a] == 2 {
            self.v1 += 1;
            self.v2 -= 1;
        }
        if self.degree[a] == 3 {
            self.v2 += 1;
        }
        if self.degree[b] == 2 {
            self.v1 += 1;
            self.v2 -= 1;
        }
        if self.degree[b] == 3 {
            self.v2 += 1;
        }
        self.degree[a] -= 1;
        self.degree[b] -= 1;

        let (a, b) = self.tree.graph().ends(ins);
        if self.degree[a] == 1 {
            self.v1 -= 1;
            self.v2 += 1;
        }
        if self.degree[a] == 2 {
            self.v2 -= 1;
        }
        if self.degree[b] == 1 {
            self.v1 -= 1;
            self.v2 += 1;
        }
        if self.degree[b] == 2 {
            self.v2 -= 1;
        }
        self.degree[a] += 1;
        self.degree[b] += 1;

        self.check();
    }

    fn value(&self) -> (u32, u32, u32) {
        (self.v1, self.v2, self.branches())
    }

    fn fitness(&self) -> i32 {
        // Relations, models and a memetic approach for three degree-dependet spanning tree
        // problems. EJOUR, 2014. C Cerrone, R Cerulli, A Raiconi
        // pg 445
        let (alfa, beta) = match self.kind {
            Kind::Mbv => (1, 1),
            Kind::Mds => (1, 2),
            Kind::Ml => (-1, 0),
            _ => (-1, 1),
        };
        alfa * self.v1 as i32 + beta * self.v2 as i32
    }

    fn is_optimum(&self) -> bool {
        self.branches() == 0
    }

    fn branches(&self) -> u32 {
        self.tree.graph().num_vertices() as u32 - self.v1 - self.v2
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
        assert_eq!(count_vertex_deg(&**g, &deg, |d| d == 1), self.v1);
        assert_eq!(count_vertex_deg(&**g, &deg, |d| d == 2), self.v2);
        assert_eq!(count_vertex_deg(&**g, &deg, |d| d > 2), self.branches());
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
        ("mbv") =>
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
                    "mds",
                    "ml",
                    "path",
                    "cycle",
                ])
                "The kind of problem to solve: Minimum branch vertices, Minimum degree sum, Minimum leaves, Hamiltonian path or Hamiltonian cycle")
            (@arg quiet:
                --quiet
                "Only prints the final solution")
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
            "mds" => Kind::Mds,
            "ml" => Kind::Ml,
            "path" => Kind::Path,
            "cycle" => Kind::Cycle,
            _ => unreachable!(),
        },
        quiet: matches.is_present("quiet"),
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
        input: matches.value_of("input").unwrap().to_string(),
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

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum Kind {
    Cycle,
    Mbv,
    Mds,
    Ml,
    Path,
}
