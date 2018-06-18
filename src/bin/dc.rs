extern crate ea_tree_repr;
extern crate fera;
extern crate rand;

#[macro_use]
extern crate clap;
#[macro_use]
extern crate log;

// external
use fera::fun::{position_max_by_key, position_min_by_key, vec};
use fera::graph::algs::Kruskal;
use fera::graph::choose::Choose;
use fera::graph::prelude::*;
use rand::prelude::*;
use rand::distributions::Normal;

// system
use std::rc::Rc;

// local
use ea_tree_repr::{init_logger, new_rng_with_seed, PredecessorTree, TargetEdge, Tree};

const SCALE: f64 = 10_000_000_000.0;

pub fn main() {
    let args = args();

    init_logger(if args.quiet { "warn" } else { "info" });

    info!("{:#?}", args);

    let (g, w) = read(&args.input);
    let (it, weight, edges) = run(&Rc::new(g), &w, &args);
    info!("it = {}, best = {}", it, weight as f64 / SCALE);
    print!("{}", weight as f64 / SCALE);
    for (u, v) in g.ends(edges) {
        print!(" {:?}-{:?}", u, v);
    }
    println!()
}

fn run(
    g: &Rc<CompleteGraph>,
    w: &DefaultEdgePropMut<CompleteGraph, u64>,
    args: &Args,
) -> (u64, u64, Vec<Edge<CompleteGraph>>) {
    let mut rng = new_rng_with_seed(args.seed);
    let mut edges = vec(g.edges());

    // Initialize the population
    let mut pop = vec![];
    for _ in 0..args.pop_size {
        rng.shuffle(&mut edges);
        if pop.is_empty() {
            pop.push(Ind::new(Rc::clone(g), w, args.dc, &edges));
        } else {
            // We create a new individual based on an existing one so tree data structures can
            // share internal state
            let mut new = pop[0].clone();
            new.set_edges(w, args.dc, &edges);
            pop.push(new);
        }
    }

    // Saves the best
    let mut best = position_min_by_key(&pop, |ind| ind.fitness()).unwrap();

    info!("best = {:?}", pop[best].weight);

    let mut op = InsertRemove::new(
        g.num_vertices(),
        args.dc,
        vec(g.edges()).sorted_by_prop(w),
        args.beta,
        rng.clone(),
    );

    let mut it_best = 0;
    for it in 0..=args.max_num_iters.unwrap_or(u64::max_value()) {
        let i = rng.gen_range(0, pop.len());
        let mut new = pop[i].clone();

        if args.op == Op::ChangePred {
            op.change_pred(w, &mut new);
        } else {
            op.change_any(w, &mut new);
        }
        if pop.contains(&new) {
            continue;
        }

        if new.fitness() <= pop[i].fitness() {
            if new.fitness() < pop[best].fitness() {
                it_best = it;
                best = i;
                info!("it = {}, best = {:?}", it, new.weight);
            }
            pop[i] = new;
        } else {
            let j = position_max_by_key(&pop, |ind| ind.fitness()).unwrap();
            pop[j] = new;
            best = position_min_by_key(&pop, |ind| ind.fitness()).unwrap();
        }

        if it - it_best >= args.max_num_iters_no_impr.unwrap_or(u64::max_value()) {
            info!(
                "max number of iterations without improvement reached: {:?}",
                args.max_num_iters_no_impr
            );
            break;
        }
    }

    (it_best, pop[best].weight, pop[best].tree.edges())
}

#[derive(Clone)]
struct Ind<G>
where
    G: Graph + WithVertexIndexProp,
    DefaultVertexPropMut<G, u32>: Clone,
{
    tree: PredecessorTree<G>,
    deg: DefaultVertexPropMut<G, u32>,
    weight: u64,
}

impl<G> PartialEq for Ind<G>
where
    G: Graph + WithVertexIndexProp,
    DefaultVertexPropMut<G, u32>: Clone,
{
    fn eq(&self, other: &Self) -> bool {
        self.weight == other.weight
    }
}

impl<G> Ind<G>
where
    G: Graph + WithVertexIndexProp,
    DefaultVertexPropMut<G, u32>: Clone,
{
    fn new(g: Rc<G>, w: &DefaultEdgePropMut<G, u64>, dc: u32, edges: &[Edge<G>]) -> Self {
        let deg = g.vertex_prop(0u32);
        let mut ind = Self {
            tree: PredecessorTree::from_iter(g, None),
            deg,
            weight: 0,
        };
        ind.set_edges(w, dc, edges);
        ind
    }

    fn set_edges(&mut self, w: &DefaultEdgePropMut<G, u64>, dc: u32, edges: &[Edge<G>]) {
        let mut weight = 0;
        let g = self.tree.graph().clone();
        let deg = &mut self.deg;
        deg.set_values(g.vertices(), 0);
        self.tree
            .set_edges(g.kruskal().edges(edges).visitor(|g: &G, e, _: &mut _| {
                let (u, v) = g.ends(e);
                if deg[u] < dc && deg[v] < dc {
                    weight += w[e];
                    deg[u] += 1;
                    deg[v] += 1;
                    true
                } else {
                    false
                }
            }));
        self.weight = weight;
    }

    fn update_deg_weight(&mut self, ins: Edge<G>, rem: Edge<G>, w: &DefaultEdgePropMut<G, u64>) {
        let g = self.tree.graph();
        self.weight -= w[rem];
        self.weight += w[ins];

        let (u, v) = g.ends(ins);
        self.deg[u] += 1;
        self.deg[v] += 1;

        let (u, v) = g.ends(rem);
        self.deg[u] -= 1;
        self.deg[v] -= 1;
    }

    fn fitness(&self) -> u64 {
        self.weight
    }
}

struct InsertRemove<G, R>
where
    G: Graph,
    R: Rng,
{
    dc: u32,
    edges: Vec<Edge<G>>,
    normal: Option<Normal>,
    rng: R,
}

impl<G, R> InsertRemove<G, R>
where
    G: Graph + WithVertexIndexProp + Choose,
    DefaultVertexPropMut<G, u32>: Clone,
    R: Rng,
{
    pub fn new(n: usize, dc: u32, edges: Vec<Edge<G>>, beta: Option<f64>, rng: R) -> Self {
        let normal = beta.map(|beta| Normal::new(0.0, beta * (n as f64)));
        InsertRemove {
            dc,
            edges,
            normal,
            rng,
        }
    }

    pub fn change_pred(&mut self, w: &DefaultEdgePropMut<G, u64>, ind: &mut Ind<G>) {
        let g = ind.tree.graph().clone();
        let mut ins;
        let rem;
        loop {
            ins = self.choose_edge(&ind.deg, &ind.tree);
            let (u, v) = g.ends(ins);

            if ind.tree.is_ancestor_of(u, v) {
                if ind.deg[u] < self.dc {
                    ins = g.reverse(ins);
                    rem = ind.tree.set_pred_edge(v, u);
                    break;
                }
            } else if ind.tree.is_ancestor_of(v, u) {
                if ind.deg[v] < self.dc {
                    rem = ind.tree.set_pred_edge(u, v);
                    break;
                }
            } else {
                match (ind.deg[u] < self.dc, ind.deg[v] < self.dc) {
                    (false, false) => panic!(),
                    (false, true) => {
                        rem = ind.tree.set_pred_edge(u, v);
                    }
                    (true, false) => {
                        ins = g.reverse(ins);
                        rem = ind.tree.set_pred_edge(v, u);
                    }
                    (true, true) => {
                        if self.rng.gen() {
                            rem = ind.tree.set_pred_edge(u, v);
                        } else {
                            ins = g.reverse(ins);
                            rem = ind.tree.set_pred_edge(v, u);
                        }
                    }
                }
                break;
            }
        }

        ind.update_deg_weight(ins, rem.unwrap(), w)
    }

    pub fn change_any(&mut self, w: &DefaultEdgePropMut<G, u64>, ind: &mut Ind<G>) {
        let g = ind.tree.graph().clone();
        let ins = self.choose_edge(&ind.deg, &ind.tree);
        let rem = {
            let (u, v) = g.ends(ins);
            let e = match (ind.deg[u] < self.dc, ind.deg[v] < self.dc) {
                (false, false) => panic!(),
                (false, true) => TargetEdge::Last,
                (true, false) => TargetEdge::First,
                (true, true) => TargetEdge::Any,
            };
            ind.tree.insert_remove(ins, e, &mut self.rng)
        };
        ind.update_deg_weight(ins, rem, w);
    }

    fn choose_edge<Deg>(&mut self, deg: &Deg, tree: &PredecessorTree<G>) -> Edge<G>
    where
        Deg: VertexPropMut<G, u32>,
    {
        let g = tree.graph();
        loop {
            let e = self._choose_edge();
            let (u, v) = g.ends(e);
            if !tree.contains(e) && (deg[u] < self.dc || deg[v] < self.dc) {
                return e;
            }
        }
    }

    fn _choose_edge(&mut self) -> Edge<G> {
        if let Some(ref normal) = self.normal {
            let m = self.edges.len();
            let i = (normal.sample(&mut self.rng).abs() as usize) % m;
            self.edges[i]
        } else {
            *self.rng.choose(&self.edges).unwrap()
        }
    }
}

#[cfg_attr(feature = "cargo-clippy", allow(float_cmp))]
fn read(input: &str) -> (CompleteGraph, DefaultEdgePropMut<CompleteGraph, u64>) {
    use std::fs;
    let data = fs::read_to_string(input).unwrap();
    let data: Result<Vec<f64>, _> = data.split_whitespace().map(str::parse).collect();
    let data = data.unwrap();
    let n = (data.len() as f64).sqrt() as usize;
    if n * n != data.len() {
        panic!()
    }
    let g = CompleteGraph::new(n as u32);
    let mut w = g.default_edge_prop(0);
    for i in 0..(n - 1) {
        for j in (i + 1)..n {
            let e = g.edge_by_ends(i as u32, j as u32);
            if data[i * n + j] != data[j * n + i] {
                panic!()
            }
            w[e] = (SCALE * data[i * n + j]) as _;
        }
    }
    (g, w)
}

#[derive(Debug)]
struct Args {
    op: Op,
    dc: u32,
    input: String,
    seed: u32,
    pop_size: u32,
    max_num_iters: Option<u64>,
    max_num_iters_no_impr: Option<u64>,
    beta: Option<f64>,
    quiet: bool,
}

fn args() -> Args {
    let app = clap_app!(
        ("dc") =>
            (version: crate_version!())
            (about: crate_description!())
            (author: crate_authors!())
            (@arg seed:
                --seed
                +takes_value
                "Seed used in the random number generator. A random value is used if none is specified")
            (@arg beta:
                --beta
                +takes_value
                "Beta parameter. If no value is specifed then no heuristic selection is used")
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
            (@arg quiet:
                --quiet
                "Only prints the final solution")
            (@arg op:
                +required
                possible_values(&[
                    "change-pred",
                    "change-any"
                ])
                "Operator")
            (@arg dc:
                +required
                "Degree constraint")
            (@arg input:
                +required
                "Input file graph (weight matrix)")
    );

    let matches = app.get_matches();

    Args {
        seed: if matches.is_present("seed") {
            value_t_or_exit!(matches, "seed", u32)
        } else {
            rand::thread_rng().gen()
        },
        pop_size: value_t_or_exit!(matches, "pop_size", u32),
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
        beta: if matches.is_present("beta") {
            Some(value_t_or_exit!(matches, "beta", f64))
        } else {
            None
        },
        quiet: matches.is_present("quiet"),
        op: match matches.value_of("op").unwrap() {
            "change-pred" => Op::ChangePred,
            "change-any" => Op::ChangeAny,
            _ => unreachable!(),
        },
        dc: value_t_or_exit!(matches, "dc", u32),
        input: matches.value_of("input").unwrap().to_string(),
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
enum Op {
    ChangePred,
    ChangeAny,
}
