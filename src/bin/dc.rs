#[macro_use]
extern crate clap;
extern crate ea_tree_repr;
extern crate fera;
extern crate rand;

// external
use clap::ErrorKind;
use fera::fun::{position_max_by_key, position_min_by_key, vec};
use fera::graph::algs::Kruskal;
use fera::graph::choose::Choose;
use fera::graph::prelude::*;
use rand::distributions::{IndependentSample, Normal};
use rand::{Rng, SeedableRng, XorShiftRng};

// std
use std::rc::Rc;

// local
use ea_tree_repr::{PredecessorTree, Tree};

const SCALE: f64 = 10000000000.0;

pub fn main() {
    let args = args();
    if !args.quiet {
        println!("{:#?}", args);
    }
    let (g, w) = read(&args.input);
    let (it, weight, edges) = run(Rc::new(g), w, &args);
    println!("it = {}, best = {}", it, weight as f64 / SCALE);
    print!("{}", weight as f64 / SCALE);
    for (u, v) in g.ends(edges) {
        print!(" {:?}-{:?}", u, v);
    }
    println!()
}

fn run(
    g: Rc<CompleteGraph>,
    w: DefaultEdgePropMut<CompleteGraph, u64>,
    args: &Args,
) -> (u64, u64, Vec<Edge<CompleteGraph>>) {
    let mut rng = XorShiftRng::from_seed([args.seed, args.seed, args.seed, args.seed]);
    let mut edges = vec(g.edges());

    // Initialize the population
    let mut pop = vec![];
    for _ in 0..args.pop_size {
        rng.shuffle(&mut edges);
        if pop.is_empty() {
            pop.push(Ind::new(g.clone(), &w, args.dc, &edges));
        } else {
            let mut new = pop[0].clone();
            new.set_edges(&w, args.dc, &edges);
            pop.push(new);
        }
    }

    // Saves the best
    let mut best = position_min_by_key(&pop, |ind| ind.fitness()).unwrap();

    if !args.quiet {
        println!("best = {:?}", pop[best].weight);
    }

    let mut op = InsertRemove::new(
        g.num_vertices(),
        args.dc,
        vec(g.edges()).sorted_by_prop(&w),
        args.beta,
        rng.clone(),
    );

    let mut it_best = 0;
    for it in 0..=args.max_num_iters.unwrap_or(u64::max_value()) {
        let i = rng.gen_range(0, pop.len());
        let mut new = pop[i].clone();

        if args.op == Op::ChangePred {
            op.change_pred(&w, &mut new);
        } else {
            op.change_any(&w, &mut new);
        }
        if pop.contains(&new) {
            continue;
        }

        if new.fitness() <= pop[i].fitness() {
            if new.fitness() < pop[best].fitness() {
                it_best = it;
                best = i;
                if !args.quiet {
                    println!("it = {}, best = {:?}", it, new.weight);
                }
            }
            pop[i] = new;
        } else {
            let j = position_max_by_key(&pop, |ind| ind.fitness()).unwrap();
            pop[j] = new;
            best = position_min_by_key(&pop, |ind| ind.fitness()).unwrap();
        }

        if it - it_best >= args.max_num_iters_no_impr.unwrap_or(u64::max_value()) {
            if !args.quiet {
                println!(
                    "max number of iterations without improvement reached: {:?}",
                    args.max_num_iters_no_impr
                );
            }
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
        let deg = g.default_vertex_prop(0u32);
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
        let normal = beta.map(|beta| Normal::new(0.0, f64::from(beta) * (n as f64)));
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
                    rem = ind.tree.set_pred(v, ins);
                    break;
                }
            } else if ind.tree.is_ancestor_of(v, u) {
                if ind.deg[v] < self.dc {
                    rem = ind.tree.set_pred(u, ins);
                    break;
                }
            } else {
                match (ind.deg[u] < self.dc, ind.deg[v] < self.dc) {
                    (false, false) => panic!(),
                    (false, true) => {
                        rem = ind.tree.set_pred(u, ins);
                    }
                    (true, false) => {
                        ins = g.reverse(ins);
                        rem = ind.tree.set_pred(v, ins);
                    }
                    (true, true) => {
                        if self.rng.gen() {
                            rem = ind.tree.set_pred(u, ins);
                        } else {
                            ins = g.reverse(ins);
                            rem = ind.tree.set_pred(v, ins);
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
            let buffer = ind.tree.buffer();
            let mut buffer = buffer.borrow_mut();
            let deg = &ind.deg;
            ind.tree.insert_remove(&mut *buffer, ins, |path| {
                let dc = self.dc;
                match (deg[u] < dc, deg[v] < dc) {
                    (false, false) => panic!(),
                    (false, true) => 0,
                    (true, false) => path.len() - 1,
                    (true, true) => self.rng.gen_range(0, path.len()),
                }
            })
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
            let i = (normal.ind_sample(&mut self.rng).abs() as usize) % m;
            self.edges[i]
        } else {
            *self.rng.choose(&self.edges).unwrap()
        }
    }
}

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
    return (g, w);
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
                "The seed used in the random number generator. A random value is used if none is specified")
            (@arg beta:
                --beta
                +takes_value
                "With no value is specifed then no heuristic selections is used")
            (@arg pop_size:
                --pop_size
                +takes_value
                default_value("10")
                "The number of individual in the population")
            (@arg max_num_iters:
                --max_num_iters
                +takes_value
                "The maximum number of iterations")
            (@arg max_num_iters_no_impr:
                --max_num_iters_no_impr
                +takes_value
                "The maximum number of iterations without improvements")
            (@arg quiet: --quiet "Only prints the final solution")
            (@arg op:
                +required
                possible_values(&[
                    "change-pred",
                    "change-any"
                ])
                "operation"
            )
            (@arg dc:
                +required
                "degree constraint"
            )
            (@arg input:
                +required
                "input file graph (weight matrix)"
            )
    );

    let matches = app.get_matches();

    Args {
        seed: value_t!(matches.value_of("seed"), u32).unwrap_or_else(|e| {
            if e.kind == ErrorKind::ArgumentNotFound {
                rand::thread_rng().gen()
            } else {
                e.exit()
            }
        }),
        pop_size: value_t_or_exit!(matches.value_of("pop_size"), u32),
        max_num_iters: value_t!(matches.value_of("max_num_iters"), u64)
            .map(|v| Some(v))
            .unwrap_or_else(|e| {
                if e.kind == ErrorKind::ArgumentNotFound {
                    None
                } else {
                    e.exit()
                }
            }),
        max_num_iters_no_impr: value_t!(matches.value_of("max_num_iters_no_impr"), u64)
            .map(|v| Some(v))
            .unwrap_or_else(|e| {
                if e.kind == ErrorKind::ArgumentNotFound {
                    None
                } else {
                    e.exit()
                }
            }),
        beta: value_t!(matches.value_of("beta"), f64)
            .map(|v| Some(v))
            .unwrap_or_else(|e| {
                if e.kind == ErrorKind::ArgumentNotFound {
                    None
                } else {
                    e.exit()
                }
            }),
        quiet: matches.is_present("quiet"),
        op: match matches.value_of("op").unwrap() {
            "change-pred" => Op::ChangePred,
            "change-any" => Op::ChangeAny,
            _ => unreachable!(),
        },
        dc: value_t_or_exit!(matches.value_of("dc"), u32),
        input: matches.value_of("input").unwrap().to_string(),
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
enum Op {
    ChangePred,
    ChangeAny,
}
