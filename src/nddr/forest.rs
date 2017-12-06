// external
use fera::ext::VecExt;
use fera::fun::vec;
use fera::graph::algs::Trees;
use fera::graph::choose::Choose;
use fera::graph::prelude::*;
use fera::graph::traverse::{continue_if, Control, Dfs, Visitor, OnDiscoverTreeEdge};
use rand::{Rng, XorShiftRng};

// system
use std::cell::{Ref, RefMut, RefCell};
use std::rc::Rc;
use std::ops::Deref;

// internal
use nddr::tree::*;

// See nddr module documentation for references.
//
// The following is missing to get the paper implementation
//     - Op2
//     - Reset pi_* (PI_x) and history (L), so they do not get too large
//     - do not replace the worst individual on the GA of bin/main.rs, replace the parent
//
// There is a lot of TODOs in this file, some of them are aesthetics, others are small
// optimizations. I don't know if its worth to implement that...
//
// FIXME: Cannot find optimal solution for OTMP...

#[derive(PartialEq, Clone, Copy)]
pub enum FindOpStrategy {
    Adj,
    AdjSmaller,
    Edge,
    Balanced,
}

#[derive(PartialEq, Clone, Copy)]
pub enum FindVertexStrategy {
    FatNode,
    Map,
}

// One instance of Data is shared between many Forest instances.
struct Data<G, W>
where
    G: Graph
        + WithVertexIndexProp
        + WithVertexProp<DefaultVertexPropMut<G, bool>>,
    W: EdgeProp<G, u64>,
{
    g: G,
    w: W,
    rng: RefCell<XorShiftRng>,
    nsqrt: usize,
    pub dc: usize,
    pub find_op_strategy: FindOpStrategy,
    pub find_vertex_strategy: FindVertexStrategy,

    // Used if find_vertex_strategy == Map to map each vertex to a number in 0..n
    vertex_index: VertexIndexProp<G>,

    // The next two field are used if find_op_strategy = Adj or AdjSmaller
    // used in sample without replacement in select_tree_if
    tree_indices: Vec<usize>,
    // used to mark edges in a tree
    m: DefaultVertexPropMut<G, DefaultVertexPropMut<G, bool>>,

    // The next fields are use if find_vertex_strategy = FatNode
    version: usize,
    // FIXME: reset
    pi_version: DefaultVertexPropMut<G, Vec<usize>>,
    pi_tree: DefaultVertexPropMut<G, Vec<usize>>,
    pi_pos: DefaultVertexPropMut<G, Vec<usize>>,
}

impl<G, W> Data<G, W>
where
    G: Graph
        + WithVertexIndexProp
        + WithVertexProp<DefaultVertexPropMut<G, bool>>,
    W: EdgeProp<G, u64>,
    DefaultVertexPropMut<G, bool>: Clone,
{
    fn new(g: G, w: W, rng: XorShiftRng) -> Self {
        let nsqrt = (g.num_vertices() as f64).sqrt().ceil() as usize;
        let m = g.vertex_prop(g.vertex_prop(false));
        let vertex_index = g.vertex_index();
        // 0 is the index of the star tree, so its keeped out
        let tree_indices = vec(1..nsqrt + 1);
        let pi_version = g.vertex_prop(Vec::<usize>::new());
        let pi_tree = g.vertex_prop(Vec::<usize>::new());
        let pi_pos = g.vertex_prop(Vec::<usize>::new());
        Data {
            g: g,
            w: w,
            rng: RefCell::new(rng),
            nsqrt: nsqrt,
            dc: ::std::usize::MAX,
            find_op_strategy: FindOpStrategy::Balanced,
            find_vertex_strategy: FindVertexStrategy::FatNode,
            vertex_index: vertex_index,
            tree_indices: tree_indices,
            m: m,
            version: 0,
            pi_version: pi_version,
            pi_tree: pi_tree,
            pi_pos: pi_pos,
        }
    }

    fn g(&self) -> &G {
        &self.g
    }

    fn w(&self) -> &W {
        &self.w
    }

    fn rng(&self) -> RefMut<XorShiftRng> {
        self.rng.borrow_mut()
    }
}

pub struct Forest<G, W>
where
    G: Graph
        + Choose
        + WithVertexIndexProp
        + WithVertexProp<DefaultVertexPropMut<G, bool>>,
    W: EdgeProp<G, u64>,
    Vertex<G>: Ord,
{
    data: Rc<RefCell<Data<G, W>>>,

    // The star tree is keeped in trees[0]. The star tree connects the roots of the trees.
    // TODO: Use RcArray
    trees: Vec<Rc<NddTree<Vertex<G>>>>,
    // Edges (u, v) such that u and v are star tree vertices.
    star_edges: Rc<RefCell<Vec<Edge<G>>>>,
    // Sum of weight of the tree edges, including star edges
    weight: u64,

    // Used if find_vertex_strategy == Map
    //
    // if trees[i][maps[i][vertex_index[v]]].vertex == v,
    // so v is in tree[i] in position maps[i][vertex_index[v]]
    // TODO: move this to NddTree
    maps: Vec<Rc<Vec<usize>>>,

    // Used if find_vertex_strategy = FatNode
    //
    // FIXME: implent a scheme to reset the history
    version: usize,
    history: Vec<usize>,
}

/*
   impl<G, W> Clone for Forest<G, W>
   where G: Graph + Choose,
   W: EdgeProp<G, u64>,
   Vertex<G>: Ord
   {
   clone!{Forest, con, trees, star_edges, weight, maps, version, history;}
   }
   */

impl<G, W> Deref for Forest<G, W>
where
    G: Graph
        + Choose
        + WithVertexIndexProp
        + WithVertexProp<DefaultVertexPropMut<G, bool>>,
    W: EdgeProp<G, u64>,
    Vertex<G>: Ord,
{
    type Target = Vec<Rc<NddTree<Vertex<G>>>>;

    fn deref(&self) -> &Self::Target {
        &self.trees
    }
}

impl<G, W> PartialEq for Forest<G, W>
where
    G: Graph
        + Choose
        + WithVertexIndexProp
        + WithVertexProp<DefaultVertexPropMut<G, bool>>,
    W: EdgeProp<G, u64>,
    Vertex<G>: Ord,
{
    fn eq(&self, _other: &Self) -> bool {
        panic!()
    }
}

impl<G, W> Forest<G, W>
where
    G: AdjacencyGraph
        + Choose
        + WithVertexIndexProp
        + WithVertexProp<DefaultVertexPropMut<G, bool>>,
    W: EdgeProp<G, u64>,
    DefaultVertexPropMut<G, bool>: Clone,
    Vertex<G>: Ord,
{
    pub fn new(g: G, w: W, rng: XorShiftRng, edges: Vec<Edge<G>>) -> Self {
        Self::new_with_data(Rc::new(RefCell::new(Data::new(g, w, rng))), edges)
    }

    fn new_with_data(data: Rc<RefCell<Data<G, W>>>, mut edges: Vec<Edge<G>>) -> Self {
        let data_rc = data.clone();
        let mut data = data.borrow_mut();
        let weight = edges.iter().map(|e| data.w()[*e]).sum();
        let nsqrt = data.nsqrt;

        let star_tree = {
            let mut rng = data.rng().clone();
            let s = data.g().spanning_subgraph(edges.clone());
            let r = s.choose_vertex(&mut rng).unwrap();
            Rc::new(NddTree::new(s.find_star_tree(r, nsqrt)))
        };

        let roots = vec(star_tree.iter().map(|ndd| ndd.vertex()));
        let star_edges = {
            let mut star_edges = vec![];
            let data = &data;
            for i in 0..roots.len() {
                for j in i + 1..roots.len() {
                    let u = roots[i];
                    let v = roots[j];
                    if let Some(e) = data.g().get_edge_by_ends(u, v) {
                        star_edges.push(e);
                        star_edges.push(data.g().reverse(e));
                    }
                }
            }
            star_edges
        };

        // TODO: improve
        edges.retain(|&e| {
            let (u, v) = data.g().ends(e);
            !star_tree.contains_edge(u, v)
        });

        let mut trees = vec![star_tree];

        // TODO: try to create a balanced Forest
        trees.extend({
            let s = data.g().spanning_subgraph(edges);
            s.collect_trees(&roots).into_iter().map(|t| {
                let mut t = NddTree::new(t);
                // TODO: find an away to not pass this closure
                t.calc_degs(|v| data.g().out_degree(v));
                Rc::new(t)
            })
        });

        assert_eq!(nsqrt + 1, trees.len());

        if data.find_vertex_strategy == FindVertexStrategy::FatNode {
            let data = &mut data;
            data.version += 1;
            let version = data.version;
            for (i, t) in trees[1..].iter().enumerate() {
                Forest::<G, W>::add_fat_node(data, version, i + 1, &t);
            }
        }

        let mut maps = vec![Rc::new(vec![])];
        if data.find_vertex_strategy == FindVertexStrategy::Map {
            let indices = &data.vertex_index;
            for (i, t) in trees[1..].iter().enumerate() {
                maps.push(Forest::<G, W>::new_map(indices, i + 1, &t));
            }
        }

        let version = data.version;

        Forest {
            data: data_rc,
            trees: trees,
            maps: maps,
            version: version,
            history: vec![version],
            star_edges: Rc::new(RefCell::new(star_edges)),
            weight: weight,
        }
    }

    pub fn op1(&mut self) {
        loop {
            let (e, from, p, to, a) = self.find_vertices_op1();
            let v = self.g().target(e);
            if !self.is_deg_max(v) {
                self.do_op1((from, p, to, a));
                assert!(self.is_deg_valid(v));
                break;
            }
        }
    }

    pub fn contains(&self, e: Edge<G>) -> bool {
        // TODO: what is the execution time?
        let (u, v) = self.g().ends(e);
        self.trees.iter().any(|t| t.contains_edge(u, v))
    }

    pub fn weight(&self) -> u64 {
        self.weight
    }

    pub fn check(&self) {
        let mut weight = 0;
        let edges = self.edges_vec();
        for &e in &edges {
            assert!(self.contains(e));
            weight += self.w()[e];
        }
        let g = self.g();
        let sub = g.spanning_subgraph(self.edges_vec());
        assert!(sub.is_tree());
        assert_eq!(weight, self.weight());
    }

    // Subtree ops

    #[inline(never)]
    fn do_op1(&mut self, params: (usize, usize, usize, usize)) -> OptionEdge<G> {
        let (ifrom, p, ito, a) = params;
        let ins = self.edge_by_ends(self[ifrom][p].vertex(), self[ito][a].vertex());
        let rem = self.edge_by_ends(self[ifrom][p].vertex(), self[ifrom].parent_vertex(p));

        if ifrom == ito {
            let new = one_tree_op1(&self[ifrom], p, a);
            self.replace1(ifrom, new);
        } else {
            let (from, to) = op1(&self[ifrom], p, &self[ito], a);
            self.replace(ifrom, from, ito, to);
        }

        let mut weight = self.weight;
        weight -= self.w()[rem];
        weight += self.w()[ins];
        self.weight = weight;

        G::edge_some(rem)
    }

    #[inline(never)]
    fn can_insert_subtree_edge(&self, params: (usize, usize, usize, usize)) -> bool {
        let (from, p, to, a) = params;
        // The root of the subtree being transfered cannot be the root
        // p cannot be ancestor of a in one_tree_op1
        !(p == 0 ||
              from == to &&
                  (p == a ||
                       self[from].contains_edge(self[from][p].vertex(), self[from][a].vertex()) ||
                       self[from].is_ancestor(p, a)))
    }

    #[inline(never)]
    fn find_vertices_op1(&self) -> (Edge<G>, usize, usize, usize, usize) {
        if self.should_mutate_star_tree() {
            return self.find_op_star_edge();
        }
        // TODO: add limit
        loop {
            let (from, p, to, a) = match self.find_op_strategy() {
                FindOpStrategy::Adj => self.find_op_adj(),
                FindOpStrategy::AdjSmaller => self.find_op_adj_smaller(),
                FindOpStrategy::Edge => self.find_op_edge(),
                FindOpStrategy::Balanced => self.find_op_balanced(),
            };

            // cannot be the star_tree
            assert!(from != 0);
            assert!(to != 0);

            if self.can_insert_subtree_edge((from, p, to, a)) {
                let e = self.edge_by_ends(self[from][p].vertex(), self[to][a].vertex());
                return (e, from, p, to, a);
            }
        }
    }

    #[inline(never)]
    fn find_op_adj(&self) -> (usize, usize, usize, usize) {
        // Sec V-B - page 834
        // In this implementation p and a are indices, not vertices
        // v_p and v_a are vertices.

        // Step 1
        let from = self.select_tree_if(|i| {
            // Must have more than one node so we can choose a node != root
            self[i].deg_in_g() > self[i].deg() && self[i].len() > 1
        }).expect("The graph is a forest");

        // Step 2
        let p = self.select_tree_vertex_if(from, |i| {
            // i is not root and have an edge tha is not in this forest
            i != 0 && self[from][i].deg() < self.g().out_degree(self[from][i].vertex())
        });

        // Step 3
        self.mark_edges(from);

        let v_a = {
            let m = &self.data().m;
            // Choose an edge (v_p, v_a) not in trees[from], that is, not marked
            let v_p = self[from][p].vertex();
            // TODO: in a complete graph the chance of choosing a vertex v_a such that (v_p, v_a)
            // is in self[from] is too small, so marking edges is not necessary and is too
            // expensive. Use self[from].contains_edge instead.
            //
            // self.g().choose_neighbor_if(&mut *self.rng(),
            //                             v_p,
            //                             &mut |v_a| !self[from].contains_edge(v_p, v_a))
            //
            self.g()
                .choose_out_neighbor_iter(v_p, &mut *self.rng())
                .filter(|&v_a| !m[v_p][v_a] && !m[v_a][v_p])
                .next()
                .unwrap()
        };

        self.unmark_edges(from);

        // Step 4
        let (to, a) = self.find_index(v_a);

        // Step 5 is executed by the callee, so just return
        (from, p, to, a)
    }

    #[inline(never)]
    fn find_op_adj_smaller(&self) -> (usize, usize, usize, usize) {
        let (from, p, to, a) = self.find_op_adj();
        if self[from].len() >= self[to].len() {
            (from, p, to, a)
        } else {
            (to, a, from, p)
        }
    }

    #[inline(never)]
    fn find_op_edge(&self) -> (usize, usize, usize, usize) {
        let g = self.g();
        // TODO: use tournament without replacement
        let mut e = g.choose_edge(&mut *self.rng()).unwrap();
        // The order of the vertices is important, so trying (u, v) is different from (v, u),
        // as only (u, v) or (v, u) can be returned from choose_edge, we trye th reverse with 50%
        // of change.
        // This is not necessary in the other methods because (u, v) and (v, u) can be choosed.
        if self.rng().gen() {
            e = self.g().reverse(e);
        }
        let (u, v) = g.ends(e);
        let (from, p) = self.find_index(u);
        let (to, a) = self.find_index(v);
        (from, p, to, a)
    }

    #[inline(never)]
    fn find_op_balanced(&self) -> (usize, usize, usize, usize) {
        let from = self.select_tree();
        let p = self.select_tree_vertex(from);
        let to = self.select_tree();
        let a = self.select_tree_vertex(to);
        (from, p, to, a)
    }


    // Star edges

    #[inline(never)]
    fn can_insert_star_edge(&self, ins: Edge<G>) -> bool {
        let (u, v) = self.g().ends(ins);
        // FIXME: contains_edge is linear!, this can compromisse O(sqrt(n)) time
        // for non complete graph
        // FIXME: whe this function is called from find_op_star_edge find_vertex in called twice
        self[0][0].vertex() != u && !self[0].contains_edge(u, v) &&
            {
                let p = self[0].find_vertex(u).unwrap();
                let a = self[0].find_vertex(v).unwrap();
                !self[0].is_ancestor(p, a)
            }
    }

    #[inline(never)]
    fn find_op_star_edge(&self) -> (Edge<G>, usize, usize, usize, usize) {
        let e = {
            *sample_without_replacement(&mut *self.star_edges.borrow_mut(), &mut *self.rng(), |e| {
                self.can_insert_star_edge(*e)
            }).unwrap()
        };
        let (u, v) = self.g().ends(e);
        let p = self[0].find_vertex(u).unwrap();
        let a = self[0].find_vertex(v).unwrap();
        (e, 0, p, 0, a)
    }

    fn replace1(&mut self, i: usize, mut ti: NddTree<Vertex<G>>) {
        // TODO: unify replace1 and replace
        if self.find_vertex_strategy() == FindVertexStrategy::FatNode {
            self.new_version();
            let data = &mut *self.data_mut();
            Self::add_fat_node(data, self.version, i, &ti);
        }

        if self.find_vertex_strategy() == FindVertexStrategy::Map {
            let indices = &(self.data.borrow().vertex_index);
            self.maps[i] = Self::new_map(indices, i, &ti);
        }

        // TODO: call only when its needed
        ti.calc_degs(|v| self.g().out_degree(v));

        self.trees[i] = Rc::new(ti);
    }


    // Misc

    fn replace(
        &mut self,
        i: usize,
        mut ti: NddTree<Vertex<G>>,
        j: usize,
        mut tj: NddTree<Vertex<G>>,
    ) {

        if self.find_vertex_strategy() == FindVertexStrategy::FatNode {
            self.new_version();
            let data = &mut *self.data_mut();
            Self::add_fat_node(data, self.version, i, &ti);
            Self::add_fat_node(data, self.version, j, &tj);
        }

        if self.find_vertex_strategy() == FindVertexStrategy::Map {
            let indices = &(self.data.borrow().vertex_index);
            self.maps[i] = Self::new_map(indices, i, &ti);
            self.maps[j] = Self::new_map(indices, j, &tj);
        }

        // TODO: call only when its needed
        ti.calc_degs(|v| self.g().out_degree(v));
        tj.calc_degs(|v| self.g().out_degree(v));

        self.trees[i] = Rc::new(ti);
        self.trees[j] = Rc::new(tj);
    }

    #[inline(never)]
    fn new_map(
        indices: &VertexIndexProp<G>,
        i: usize,
        ti: &NddTree<Vertex<G>>,
    ) -> Rc<Vec<usize>> {
        if i == 0 {
            // TODO: do not create
            return Rc::new(vec![]);
        }
        let inds = vec(ti.iter().map(|ndd| indices.get(ndd.vertex())));
        let max = *inds.iter().max().unwrap();
        let mut map = unsafe { Vec::new_uninitialized(max + 1) };
        for (pos, iv) in inds.iter().enumerate() {
            map[*iv] = pos;
        }
        Rc::new(map)
    }

    fn add_fat_node(data: &mut Data<G, W>, version: usize, i: usize, ti: &NddTree<Vertex<G>>) {
        if i == 0 {
            return;
        }
        for (pos, n) in ti.iter().enumerate() {
            let v = n.vertex();
            data.pi_version[v].push(version);
            data.pi_tree[v].push(i);
            data.pi_pos[v].push(pos);
        }
    }

    fn should_mutate_star_tree(&self) -> bool {
        // TODO: explain
        self.trees[0].len() > 2 && 2 * self.star_edges.borrow().len() >= self.trees[0].len() &&
            self.rng().gen_range(0, self.data().nsqrt + 1) == self.data().nsqrt
    }

    #[inline(never)]
    fn select_tree_if<F>(&self, mut f: F) -> Option<usize>
    where
        F: FnMut(usize) -> bool,
    {
        let mut rng: XorShiftRng = self.rng().clone();
        sample_without_replacement(
            &mut *self.data_mut().tree_indices,
            &mut rng,
            |&i| f(i),
        ).cloned()
    }

    #[inline(never)]
    fn select_tree(&self) -> usize {
        self.rng().gen_range(1, self.len())
    }

    #[inline(never)]
    fn select_tree_vertex(&self, i: usize) -> usize {
        self.rng().gen_range(0, self[i].len())
    }

    #[inline(never)]
    fn select_tree_vertex_if<F>(&self, i: usize, mut f: F) -> usize
    where
        F: FnMut(usize) -> bool,
    {
        // TODO: add a limit or use sample without replacement
        loop {
            let i = self.rng().gen_range(0, self[i].len());
            if f(i) {
                return i;
            }
        }
    }

    #[inline(never)]
    fn find_index(&self, v: Vertex<G>) -> (usize, usize) {
        match self.find_vertex_strategy() {
            FindVertexStrategy::FatNode => {
                for ver in self.history.iter().rev() {
                    if let Some(i) = self.data().pi_version[v].binary_search(ver).ok() {
                        let t = self.data().pi_tree[v][i];
                        let p = self.data().pi_pos[v][i];
                        assert_eq!(v, self[t][p].vertex());
                        return (t, p);
                    }
                }
                panic!()
            }
            FindVertexStrategy::Map => {
                let iv = self.data().vertex_index.get(v);
                for (i, map) in self.maps[1..].iter().enumerate() {
                    if let Some(&p) = map.get(iv) {
                        if let Some(tree) = self.get(i + 1) {
                            if let Some(ndd) = tree.get(p) {
                                if ndd.vertex() == v {
                                    return (i + 1, p);
                                }
                            }
                        }
                    }
                }
                panic!("find_index with Map do not find vertex");
            }
        }
    }

    fn mark_edges(&self, t: usize) {
        self.set_edges_on_m(t, true);
    }

    fn unmark_edges(&self, t: usize) {
        self.set_edges_on_m(t, false);
    }

    #[inline(never)]
    fn set_edges_on_m(&self, t: usize, value: bool) {
        let m = &mut self.data_mut().m;
        self[t].for_each_edge(|u, v| m[u][v] = value);
    }

    fn new_version(&mut self) {
        self.data_mut().version += 1;
        let version = self.data().version;
        self.version = version;
        self.history.push(self.version);
    }

    fn edges_vec(&self) -> Vec<Edge<G>> {
        let v = vec(self.iter().map(|t| t.edges()));
        vec(v.iter().flat_map(|t| t.iter()).map(|&(u, v)| {
            self.edge_by_ends(u, v)
        }))
    }

    fn edge_by_ends(&self, u: Vertex<G>, v: Vertex<G>) -> Edge<G> {
        self.g().edge_by_ends(u, v)
    }

    fn find_op_strategy(&self) -> FindOpStrategy {
        self.data().find_op_strategy
    }

    fn find_vertex_strategy(&self) -> FindVertexStrategy {
        self.data().find_vertex_strategy
    }

    fn degree(&self, u: Vertex<G>) -> usize {
        let (t, i) = self.find_index(u);
        self[t][i].deg() +
            if i == 0 {
                // u is a root so get the degree on start_edge
                self[0][self[0].find_vertex(u).unwrap()].deg()
            } else {
                0
            }
    }

    fn is_deg_max(&self, u: Vertex<G>) -> bool {
        self.degree(u) == self.dc()
    }

    fn is_deg_valid(&self, u: Vertex<G>) -> bool {
        self.degree(u) <= self.dc()
    }

    fn dc(&self) -> usize {
        self.data().dc
    }

    fn w(&self) -> Ref<W> {
        Ref::map(self.data(), |d| d.w())
    }

    fn g(&self) -> Ref<G> {
        Ref::map(self.data(), |d| d.g())
    }

    fn rng<'a>(&'a self) -> RefMut<'a, XorShiftRng> {
        let d = self.data();
        let x: RefMut<'a, XorShiftRng> = unsafe { ::std::mem::transmute(d.rng()) };
        x
    }

    fn data(&self) -> Ref<Data<G, W>> {
        self.data.borrow()
    }

    fn data_mut(&self) -> RefMut<Data<G, W>> {
        self.data.borrow_mut()
    }
}


// TODO: move to other file

trait GraphForestExt: IncidenceGraph {
    fn collect_trees(&self, roots: &Vec<Vertex<Self>>) -> Vec<Vec<Ndd<Vertex<Self>>>> {
        let mut vis = CompsVisitor {
            trees: vec![],
            depth: self.vertex_prop(0usize),
        };
        self.dfs(&mut vis).roots(roots.iter().cloned()).run();
        vis.trees
    }

    fn find_star_tree(&self, v: Vertex<Self>, n: usize) -> Vec<Ndd<Vertex<Self>>> {
        let mut edges = vec![];
        // TODO: use a randwalk, so star_edges do not have a tendency
        self.dfs(&mut OnDiscoverTreeEdge(|e| {
            edges.push(e);
            continue_if(edges.len() + 1 < n)
        })).root(v)
            .run();

        let s = self.edge_induced_subgraph(edges);
        let roots = vec![v];
        s.collect_trees(&roots).into_iter().next().unwrap()
    }
}

impl<G: IncidenceGraph> GraphForestExt for G {}

struct CompsVisitor<G: AdjacencyGraph> {
    trees: Vec<Vec<Ndd<Vertex<G>>>>,
    depth: DefaultVertexPropMut<G, usize>,
}

impl<G: AdjacencyGraph> Visitor<G> for CompsVisitor<G> {
    fn discover_root_vertex(&mut self, g: &G, v: Vertex<G>) -> Control {
        self.trees.push(vec![Ndd::new(v, 0, g.out_degree(v))]);
        Control::Continue
    }

    fn discover_edge(&mut self, g: &G, e: Edge<G>) -> Control {
        let (u, v) = g.ends(e);
        self.depth[v] = self.depth[u] + 1;
        self.trees.last_mut().unwrap().push(Ndd::new(
            v,
            self.depth[v],
            g.out_degree(v),
        ));
        Control::Continue
    }
}

fn sample_without_replacement<T, R, F>(data: &mut [T], mut rng: R, mut accept: F) -> Option<&T>
where
    R: Rng,
    F: FnMut(&T) -> bool,
{
    let n = data.len();
    for i in 0..n {
        let j = rng.gen_range(i, n);
        if accept(&data[j]) {
            return Some(&data[j]);
        }
        data.swap(i, j);
    }
    None
}


// Tests

#[cfg(test)]
mod tests {
    use super::*;
    use rand;

    #[test]
    fn test_collect_trees() {
        //      . 0
        //    .   .  \
        //   1    2   3
        //  / \   |   |
        // 4   5  6   7
        //            .
        //            8

        let mut b = StaticGraph::builder(9, 5);
        b.add_edge(1, 4);
        b.add_edge(1, 5);
        b.add_edge(2, 6);
        b.add_edge(0, 3);
        b.add_edge(3, 7);
        let g = b.finalize();

        let trees = g.collect_trees(&vec![0, 1, 2, 8]);

        assert_eq!(
            NddTree::new(trees[0].clone()),
            NddTree::from_vecs(&[0, 3, 7], &[0, 1, 2], &[1, 2, 1])
        );

        assert_eq!(
            NddTree::new(trees[1].clone()),
            NddTree::from_vecs(&[1, 4, 5], &[0, 1, 1], &[2, 1, 1])
        );

        assert_eq!(
            NddTree::new(trees[2].clone()),
            NddTree::from_vecs(&[2, 6], &[0, 1], &[1, 1])
        );
    }

    fn data()
        -> (Data<CompleteGraph, DefaultEdgePropMut<CompleteGraph, u64>>, Vec<Edge<CompleteGraph>>)
    {
        let mut rng = rand::weak_rng();
        let g = CompleteGraph::new(100);
        let w: DefaultEdgePropMut<CompleteGraph, u64> =
            g.edge_prop_from_fn(|_| rng.gen_range(0, 1_000_000));
        let tree = vec(
            StaticGraph::new_random_tree(100, &mut rng)
                .edges_ends()
                .map(|(u, v)| g.edge_by_ends(u, v)),
        );

        (Data::new(g.clone(), w.clone(), rng), tree)
    }

    macro_rules! def_test {
        ($name:ident, $op:ident, $vertex:ident) => (
            #[test]
            fn $name() {
                let (mut data, tree) = data();
                data.find_op_strategy = FindOpStrategy::$op;
                data.find_vertex_strategy = FindVertexStrategy::$vertex;
                let mut forest = Forest::new_with_data(Rc::new(RefCell::new(data)), tree);
                for _ in 0..1000 {
                    forest.op1();
                    forest.check();
                }
            }
        )
    }

    def_test!(test_op1_adj_fatnode, Adj, FatNode);
    def_test!(test_op1_adj_map, Adj, Map);

    def_test!(test_op1_adj_smaller_fatnode, AdjSmaller, FatNode);
    def_test!(test_op1_adj_smaller_map, AdjSmaller, Map);

    def_test!(test_op1_balanced_fatnode, Edge, FatNode);
    def_test!(test_op1_balanced_map, Edge, Map);

    def_test!(test_op1_edge_fatnode, Edge, FatNode);
    def_test!(test_op1_edge_map, Edge, Map);
}
