// external
use fera::ext::VecExt;
use fera::fun::vec;
use fera::graph::choose::Choose;
use fera::graph::prelude::*;
use rand::Rng;

// system
use std::cell::{Cell, Ref, RefCell, RefMut};
use std::ops::Deref;
use std::rc::Rc;

// internal
use {collect_ndds, find_star_tree, one_tree_op1, one_tree_op2, op1, op2, Bitset, NddTree};

// pg 836 5) Step 5 - the value of the k constant is not specified, we use the default value of 1
const DEFAULT_K: usize = 1;

#[derive(PartialEq, Clone, Copy)]
pub enum FindOpStrategy {
    Adj,
    AdjSmaller,
    Edge,
    Balanced,
}

// The strategy used to find the tree that contains a vertex.
#[derive(PartialEq, Clone, Copy)]
pub enum FindVertexStrategy {
    // This is the strategy described in the article. This is called FatNode because it is like the
    // fat node strategy used in persistent data structures.
    FatNode,
    // This strategy avoids the use of PI matrix used in the FatNode strategy. It's uses a map in
    // each tree to indicate if a vertex is present in that tree. To find the tree that contains a
    // vertex, the trees are queried in sequence. There are O(sqrt(n)) trees, and each queries takes
    // O(1), so the running time is O(sqrt(n)).
    Map,
}

#[derive(Clone)]
struct PiValue {
    version: u16,
    tree: u16,
    pos: u32,
}

impl PiValue {
    #[inline]
    fn new(version: u16, tree: u16, pos: u32) -> Self {
        PiValue { version, tree, pos }
    }
}

// One instance of Data is shared between many NddrOneTreeForest instances.
struct Data<G>
where
    G: Graph + WithVertexIndexProp + WithVertexProp<DefaultVertexPropMut<G, bool>>,
{
    nsqrt: usize,
    find_op_strategy: FindOpStrategy,
    find_vertex_strategy: FindVertexStrategy,

    // Used if find_vertex_strategy == Map or find_op_strategy = Adj* to map each vertex to a
    // number in 0..n
    vertex_index: VertexIndexProp<G>,

    // See section VI-B - 3
    // The next two field are used if find_op_strategy = Adj or AdjSmaller used in sample without
    // replacement in select_tree_if. This does the role of the L_O array described in the paper.
    tree_indices: Vec<usize>,
    // Used to mark edges in a tree (M_E_T in the paper)
    m: Vec<Bitset>,

    // The next fields are use if find_vertex_strategy = FatNode. In forest version h, the vertex
    // x is in position pi[x][h].pos of the tree with index pi[x][h].tree
    version: usize,
    // pg 836, reinitialization constant
    k: usize,
    // number of reinits
    num_reinits: usize,
    pi: Vec<Vec<PiValue>>,
}

impl<G> Data<G>
where
    G: Graph + WithVertexIndexProp + WithVertexProp<DefaultVertexPropMut<G, bool>>,
    DefaultVertexPropMut<G, bool>: Clone,
{
    pub fn new(g: &Rc<G>, find_op: FindOpStrategy, find_vertex: FindVertexStrategy) -> Self {
        let nsqrt = (g.num_vertices() as f64).sqrt().ceil() as usize;
        // Only allocate m and tree_indices if they will be used
        let m = match find_op {
            FindOpStrategy::Adj | FindOpStrategy::AdjSmaller => {
                let n = g.num_vertices();
                vec![Bitset::with_capacity(n); n]
            }
            _ => vec![],
        };
        let tree_indices = match find_op {
            // 0 is the index of the star tree, so it's kept out
            FindOpStrategy::Adj | FindOpStrategy::AdjSmaller => vec(1..nsqrt + 1),
            _ => vec![],
        };
        let pi = if find_vertex == FindVertexStrategy::FatNode {
            vec![vec![]; g.num_vertices()]
        } else {
            vec![]
        };
        let vertex_index = g.vertex_index();
        Data {
            nsqrt,
            find_op_strategy: find_op,
            find_vertex_strategy: find_vertex,
            vertex_index,
            tree_indices,
            m,
            version: 0,
            k: DEFAULT_K,
            num_reinits: 0,
            pi: pi,
        }
    }

    fn is_marked(&self, u: Vertex<G>, v: Vertex<G>) -> bool {
        let u = self.vertex_index.get(u);
        let v = self.vertex_index.get(v);
        self.m[u][v] || self.m[v][u]
    }

    fn set_m(&mut self, u: Vertex<G>, v: Vertex<G>, value: bool) {
        let u = self.vertex_index.get(u);
        let v = self.vertex_index.get(v);
        self.m[u].set(v, value);
    }
}

pub struct NddrOneTreeForest<G>
where
    G: Graph + WithVertexIndexProp + WithVertexProp<DefaultVertexPropMut<G, bool>>,
{
    g: Rc<G>,

    data: Rc<RefCell<Data<G>>>,

    last_op_size: usize,

    // The star tree is kept in trees[0]. The star tree connects the roots of the trees.
    trees: Vec<Rc<NddTree<Vertex<G>>>>,

    // Edges (u, v) such that u and v are star tree vertices (that is, tree roots)
    // The set of star edges does not change after initialized, but it can be shufflet in the
    // mutation process
    star_edges: Rc<RefCell<Vec<Edge<G>>>>,

    // Used if find_vertex_strategy == Map. Does not allocate if find_vertex_strategy != Map.
    //
    // if trees[i][maps[i][vertex_index[v]]].vertex == v,
    // then v is in tree[i] in position maps[i][vertex_index[v]]
    // len(trees) == len(maps) == O(sqrt(n))
    // TODO: move this to NddTree and use bitset like in euler tour
    maps: Vec<Rc<Vec<usize>>>,

    // Used if find_vertex_strategy = FatNode
    // In page 836, B-5) L = history
    version: Cell<usize>,
    reinit: Cell<usize>,
    history: RefCell<Vec<u16>>,
}

impl<G> Clone for NddrOneTreeForest<G>
where
    G: AdjacencyGraph
        + Choose
        + WithVertexIndexProp
        + WithVertexProp<DefaultVertexPropMut<G, bool>>,
    DefaultVertexPropMut<G, bool>: Clone,
{
    fn clone(&self) -> Self {
        self.reinit_if_needed();
        Self {
            g: Rc::clone(&self.g),
            data: Rc::clone(&self.data),
            last_op_size: self.last_op_size,
            trees: self.trees.clone(),
            star_edges: Rc::clone(&self.star_edges),
            maps: self.maps.clone(),
            version: self.version.clone(),
            reinit: self.reinit.clone(),
            // the max len of history is (k * nsqrt), so it does not affect the running time
            history: self.history.clone(),
        }
    }
}

impl<G> Deref for NddrOneTreeForest<G>
where
    G: Graph + Choose + WithVertexIndexProp + WithVertexProp<DefaultVertexPropMut<G, bool>>,
{
    type Target = Vec<Rc<NddTree<Vertex<G>>>>;

    fn deref(&self) -> &Self::Target {
        &self.trees
    }
}

impl<G> PartialEq for NddrOneTreeForest<G>
where
    G: AdjacencyGraph
        + Choose
        + WithVertexIndexProp
        + WithVertexProp<DefaultVertexPropMut<G, bool>>,
    DefaultVertexPropMut<G, bool>: Clone,
{
    fn eq(&self, other: &Self) -> bool {
        if (self as *const _) == (other as *const _) {
            return true;
        }

        for e in self.edges() {
            if !other.contains(e) {
                return false;
            }
        }

        true
    }
}

impl<G> NddrOneTreeForest<G>
where
    G: AdjacencyGraph
        + Choose
        + WithVertexIndexProp
        + WithVertexProp<DefaultVertexPropMut<G, bool>>,
    DefaultVertexPropMut<G, bool>: Clone,
{
    pub fn new<R: Rng>(g: Rc<G>, edges: Vec<Edge<G>>, rng: R) -> Self {
        Self::new_with_strategies(
            g,
            edges,
            // init with the default strategies described in the paper
            FindOpStrategy::Adj,
            FindVertexStrategy::FatNode,
            rng,
        )
    }

    pub fn new_with_strategies<R: Rng>(
        g: Rc<G>,
        edges: Vec<Edge<G>>,
        find_op: FindOpStrategy,
        find_vertex: FindVertexStrategy,
        rng: R,
    ) -> Self {
        let data = Rc::new(RefCell::new(Data::new(&g, find_op, find_vertex)));
        Self::new_with_data(g, data, edges, rng)
    }

    fn new_with_data<R: Rng>(
        g: Rc<G>,
        data: Rc<RefCell<Data<G>>>,
        mut edges: Vec<Edge<G>>,
        rng: R,
    ) -> Self {
        let data_ = data;
        // use a clone to make the borrow checker happy
        let data = Rc::clone(&data_);
        let mut data = data.borrow_mut();
        let nsqrt = data.nsqrt;

        let star_tree = Rc::new(NddTree::new(find_star_tree(
            &g.spanning_subgraph(&edges),
            nsqrt,
            rng,
        )));

        let roots = vec(star_tree.iter().map(|ndd| ndd.vertex()));
        let star_edges = {
            let mut star_edges = vec![];
            for i in 0..roots.len() {
                for j in i + 1..roots.len() {
                    let u = roots[i];
                    let v = roots[j];
                    if let Some(e) = g.get_edge_by_ends(u, v) {
                        star_edges.push(e);
                        star_edges.push(g.reverse(e));
                    }
                }
            }
            star_edges
        };

        edges.retain(|&e| {
            let (u, v) = g.ends(e);
            !star_tree.contains_edge(u, v)
        });

        let mut trees = vec![star_tree];

        trees.extend({
            let s = g.spanning_subgraph(edges);
            collect_ndds(&s, &roots).into_iter().map(|t| {
                let mut t = NddTree::new(t);
                // TODO: find an away to not pass this closure
                t.calc_degs(|v| g.out_degree(v) as u32);
                Rc::new(t)
            })
        });

        assert_eq!(nsqrt + 1, trees.len());

        if data.find_vertex_strategy == FindVertexStrategy::FatNode {
            data.version += 1;
            let version = data.version;
            for (i, t) in trees[1..].iter().enumerate() {
                NddrOneTreeForest::<G>::add_fat_node(&mut data, version, i + 1, t);
            }
        }

        let mut maps = vec![Rc::new(vec![])];
        if data.find_vertex_strategy == FindVertexStrategy::Map {
            let indices = &data.vertex_index;
            for (i, t) in trees[1..].iter().enumerate() {
                maps.push(NddrOneTreeForest::<G>::new_map(indices, i + 1, t));
            }
        }

        let version = data.version;
        let reinit = data.num_reinits;

        NddrOneTreeForest {
            g,
            data: data_,
            last_op_size: 0,
            trees,
            maps,
            version: Cell::new(version),
            reinit: Cell::new(reinit),
            history: RefCell::new(vec![version as _]),
            star_edges: Rc::new(RefCell::new(star_edges)),
        }
    }

    pub fn set_edges<R: Rng>(&mut self, edges: &[Edge<G>], rng: R) {
        *self = Self::new_with_data(Rc::clone(&self.g), Rc::clone(&self.data), edges.into(), rng);
    }

    pub fn graph(&self) -> &Rc<G> {
        &self.g
    }

    pub fn last_op_size(&self) -> usize {
        self.last_op_size
    }

    // TODO: create a version of op1 and op2 that takes degree constraint parameter
    pub fn op1<R: Rng>(&mut self, rng: R) -> (Edge<G>, Edge<G>) {
        self.reinit_if_needed();
        let (from, p, to, a) = self.find_vertices_op1(rng);
        self.last_op_size = self[from].len() + self[to].len();
        self.do_op1((from, p, to, a))
    }

    pub fn op2<R: Rng>(&mut self, rng: R) -> (Edge<G>, Edge<G>) {
        self.reinit_if_needed();
        let (from, p, r, to, a) = self.find_vertices_op2(rng);
        self.last_op_size = self[from].len() + self[to].len();
        self.do_op2((from, p, r, to, a))
    }

    fn contains(&self, e: Edge<G>) -> bool {
        self.reinit_if_needed();
        let (u, v) = self.graph().ends(e);
        self.trees[0].contains_edge(u, v) || {
            let (i, _) = self.find_index(u);
            self.trees[i].contains_edge(u, v)
        }
    }

    #[cfg(test)]
    fn check(&self) {
        use fera::graph::algs::Trees;
        let edges = self.edges();
        for &e in &edges {
            assert!(self.contains(e));
        }

        let g = self.graph();
        let sub = g.spanning_subgraph(&edges);
        assert!(sub.is_tree());
    }

    fn reinit_if_needed(&self) {
        // reset the history if needed
        let mut data = self.data.borrow_mut();
        if data.find_vertex_strategy != FindVertexStrategy::FatNode {
            return;
        }
        // If the history is too long, reinit
        if data.version >= data.k * data.nsqrt {
            data.version = 1;
            data.num_reinits += 1;
            for v in self.graph().vertices() {
                let v = data.vertex_index.get(v);
                data.pi[v].clear();
            }
            let version = data.version;
            let reinit = data.num_reinits;
            for (i, t) in self.trees[1..].iter().enumerate() {
                NddrOneTreeForest::<G>::add_fat_node(&mut data, version, i + 1, t);
            }
            self.version.set(version);
            self.reinit.set(reinit);
            self.history.borrow_mut().clear();
            self.history.borrow_mut().push(version as _);
            return;
        }
        // If this tree was not update after the last reinit, update it
        if data.num_reinits != self.reinit.get() {
            data.version += 1;
            let version = data.version;
            let reinit = data.num_reinits;
            for (i, t) in self.trees[1..].iter().enumerate() {
                NddrOneTreeForest::<G>::add_fat_node(&mut data, version, i + 1, t);
            }
            self.version.set(version);
            self.reinit.set(reinit);
            self.history.borrow_mut().clear();
            self.history.borrow_mut().push(version as _);
        }
    }

    // Subtree ops

    fn do_op1(&mut self, params: (usize, usize, usize, usize)) -> (Edge<G>, Edge<G>) {
        let (ifrom, p, ito, a) = params;
        let ins = self.edge_by_ends(self[ifrom][p].vertex(), self[ito][a].vertex());
        let rem = self.edge_by_ends(
            self[ifrom][p].vertex(),
            self[ifrom].parent_vertex(p).unwrap(),
        );

        if ifrom == ito {
            let new = one_tree_op1(&self[ifrom], p, a);
            self.replace1(ifrom, new);
        } else {
            let (from, to) = op1(&self[ifrom], p, &self[ito], a);
            self.replace(ifrom, from, ito, Some(to));
        }

        (ins, rem)
    }

    fn do_op2(&mut self, params: (usize, usize, usize, usize, usize)) -> (Edge<G>, Edge<G>) {
        let (ifrom, p, r, ito, a) = params;
        let ins = self.edge_by_ends(self[ifrom][r].vertex(), self[ito][a].vertex());
        let rem = self.edge_by_ends(
            self[ifrom][p].vertex(),
            self[ifrom].parent_vertex(p).unwrap(),
        );

        if ifrom == ito {
            let new = one_tree_op2(&self[ifrom], p, r, a);
            self.replace1(ifrom, new);
        } else {
            let (from, to) = op2(&self[ifrom], p, r, &self[ito], a);
            self.replace(ifrom, from, ito, Some(to));
        }

        (ins, rem)
    }

    fn can_insert_subtree_edge(&self, params: (usize, usize, usize, usize)) -> bool {
        let (from, p, to, a) = params;
        // The root of the subtree being transfered cannot be the root
        // p cannot be ancestor of a in one_tree_op1
        !(p == 0
            || from == to
                && (p == a
                    || self[from].contains_edge(self[from][p].vertex(), self[from][a].vertex())
                    || self[from].is_ancestor(p, a)))
    }

    fn find_vertices_op1<R: Rng>(&self, mut rng: R) -> (usize, usize, usize, usize) {
        if self.should_mutate_star_tree(&mut rng) {
            if let Some(ops) = self.find_op_star_edge(&mut rng) {
                return ops;
            }
        }
        for _ in 0..1_000_000 {
            let (from, p, to, a) = match self.find_op_strategy() {
                FindOpStrategy::Adj => self.find_op_adj(&mut rng),
                FindOpStrategy::AdjSmaller => self.find_op_adj_smaller(&mut rng),
                FindOpStrategy::Edge => self.find_op_edge(&mut rng),
                FindOpStrategy::Balanced => self.find_op_balanced(&mut rng),
            };

            // cannot be the star_tree
            assert!(from != 0);
            assert!(to != 0);

            if self.can_insert_subtree_edge((from, p, to, a)) {
                return (from, p, to, a);
            }
        }
        unreachable!("find_vertices_op1: could not find valid operands!")
    }

    fn find_vertices_op2<R: Rng>(&self, mut rng: R) -> (usize, usize, usize, usize, usize) {
        for _ in 0..1_000 {
            let (from, r, to, a) = self.find_vertices_op1(&mut rng);
            let mut count = 0;
            let mut p = r;
            while let Some(pp) = self[from].parent(p) {
                count += 1;
                p = pp;
            }

            p = r;
            for _ in 0..rng.gen_range(0, count) {
                p = self[from].parent(p).unwrap()
            }
            // If from == to, then one_tree_op2 will be called, for now, we restrict the value of p
            // of not an ancestor of a, see the comments on one_tree_op2 function. In the unit
            // tests, the loop is execute 1.361 times to find valid operands.
            // This is enough to run the experiments in complete graphs (and for general graphs?).
            if from != to || !self[from].is_ancestor(p, a) {
                return (from, p, r, to, a);
            }
        }
        unreachable!("find_vertices_op2: could not find valid operands!")
    }

    fn find_op_adj<R: Rng>(&self, mut rng: R) -> (usize, usize, usize, usize) {
        // Sec V-B - page 834
        // In this implementation p and a are indices, not vertices
        // v_p and v_a are vertices.

        // Step 1
        let from =
            self.select_tree_if(&mut rng, |i| {
                // Must have more than one node so we can choose a node != root
                self[i].deg_in_g() > self[i].deg() && self[i].len() > 1
            }).expect("The graph is a forest");

        // Step 2
        let p = self.select_tree_vertex_if(from, &mut rng, |i| {
            // i is not root and have an edge tha is not in this forest
            i != 0 && self[from][i].deg() < self.graph().out_degree(self[from][i].vertex()) as u32
        });

        // Step 3
        self.mark_edges(from);

        let v_a = {
            let data = self.data();
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
            self.graph()
                .choose_out_neighbor_iter(v_p, rng)
                .find(|&v_a| !data.is_marked(v_p, v_a))
                .unwrap()
        };

        self.unmark_edges(from);

        // Step 4
        let (to, a) = self.find_index(v_a);

        // Step 5 is executed by the callee, so just return
        (from, p, to, a)
    }

    fn find_op_adj_smaller<R: Rng>(&self, rng: R) -> (usize, usize, usize, usize) {
        let (from, p, to, a) = self.find_op_adj(rng);
        if self[from].len() >= self[to].len() {
            (from, p, to, a)
        } else {
            (to, a, from, p)
        }
    }

    fn find_op_edge<R: Rng>(&self, mut rng: R) -> (usize, usize, usize, usize) {
        let g = self.graph();
        // TODO: use tournament without replacement
        let mut e = g.choose_edge(&mut rng).unwrap();
        // The order of the vertices is important, so trying (u, v) is different from (v, u),
        // as only (u, v) or (v, u) can be returned from choose_edge, we try the reverse with 50%
        // of change.
        // This is not necessary in the other methods because (u, v) and (v, u) can be choosed.
        if rng.gen() {
            e = self.graph().reverse(e);
        }
        let (u, v) = g.ends(e);
        let (from, p) = self.find_index(u);
        let (to, a) = self.find_index(v);
        (from, p, to, a)
    }

    fn find_op_balanced<R: Rng>(&self, mut rng: R) -> (usize, usize, usize, usize) {
        let from = self.select_tree(&mut rng);
        let p = self.select_tree_vertex(from, &mut rng);
        let to = self.select_tree(&mut rng);
        let a = self.select_tree_vertex(to, &mut rng);
        (from, p, to, a)
    }

    // Star edges

    fn can_insert_star_edge(&self, ins: Edge<G>) -> bool {
        let (u, v) = self.graph().ends(ins);
        self[0][0].vertex() != u && !self[0].contains_edge(u, v)
    }

    fn find_op_star_edge<R: Rng>(&self, rng: &mut R) -> Option<(usize, usize, usize, usize)> {
        let e = {
            if let Some(&e) =
                sample_without_replacement(&mut *self.star_edges.borrow_mut(), rng, |e| {
                    self.can_insert_star_edge(*e)
                }) {
                e
            } else {
                // No edge can be inserted
                return None;
            }
        };
        let (u, v) = self.graph().ends(e);
        let p = self[0].find_vertex(u).unwrap();
        let a = self[0].find_vertex(v).unwrap();
        if !self[0].is_ancestor(p, a) {
            Some((0, p, 0, a))
        } else {
            Some((0, a, 0, p))
        }
    }

    // Misc

    fn replace1(&mut self, i: usize, ti: NddTree<Vertex<G>>) {
        self.replace(i, ti, 0, None);
    }

    fn replace(
        &mut self,
        i: usize,
        mut ti: NddTree<Vertex<G>>,
        j: usize,
        mut tj: Option<NddTree<Vertex<G>>>,
    ) {
        if self.find_vertex_strategy() == FindVertexStrategy::FatNode {
            self.new_version();
            let data = &mut *self.data_mut();
            Self::add_fat_node(data, self.version.get(), i, &ti);
            if let Some(ref tj) = tj {
                Self::add_fat_node(data, self.version.get(), j, tj);
            }
        }

        if self.find_vertex_strategy() == FindVertexStrategy::Map {
            let indices = &(self.data.borrow().vertex_index);
            self.maps[i] = Self::new_map(indices, i, &ti);
            if let Some(ref tj) = tj {
                self.maps[j] = Self::new_map(indices, j, tj);
            }
        }

        // TODO: call only when its needed
        ti.calc_degs(|v| self.graph().out_degree(v) as u32);
        if let Some(ref mut tj) = tj {
            tj.calc_degs(|v| self.graph().out_degree(v) as u32);
        }

        self.trees[i] = Rc::new(ti);
        if let Some(tj) = tj {
            self.trees[j] = Rc::new(tj);
        }
    }

    fn new_map(indices: &VertexIndexProp<G>, i: usize, ti: &NddTree<Vertex<G>>) -> Rc<Vec<usize>> {
        if i == 0 {
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

    fn add_fat_node(data: &mut Data<G>, version: usize, i: usize, ti: &NddTree<Vertex<G>>) {
        if i == 0 {
            return;
        }
        for (pos, n) in ti.iter().enumerate() {
            let v = data.vertex_index.get(n.vertex());
            data.pi[v].push(PiValue::new(version as _, i as _, pos as _));
        }
    }

    fn should_mutate_star_tree<R: Rng>(&self, rng: &mut R) -> bool {
        // Do it with probability 1/nsqrt
        rng.gen_range(0, self.data().nsqrt + 1) == 0
    }

    fn select_tree_if<R, F>(&self, rng: &mut R, mut f: F) -> Option<usize>
    where
        R: Rng,
        F: FnMut(usize) -> bool,
    {
        sample_without_replacement(&mut *self.data_mut().tree_indices, rng, |&i| f(i)).cloned()
    }

    fn select_tree<R: Rng>(&self, rng: &mut R) -> usize {
        rng.gen_range(1, self.len())
    }

    fn select_tree_vertex<R: Rng>(&self, i: usize, rng: &mut R) -> usize {
        rng.gen_range(0, self[i].len())
    }

    fn select_tree_vertex_if<R: Rng, F>(&self, i: usize, rng: &mut R, mut f: F) -> usize
    where
        F: FnMut(usize) -> bool,
    {
        for _ in 0..10 * self[i].len() {
            let i = rng.gen_range(0, self[i].len());
            if f(i) {
                return i;
            }
        }
        unreachable!("select_tree_vertex_if: could not find vertex!")
    }

    fn find_index(&self, v: Vertex<G>) -> (usize, usize) {
        match self.find_vertex_strategy() {
            FindVertexStrategy::FatNode => {
                let vi = self.data().vertex_index.get(v);
                for &ver in self.history.borrow().iter().rev() {
                    if let Ok(i) =
                        self.data().pi[vi].binary_search_by(|p| p.version.cmp(&(ver as _)))
                    {
                        let t = self.data().pi[vi][i].tree as usize;
                        let p = self.data().pi[vi][i].pos as usize;
                        assert_eq!(v, self[t][p].vertex());
                        return (t, p);
                    }
                }
                panic!("find_index with FatNode did not find the vertex");
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
                panic!("find_index with Map did not find the vertex");
            }
        }
    }

    fn mark_edges(&self, t: usize) {
        self.set_edges_on_m(t, true);
    }

    fn unmark_edges(&self, t: usize) {
        self.set_edges_on_m(t, false);
    }

    fn set_edges_on_m(&self, t: usize, value: bool) {
        let mut data = self.data_mut();
        self[t].for_each_edge(|u, v| data.set_m(u, v, value));
    }

    fn new_version(&mut self) {
        self.data_mut().version += 1;
        let version = self.data().version;
        self.version = Cell::new(version);
        self.history.borrow_mut().push(version as _);
    }

    pub fn edges(&self) -> Vec<Edge<G>> {
        self.iter()
            .flat_map(|t| t.edges())
            .map(|(u, v)| self.edge_by_ends(u, v))
            .collect()
    }

    fn edge_by_ends(&self, u: Vertex<G>, v: Vertex<G>) -> Edge<G> {
        self.graph().edge_by_ends(u, v)
    }

    fn find_op_strategy(&self) -> FindOpStrategy {
        self.data().find_op_strategy
    }

    fn find_vertex_strategy(&self) -> FindVertexStrategy {
        self.data().find_vertex_strategy
    }

    fn _degree(&self, u: Vertex<G>) -> u32 {
        let (t, i) = self.find_index(u);
        self[t][i].deg() + if i == 0 {
            // u is a root so get the degree on start_edge
            self[0][self[0].find_vertex(u).unwrap()].deg()
        } else {
            0
        }
    }

    fn data(&self) -> Ref<Data<G>> {
        self.data.borrow()
    }

    fn data_mut(&self) -> RefMut<Data<G>> {
        self.data.borrow_mut()
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
    use random_sp;

    #[test]
    fn test_eq() {
        let n = 30;
        let mut rng = rand::weak_rng();
        let (g, data, mut tree) = data(n, FindOpStrategy::Adj, FindVertexStrategy::FatNode);
        let data = Rc::new(RefCell::new(data));
        let forests = vec((0..n).map(|_| {
            rng.shuffle(&mut tree);
            NddrOneTreeForest::new_with_data(g.clone(), data.clone(), tree.clone(), &mut rng)
        }));

        for i in 0..(n as usize) {
            for j in 0..(n as usize) {
                assert!(forests[i] == forests[j]);
            }
        }
    }

    fn data(
        n: u32,
        find_op: FindOpStrategy,
        find_vertex: FindVertexStrategy,
    ) -> (
        Rc<CompleteGraph>,
        Data<CompleteGraph>,
        Vec<Edge<CompleteGraph>>,
    ) {
        let mut rng = rand::weak_rng();
        let g = Rc::new(CompleteGraph::new(n));
        let tree = random_sp(&g, &mut rng);
        let data = Data::new(&g, find_op, find_vertex);
        (g, data, tree)
    }

    macro_rules! def_test {
        ($name:ident, $op:ident, $vertex:ident) => {
            #[test]
            fn $name() {
                let mut rng = rand::weak_rng();
                let (g, data, tree) = data(100, FindOpStrategy::$op, FindVertexStrategy::$vertex);
                let data = Rc::new(RefCell::new(data));
                let mut forest = NddrOneTreeForest::new_with_data(g, data, tree, &mut rng);
                for _ in 0..100 {
                    let mut rng = rand::weak_rng();
                    for &op in &[1, 2] {
                        let mut f = forest.clone();
                        assert!(forest == f);
                        let (ins, rem) = if op == 1 {
                            f.op1(&mut rng)
                        } else {
                            f.op2(&mut rng)
                        };
                        assert!(f.contains(ins));
                        assert!(!f.contains(rem));
                        f.check();
                        assert!(forest != f);
                        forest = f;
                    }
                }
            }
        };
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
