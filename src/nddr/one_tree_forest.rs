// external
use fera::ext::VecExt;
use fera::fun::vec;
use fera::graph::algs::Trees;
use fera::graph::choose::Choose;
use fera::graph::prelude::*;
use rand::Rng;

// system
use std::cell::{Ref, RefCell, RefMut};
use std::rc::Rc;
use std::ops::Deref;

// internal
use {collect_ndds, find_star_tree, NddTree, one_tree_op1, one_tree_op2, op1, op2};

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

// One instance of Data is shared between many NddrOneTreeForest instances.
struct Data<G>
where
    G: Graph + WithVertexIndexProp + WithVertexProp<DefaultVertexPropMut<G, bool>>,
{
    // TODO: remove Rc inside Rc, move g to Forest?
    g: Rc<G>,
    nsqrt: usize,
    find_op_strategy: FindOpStrategy,
    find_vertex_strategy: FindVertexStrategy,

    // Used if find_vertex_strategy == Map to map each vertex to a number in 0..n
    vertex_index: VertexIndexProp<G>,

    // See section VI-B - 3
    // The next two field are used if find_op_strategy = Adj or AdjSmaller used in sample without
    // replacement in select_tree_if. This does the role of the L_O array described in the paper.
    tree_indices: Vec<usize>,
    // Used to mark edges in a tree (M_E_T in the paper)
    // TODO: Use a thread local bitmap
    m: Option<DefaultVertexPropMut<G, DefaultVertexPropMut<G, bool>>>,

    // The next fields are use if find_vertex_strategy = FatNode. In forest version h, the vertex
    // x was in position pi_pos[x][h] of the tree with index pi_tree[x][h]
    version: usize,
    // FIXME: implements reinitialization (In section VI-B - 5 - is not said which value was
    // used for k)
    pi_version: DefaultVertexPropMut<G, Vec<usize>>,
    pi_tree: DefaultVertexPropMut<G, Vec<usize>>,
    pi_pos: DefaultVertexPropMut<G, Vec<usize>>,
}

impl<G> Data<G>
where
    G: Graph + WithVertexIndexProp + WithVertexProp<DefaultVertexPropMut<G, bool>>,
    DefaultVertexPropMut<G, bool>: Clone,
{
    pub fn new(g: Rc<G>, find_op: FindOpStrategy, find_vertex: FindVertexStrategy) -> Self {
        let nsqrt = (g.num_vertices() as f64).sqrt().ceil() as usize;
        let m = match find_op {
            FindOpStrategy::Adj | FindOpStrategy::AdjSmaller => {
                Some(g.vertex_prop(g.vertex_prop(false)))
            }
            _ => None,
        };
        let vertex_index = g.vertex_index();
        // 0 is the index of the star tree, so its keeped out
        let tree_indices = vec(1..nsqrt + 1);
        let pi_version = g.vertex_prop(Vec::<usize>::new());
        let pi_tree = g.vertex_prop(Vec::<usize>::new());
        let pi_pos = g.vertex_prop(Vec::<usize>::new());
        Data {
            g: g,
            nsqrt: nsqrt,
            find_op_strategy: find_op,
            find_vertex_strategy: find_vertex,
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
}

pub struct NddrOneTreeForest<G>
where
    G: Graph + WithVertexIndexProp + WithVertexProp<DefaultVertexPropMut<G, bool>>,
{
    data: Rc<RefCell<Data<G>>>,

    last_op_size: usize,

    // The star tree is kept in trees[0]. The star tree connects the roots of the trees.
    // TODO: Use RcArray
    trees: Vec<Rc<NddTree<Vertex<G>>>>,
    // Edges (u, v) such that u and v are star tree vertices.
    star_edges: Rc<RefCell<Vec<Edge<G>>>>,

    // Used if find_vertex_strategy == Map. This does not allocate if find_vertex_strategy != Map.
    //
    // if trees[i][maps[i][vertex_index[v]]].vertex == v,
    // so v is in tree[i] in position maps[i][vertex_index[v]]
    // TODO: move this to NddTree
    maps: Vec<Rc<Vec<usize>>>,

    // Used if find_vertex_strategy = FatNode
    //
    // FIXME: implements a scheme to reset the history
    // In page 836, B-5) L = history, what is the value of k?
    version: usize,
    history: Vec<usize>,
}

impl<G> Clone for NddrOneTreeForest<G>
where
    G: Graph + WithVertexIndexProp + WithVertexProp<DefaultVertexPropMut<G, bool>>,
{
    fn clone(&self) -> Self {
        Self {
            data: Rc::clone(&self.data),
            last_op_size: self.last_op_size,
            trees: self.trees.clone(),
            star_edges: Rc::clone(&self.star_edges),
            maps: self.maps.clone(),
            version: self.version,
            // TODO: Should history be shared? How this affects running time?
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

        // TODO: make it faster
        for e in self.edges_vec() {
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
        Self::new_with_data(
            Rc::new(RefCell::new(Data::new(g, find_op, find_vertex))),
            edges,
            rng,
        )
    }

    fn new_with_data<R: Rng>(data: Rc<RefCell<Data<G>>>, mut edges: Vec<Edge<G>>, rng: R) -> Self {
        let data_ = data;
        // use a clone to make the borrow checker happy
        let data = Rc::clone(&data_);
        let mut data = data.borrow_mut();
        let nsqrt = data.nsqrt;

        let star_tree = {
            let s = data.g().spanning_subgraph(&edges);
            let r = s.choose_vertex(rng).unwrap();
            Rc::new(NddTree::new(find_star_tree(&s, r, nsqrt)))
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

        // TODO: try to create a balanced NddrOneTreeForest
        trees.extend({
            let s = data.g().spanning_subgraph(edges);
            collect_ndds(&s, &roots).into_iter().map(|t| {
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
                NddrOneTreeForest::<G>::add_fat_node(data, version, i + 1, t);
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

        NddrOneTreeForest {
            data: data_,
            last_op_size: 0,
            trees: trees,
            maps: maps,
            version: version,
            history: vec![version],
            star_edges: Rc::new(RefCell::new(star_edges)),
        }
    }

    pub fn last_op_size(&self) -> usize {
        self.last_op_size
    }

    // TODO: create a version of op1 and op2 that takes degree constraint parameter
    pub fn op1<R: Rng>(&mut self, rng: R) -> (Edge<G>, Edge<G>) {
        let (from, p, to, a) = self.find_vertices_op1(rng);
        self.last_op_size = self[from].len() + self[to].len();
        self.do_op1((from, p, to, a))
    }

    pub fn op2<R: Rng>(&mut self, rng: R) -> (Edge<G>, Edge<G>) {
        let (from, p, r, to, a) = self.find_vertices_op2(rng);
        self.last_op_size = self[from].len() + self[to].len();
        self.do_op2((from, p, r, to, a))
    }

    pub fn contains(&self, e: Edge<G>) -> bool {
        // TODO: what is the execution time?
        let (u, v) = self.g().ends(e);
        self.trees.iter().any(|t| t.contains_edge(u, v))
    }

    pub fn check(&self) {
        let edges = self.edges_vec();
        for &e in &edges {
            assert!(self.contains(e));
        }

        let g = self.g();
        let sub = g.spanning_subgraph(&edges);
        assert!(sub.is_tree());
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
            self.replace(ifrom, from, ito, to);
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
            self.replace(ifrom, from, ito, to);
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
            return self.find_op_star_edge(&mut rng);
        }
        // TODO: add limit
        loop {
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
    }

    fn find_vertices_op2<R: Rng>(&self, mut rng: R) -> (usize, usize, usize, usize, usize) {
        loop {
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
            // This is enough to run the experiments in complete graphs.
            if from != to || !self[from].is_ancestor(p, a) {
                return (from, p, r, to, a);
            }
        }
    }

    fn find_op_adj<R: Rng>(&self, mut rng: R) -> (usize, usize, usize, usize) {
        // Sec V-B - page 834
        // In this implementation p and a are indices, not vertices
        // v_p and v_a are vertices.

        // Step 1
        let from = self.select_tree_if(&mut rng, |i| {
            // Must have more than one node so we can choose a node != root
            self[i].deg_in_g() > self[i].deg() && self[i].len() > 1
        }).expect("The graph is a forest");

        // Step 2
        let p = self.select_tree_vertex_if(from, &mut rng, |i| {
            // i is not root and have an edge tha is not in this forest
            i != 0 && self[from][i].deg() < self.g().out_degree(self[from][i].vertex())
        });

        // Step 3
        self.mark_edges(from);

        let v_a = {
            let data = self.data();
            let m = data.m.as_ref().unwrap();
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
                .choose_out_neighbor_iter(v_p, rng)
                .find(|&v_a| !m[v_p][v_a] && !m[v_a][v_p])
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
        let g = self.g();
        // TODO: use tournament without replacement
        let mut e = g.choose_edge(&mut rng).unwrap();
        // The order of the vertices is important, so trying (u, v) is different from (v, u),
        // as only (u, v) or (v, u) can be returned from choose_edge, we trye th reverse with 50%
        // of change.
        // This is not necessary in the other methods because (u, v) and (v, u) can be choosed.
        if rng.gen() {
            e = self.g().reverse(e);
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
        let (u, v) = self.g().ends(ins);
        // FIXME: contains_edge is linear!, this can compromisse O(sqrt(n)) time
        // for non complete graph
        // FIXME: whe this function is called from find_op_star_edge find_vertex in called twice
        self[0][0].vertex() != u && !self[0].contains_edge(u, v) && {
            let p = self[0].find_vertex(u).unwrap();
            let a = self[0].find_vertex(v).unwrap();
            !self[0].is_ancestor(p, a)
        }
    }

    fn find_op_star_edge<R: Rng>(&self, rng: &mut R) -> (usize, usize, usize, usize) {
        let e = {
            *sample_without_replacement(&mut *self.star_edges.borrow_mut(), rng, |e| {
                self.can_insert_star_edge(*e)
            }).unwrap()
        };
        let (u, v) = self.g().ends(e);
        let p = self[0].find_vertex(u).unwrap();
        let a = self[0].find_vertex(v).unwrap();
        (0, p, 0, a)
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

    fn new_map(indices: &VertexIndexProp<G>, i: usize, ti: &NddTree<Vertex<G>>) -> Rc<Vec<usize>> {
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

    fn add_fat_node(data: &mut Data<G>, version: usize, i: usize, ti: &NddTree<Vertex<G>>) {
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

    fn should_mutate_star_tree<R: Rng>(&self, rng: &mut R) -> bool {
        // TODO: explain
        // For now, we are only interested in moving subtrees. When we implement a real GA, we should
        // enable this
        let f = || false;
        f() && self.trees[0].len() > 2
            && 2 * self.star_edges.borrow().len() >= self.trees[0].len()
            && rng.gen_range(0, self.data().nsqrt + 1) == self.data().nsqrt
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
        // TODO: add a limit or use sample without replacement
        loop {
            let i = rng.gen_range(0, self[i].len());
            if f(i) {
                return i;
            }
        }
    }

    fn find_index(&self, v: Vertex<G>) -> (usize, usize) {
        match self.find_vertex_strategy() {
            FindVertexStrategy::FatNode => {
                for ver in self.history.iter().rev() {
                    if let Ok(i) = self.data().pi_version[v].binary_search(ver) {
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

    fn set_edges_on_m(&self, t: usize, value: bool) {
        let mut data = self.data_mut();
        let m = data.m.as_mut().unwrap();
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
        vec(v.iter()
            .flat_map(|t| t.iter())
            .map(|&(u, v)| self.edge_by_ends(u, v)))
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

    fn _degree(&self, u: Vertex<G>) -> usize {
        let (t, i) = self.find_index(u);
        self[t][i].deg() + if i == 0 {
            // u is a root so get the degree on start_edge
            self[0][self[0].find_vertex(u).unwrap()].deg()
        } else {
            0
        }
    }

    fn g(&self) -> Ref<G> {
        Ref::map(self.data(), |d| d.g())
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
        let (data, mut tree) = data(n, FindOpStrategy::Adj, FindVertexStrategy::FatNode);
        let data = Rc::new(RefCell::new(data));
        let forests = vec((0..n).map(|_| {
            rng.shuffle(&mut tree);
            NddrOneTreeForest::new_with_data(data.clone(), tree.clone(), &mut rng)
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
    ) -> (Data<CompleteGraph>, Vec<Edge<CompleteGraph>>) {
        let mut rng = rand::weak_rng();
        let g = Rc::new(CompleteGraph::new(n));
        let tree = random_sp(&g, &mut rng);
        (Data::new(g, find_op, find_vertex), tree)
    }

    macro_rules! def_test {
        ($name:ident, $op:ident, $vertex:ident) => (
            #[test]
            fn $name() {
                let mut rng = rand::weak_rng();
                let (data, tree) = data(100, FindOpStrategy::$op, FindVertexStrategy::$vertex);
                let data = Rc::new(RefCell::new(data));
                let mut forest = NddrOneTreeForest::new_with_data(data, tree, &mut rng);
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
