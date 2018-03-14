use std::cell::RefCell;
use std::rc::Rc;

pub type Bitset = Vec<bool>;

pub fn bitset_pool() -> Rc<RefCell<Vec<Bitset>>> {
    thread_local! {
        pub static BUFFER: Rc<RefCell<Vec<Bitset>>> = Rc::new(RefCell::new(vec![vec![]]));
    }
    BUFFER.with(|f| f.clone())
}

pub fn bitset_acquire(n: usize) -> Bitset {
    let pool = bitset_pool();
    let mut pool = pool.borrow_mut();
    let mut bitset = pool.pop().unwrap_or_else(|| vec![false; n]);
    if bitset.len() < n {
        bitset.resize(n, false);
    }
    bitset
}

pub fn bitset_release(bitset: Bitset) {
    let pool = bitset_pool();
    let mut pool = pool.borrow_mut();
    pool.push(bitset);
}
