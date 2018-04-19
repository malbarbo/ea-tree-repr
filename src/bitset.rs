use std::cell::RefCell;
use std::rc::Rc;

use fixedbitset::FixedBitSet;

pub type Bitset = FixedBitSet;

pub fn bitset_pool() -> Rc<RefCell<Vec<Bitset>>> {
    thread_local! {
        pub static BUFFER: Rc<RefCell<Vec<Bitset>>> = Rc::new(RefCell::new(vec![Default::default()]));
    }
    BUFFER.with(|f| Rc::clone(f))
}

pub fn bitset_acquire(n: usize) -> Bitset {
    let pool = bitset_pool();
    let mut pool = pool.borrow_mut();
    let mut bitset = pool.pop().unwrap_or_else(|| FixedBitSet::with_capacity(n));
    if bitset.len() < n {
        bitset.grow(n);
    }
    bitset
}

pub fn bitset_release(bitset: Bitset) {
    let pool = bitset_pool();
    let mut pool = pool.borrow_mut();
    pool.push(bitset);
}
