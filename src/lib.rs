#![cfg_attr(feature = "cargo-clippy", warn(clone_on_ref_ptr))]

#[cfg(feature = "system_allocator")]
#[global_allocator] static A: std::alloc::System = std::alloc::System;

extern crate fera;
extern crate fera_array;
#[macro_use]
extern crate log;
extern crate pbr;
extern crate rand;

mod bitset;
mod cowprop;
mod euler;
#[cfg(test)]
mod euler_simple;
mod logger;
mod nddr;
mod predecessor;
mod progress;
mod random;
mod tree;

pub use self::bitset::*;
pub use self::cowprop::*;
pub use self::euler::*;
pub use self::logger::*;
pub use self::nddr::*;
pub use self::predecessor::*;
pub use self::progress::*;
pub use self::random::*;
pub use self::tree::*;

use std::cell::RefCell;
use std::time::{Duration, Instant};

pub fn time_it<T, F: FnOnce() -> T>(fun: F) -> (Duration, T) {
    let start = Instant::now();
    let r = fun();
    (start.elapsed(), r)
}

pub fn nano_secs(t: Duration) -> u64 {
    1_000_000_000 * t.as_secs() + u64::from(t.subsec_nanos())
}

pub fn micro_secs(t: Duration) -> f64 {
    1_000_000.0 * t.as_secs() as f64 + f64::from(t.subsec_nanos()) / 1_000.0
}

pub fn milli_secs(t: Duration) -> f64 {
    1_000.0 * t.as_secs() as f64 + f64::from(t.subsec_nanos()) / 1_000_000.0
}

pub fn setup_rayon() {
    if ::std::env::var("RAYON_NUM_THREADS").is_err() {
        ::std::env::set_var("RAYON_NUM_THREADS", "1");
    }
}

pub unsafe fn transmute_lifetime<'a, 'b, T: ?Sized>(value: &'a T) -> &'b T {
    ::std::mem::transmute(value)
}

pub struct Linspace {
    last: usize,
    cur: usize,
    step: f64,
}

impl Linspace {
    fn new(last: usize, step: f64) -> Self {
        Self { last, cur: 0, step }
    }
}

impl Iterator for Linspace {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        if self.cur < self.last {
            self.cur += 1;
            let value = (self.cur as f64) * self.step;
            Some(value.round() as usize)
        } else {
            None
        }
    }
}

#[derive(Clone)]
pub struct Pool<T>(RefCell<Vec<T>>);

impl<T> Default for Pool<T> {
    fn default() -> Self {
        Pool(RefCell::new(vec![]))
    }
}

impl<T> Pool<T> {
    pub fn new() -> Self {
        Pool::default()
    }

    pub fn acquire(&self) -> Option<T> {
        self.0.borrow_mut().pop()
    }

    pub fn release(&self, value: T) {
        self.0.borrow_mut().push(value);
    }
}
