#![cfg_attr(test, feature(slice_rotate))]
#![feature(vec_remove_item)]
#![cfg_attr(feature = "cargo-clippy", warn(clone_on_ref_ptr))]

extern crate fera;
extern crate fera_array;
extern crate fixedbitset;
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
