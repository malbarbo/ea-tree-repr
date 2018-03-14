#![cfg_attr(test, feature(slice_rotate))]

extern crate fera;
extern crate fera_array;
extern crate pbr;
extern crate rand;

mod bitset;
mod cowprop;
mod euler;
mod nddr;
mod parent;
mod progress;
mod random;
mod tree;

pub use self::bitset::*;
pub use self::cowprop::*;
pub use self::euler::*;
pub use self::nddr::*;
pub use self::parent::*;
pub use self::progress::*;
pub use self::random::*;
pub use self::tree::*;

use std::time::Duration;

pub fn nano_secs(t: Duration) -> u64 {
    1_000_000_000 * t.as_secs() + t.subsec_nanos() as u64
}

pub fn micro_secs(t: Duration) -> f64 {
    1_000_000.0 * t.as_secs() as f64 + t.subsec_nanos() as f64 / 1_000.0
}

pub fn milli_secs(t: Duration) -> f64 {
    1_000.0 * t.as_secs() as f64 + t.subsec_nanos() as f64 / 1_000_000.0
}

pub fn setup_rayon() {
    if ::std::env::var("RAYON_NUM_THREADS").is_err() {
        ::std::env::set_var("RAYON_NUM_THREADS", "1");
    }
}
