#![cfg_attr(test, feature(slice_rotate))]
#![feature(option_filter)]

extern crate fera;
extern crate fera_array;
extern crate pbr;
extern crate rand;
extern crate rpds;

mod nddr;
mod parent;
mod progress;
mod random;
mod tour;
mod tree;

pub use self::nddr::*;
pub use self::parent::*;
pub use self::progress::*;
pub use self::random::*;
pub use self::tour::*;
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
