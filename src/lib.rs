#![feature(slice_rotate)]

extern crate fera;
extern crate fera_array;
extern crate pbr;
extern crate rand;

mod euler;
mod nddr;
mod parent;
mod progress;
mod random;
mod tour;
mod tree;

pub use self::euler::*;
pub use self::nddr::*;
pub use self::parent::*;
pub use self::progress::*;
pub use self::random::*;
pub use self::tree::*;

use std::time::Duration;

pub fn micro_secs(t: Duration) -> u64 {
    let micro = 1_000_000.0 * t.as_secs() as f64 + f64::from(t.subsec_nanos()) / 1_000.0;
    micro as u64
}
