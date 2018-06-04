use std::ops::{Deref, DerefMut};

use fixedbitset::FixedBitSet;

#[derive(Default, Clone, Debug)]
pub struct Bitset(FixedBitSet);

impl Bitset {
    #[inline]
    pub fn with_capacity(n: usize) -> Self {
        Bitset(FixedBitSet::with_capacity(n))
    }

    #[inline]
    pub fn unchecked_set(&mut self, i: usize) {
        let b = i / 32;
        let i = i % 32;
        unsafe { *self.0.as_mut_slice().get_unchecked_mut(b) |= 1 << i };
    }

    #[inline]
    pub fn unchecked_get(&self, i: usize) -> bool {
        let b = i / 32;
        let i = i % 32;
        unsafe { self.0.as_slice().get_unchecked(b) & (1 << i) != 0 }
    }

    #[inline]
    pub fn unchecked_clear_block(&mut self, i: usize) {
        unsafe { *self.0.as_mut_slice().get_unchecked_mut(i / 32) = 0 };
    }

    #[inline]
    pub fn clear_block(&mut self, i: usize) {
        self.0.as_mut_slice()[i / 32] = 0;
    }
}

impl Deref for Bitset {
    type Target = FixedBitSet;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for Bitset {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
