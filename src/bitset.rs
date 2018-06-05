type Block = u32;
const BITS: usize = 8 * ::std::mem::size_of::<Block>();

#[derive(Default, Clone, Debug)]
pub struct Bitset {
    len: usize,
    blocks: Vec<Block>,
}

impl Bitset {
    #[inline]
    pub fn with_capacity(bits: usize) -> Self {
        let (mut blocks, extra) = div_rem(bits);
        blocks = blocks + ((extra != 0) as usize);
        Bitset {
            len: bits,
            blocks: vec![0; blocks],
        }
    }

    #[inline]
    pub fn unchecked_set(&mut self, i: usize) {
        let (b, i) = div_rem(i);
        unsafe { *self.blocks.get_unchecked_mut(b) |= 1 << i };
    }

    #[inline]
    pub fn unchecked_get(&self, i: usize) -> bool {
        let (b, i) = div_rem(i);
        unsafe { self.blocks.get_unchecked(b) & (1 << i) != 0 }
    }

    #[inline]
    pub fn unchecked_clear_block(&mut self, i: usize) {
        unsafe { *self.blocks.get_unchecked_mut(i / BITS) = 0 };
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }
}

#[inline]
fn div_rem(n: usize) -> (usize, usize) {
    (n / BITS, n % BITS)
}
