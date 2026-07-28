//! Tiny dependency-free PRNG (xorshift128+ style) so the sim is deterministic
//! for a given seed and adds no crates to the build.

pub struct Rng {
    s0: u64,
    s1: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        // SplitMix64 to spread a single seed into two non-zero state words.
        let mut z = seed.wrapping_add(0x9E3779B97F4A7C15);
        let mut mix = || {
            z = z.wrapping_add(0x9E3779B97F4A7C15);
            let mut x = z;
            x = (x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94D049BB133111EB);
            x ^ (x >> 31)
        };
        let a = mix();
        let b = mix();
        Self {
            s0: a | 1,
            s1: b | 1,
        }
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        // xorshift128+
        let mut x = self.s0;
        let y = self.s1;
        self.s0 = y;
        x ^= x << 23;
        x ^= x >> 17;
        x ^= y ^ (y >> 26);
        self.s1 = x;
        x.wrapping_add(y)
    }

    /// Uniform in [0, 1).
    #[inline]
    pub fn f32(&mut self) -> f32 {
        // Top 24 bits -> mantissa.
        ((self.next_u64() >> 40) as f32) / (1u32 << 24) as f32
    }

    /// Uniform in [lo, hi).
    #[inline]
    pub fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.f32()
    }

    /// Uniform integer in [0, n).
    #[inline]
    pub fn below(&mut self, n: usize) -> usize {
        if n == 0 {
            0
        } else {
            (self.next_u64() % n as u64) as usize
        }
    }

    /// True with probability `p`.
    #[inline]
    pub fn chance(&mut self, p: f32) -> bool {
        self.f32() < p
    }

    /// Weighted choice: returns an index into `weights` proportional to weight.
    pub fn weighted(&mut self, weights: &[f32]) -> usize {
        let total: f32 = weights.iter().copied().sum();
        if total <= 0.0 {
            return self.below(weights.len().max(1));
        }
        let mut t = self.f32() * total;
        for (i, w) in weights.iter().enumerate() {
            t -= *w;
            if t <= 0.0 {
                return i;
            }
        }
        weights.len() - 1
    }
}
