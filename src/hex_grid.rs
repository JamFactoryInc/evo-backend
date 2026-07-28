//! Axial hex-grid math, dependency-free (previously drafted against `bevy`).
//!
//! Coordinates are axial `(q, r)`. The six directions and their offsets keep
//! the original convention:
//!
//! ```text
//!   NW(-1, 1)   NE(0, 1)
//!         \    /
//!   W(-1,0)-- * --E(1,0)
//!         /    \
//!   SW(0,-1)   SE(1,-1)
//! ```
//!
//! Direction `d` and `d.opposite()` (`(d+3)%6`) address the two ends of a bond,
//! which is what lets one atom's output feed the facing neighbour's input.

use std::collections::HashMap;

/// √3 / 2, the vertical step of a pointy-top hex row.
pub const H: f32 = 0.866_025_4;

/// Axial hex coordinate.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Default)]
pub struct Axial {
    pub q: i32,
    pub r: i32,
}

impl Axial {
    pub const ORIGIN: Axial = Axial { q: 0, r: 0 };

    #[inline]
    pub fn new(q: i32, r: i32) -> Self {
        Self { q, r }
    }

    /// Neighbour in direction `d`.
    #[inline]
    pub fn step(self, d: Dir) -> Axial {
        let (dq, dr) = d.offset();
        Axial::new(self.q + dq, self.r + dr)
    }

    /// All six neighbours, indexed by `Dir as usize`.
    #[inline]
    pub fn neighbors(self) -> [Axial; 6] {
        let mut out = [Axial::ORIGIN; 6];
        for (i, o) in out.iter_mut().enumerate() {
            *o = self.step(Dir::ALL[i]);
        }
        out
    }

    /// World-space (pixel) centre for a hex of the given size (y grows with r;
    /// note Godot screen-y is downward, which is fine — it is only a mapping).
    #[inline]
    pub fn to_world(self, size: f32) -> (f32, f32) {
        let x = size * (self.q as f32 + 0.5 * self.r as f32);
        let y = size * H * self.r as f32;
        (x, y)
    }
}

/// One of the six hex bond directions.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Dir {
    NorthWest = 0,
    NorthEast = 1,
    East = 2,
    SouthEast = 3,
    SouthWest = 4,
    West = 5,
}

impl Dir {
    pub const ALL: [Dir; 6] = [
        Dir::NorthWest,
        Dir::NorthEast,
        Dir::East,
        Dir::SouthEast,
        Dir::SouthWest,
        Dir::West,
    ];

    #[inline]
    pub fn from_index(i: usize) -> Dir {
        Dir::ALL[i % 6]
    }

    #[inline]
    pub fn index(self) -> usize {
        self as usize
    }

    /// Axial offset for this direction.
    #[inline]
    pub fn offset(self) -> (i32, i32) {
        match self {
            Dir::NorthWest => (-1, 1),
            Dir::NorthEast => (0, 1),
            Dir::East => (1, 0),
            Dir::SouthEast => (1, -1),
            Dir::SouthWest => (0, -1),
            Dir::West => (-1, 0),
        }
    }

    /// The bond site facing this one on the neighbouring atom.
    #[inline]
    pub fn opposite(self) -> Dir {
        Dir::from_index((self.index() + 3) % 6)
    }

    /// Unit vector pointing this way in local (unrotated) world space.
    #[inline]
    pub fn unit(self) -> (f32, f32) {
        match self {
            Dir::NorthWest => (-0.5, H),
            Dir::NorthEast => (0.5, H),
            Dir::East => (1.0, 0.0),
            Dir::SouthEast => (0.5, -H),
            Dir::SouthWest => (-0.5, -H),
            Dir::West => (-1.0, 0.0),
        }
    }
}

/// Values attached to each of the six neighbours of a cell.
#[derive(Default, Clone)]
pub struct Neighbors<T> {
    pub sites: [T; 6],
}

impl<T> Neighbors<T> {
    pub fn get(&self, d: Dir) -> &T {
        &self.sites[d.index()]
    }
    pub fn get_mut(&mut self, d: Dir) -> &mut T {
        &mut self.sites[d.index()]
    }
}

/// A sparse, unbounded hex grid. Kept generic for reuse (e.g. a global spatial
/// index) even though molecules carry their own small coordinate maps.
pub struct HexGrid<T> {
    cells: HashMap<Axial, T>,
}

impl<T> Default for HexGrid<T> {
    fn default() -> Self {
        Self {
            cells: HashMap::new(),
        }
    }
}

impl<T> HexGrid<T> {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn get(&self, c: Axial) -> Option<&T> {
        self.cells.get(&c)
    }
    pub fn get_mut(&mut self, c: Axial) -> Option<&mut T> {
        self.cells.get_mut(&c)
    }
    pub fn insert(&mut self, c: Axial, v: T) -> Option<T> {
        self.cells.insert(c, v)
    }
    pub fn remove(&mut self, c: Axial) -> Option<T> {
        self.cells.remove(&c)
    }
    pub fn contains(&self, c: Axial) -> bool {
        self.cells.contains_key(&c)
    }
    pub fn len(&self) -> usize {
        self.cells.len()
    }
    pub fn is_empty(&self) -> bool {
        self.cells.is_empty()
    }
    pub fn iter(&self) -> impl Iterator<Item = (&Axial, &T)> {
        self.cells.iter()
    }

    /// The values on the six neighbours of `c` (None where empty).
    pub fn neighbors(&self, c: Axial) -> Neighbors<Option<&T>> {
        let mut n: Neighbors<Option<&T>> = Neighbors {
            sites: [None, None, None, None, None, None],
        };
        for d in Dir::ALL {
            n.sites[d.index()] = self.get(c.step(d));
        }
        n
    }
}
