//! The heritable blueprint of a molecule and the operators that create and
//! mutate it.
//!
//! A genome is just an ordered list of placed atoms (gene 0 is always the
//! `Seed` at the origin). Growth is *organic*: atoms are added one at a time
//! onto free neighbouring cells, and the kind of each new atom is drawn from a
//! preference table keyed on its parent's kind. That bias is what makes
//! functional chains — Solar → Conductor → Gate → Thruster, or Sensor → Gate —
//! far more likely than random noise, which raises the odds that a fresh
//! molecule does something coherent instead of nothing.

use std::collections::{HashMap, HashSet, VecDeque};
use fxhash::{FxHashMap, FxHashSet};
use crate::atom::{AtomKind, GateOp};
use crate::config::*;
use crate::hex_grid::{Axial, Dir};
use crate::rng::Rng;

/// One placed atom in a blueprint.
#[derive(Copy, Clone, Debug)]
pub struct Gene {
    pub coord: Axial,
    pub kind: AtomKind,
    pub facing: Dir,
    /// Kind-specific scalar (battery initial charge fraction, etc.).
    pub param: f32,
}

#[derive(Clone)]
pub struct Genome {
    pub genes: Vec<Gene>,
}

/// Coarse categories used only by the adjacency-preference table.
#[derive(Copy, Clone)]
enum Cat {
    Conductor,
    Solar,
    Gate,
    Battery,
    Thruster,
    Eater,
    Sensor,
}

fn cat_of(k: AtomKind) -> Cat {
    match k {
        AtomKind::Conductor | AtomKind::Seed => Cat::Conductor,
        AtomKind::Solar => Cat::Solar,
        AtomKind::Gate(_) => Cat::Gate,
        AtomKind::Battery => Cat::Battery,
        AtomKind::Thruster => Cat::Thruster,
        AtomKind::Eater => Cat::Eater,
        AtomKind::Sensor => Cat::Sensor,
        AtomKind::Food => Cat::Conductor,
    }
}

/// Preferred neighbour weights `[conductor, solar, gate, battery, thruster,
/// eater, sensor]` for an atom of the given parent category. This is the
/// "grammar" that biases growth toward sensible sequences.
fn pref_weights(parent: Cat) -> [f32; 7] {
    match parent {
        //          cond solar gate batt thr  eat  sens
        Cat::Conductor => [2.0, 1.5, 3.0, 1.5, 2.0, 0.8, 1.0],
        Cat::Solar => [3.0, 1.0, 2.0, 2.5, 0.5, 0.3, 0.5],
        Cat::Gate => [2.0, 0.8, 2.0, 2.0, 3.0, 1.0, 1.0],
        Cat::Battery => [2.0, 1.5, 2.0, 0.5, 3.0, 0.5, 0.5],
        Cat::Thruster => [3.0, 2.5, 1.5, 1.5, 0.5, 0.3, 0.5],
        Cat::Eater => [3.0, 1.5, 2.5, 1.0, 0.5, 0.2, 1.0],
        Cat::Sensor => [2.5, 1.0, 3.5, 1.0, 1.0, 0.5, 0.5],
    }
}

impl Genome {
    /// Map of occupied cells -> gene index.
    fn occupied(&self) -> FxHashMap<Axial, usize> {
        self.genes.iter().enumerate().map(|(i, g)| (g.coord, i)).collect()
    }

    /// Pick a concrete `AtomKind` for a child of `parent`, biased by grammar.
    fn choose_kind(rng: &mut Rng, parent: AtomKind) -> AtomKind {
        let w = pref_weights(cat_of(parent));
        // Weights per PALETTE entry: gate category is split across its 4 gate
        // variants and conductor across its 2 entries so category odds hold.
        let palette = AtomKind::PALETTE;
        let mut weights = [0.0f32; AtomKind::PALETTE.len()];
        for (i, k) in palette.iter().enumerate() {
            weights[i] = match k {
                AtomKind::Conductor => w[0] / 2.0,
                AtomKind::Solar => w[1],
                AtomKind::Gate(_) => w[2] / 4.0,
                AtomKind::Battery => w[3],
                AtomKind::Thruster => w[4],
                AtomKind::Eater => w[5],
                AtomKind::Sensor => w[6],
                _ => 0.0,
            };
        }
        palette[rng.weighted(&weights)]
    }

    /// Sensible default facing for a freshly placed atom, given the direction
    /// `d` that points from the parent to the new cell.
    fn choose_facing(rng: &mut Rng, kind: AtomKind, d: Dir) -> Dir {
        match kind {
            // Actuators/organs interface with the world: point outward.
            AtomKind::Thruster | AtomKind::Sensor | AtomKind::Eater => d,
            // Logic/storage feed back into the existing circuit.
            AtomKind::Gate(_) | AtomKind::Battery => d.opposite(),
            _ => Dir::from_index(rng.below(6)),
        }
    }

    fn default_param(rng: &mut Rng, kind: AtomKind) -> f32 {
        match kind {
            AtomKind::Battery => rng.range(0.0, 0.5),
            _ => 0.0,
        }
    }

    /// Grow a random genome of `min..=max` atoms around a single seed.
    pub fn random(rng: &mut Rng, min: usize, max: usize) -> Genome {
        let mut g = Genome {
            genes: vec![Gene {
                coord: Axial::ORIGIN,
                kind: AtomKind::Seed,
                facing: Dir::from_index(rng.below(6)),
                param: 0.0,
            }],
        };
        let target = min + rng.below((max - min).max(1) + 1);
        while g.genes.len() < target {
            if !g.grow_one(rng) {
                break;
            }
        }
        g
    }

    pub fn rocket(rng: &mut Rng, min: usize, max: usize) -> Genome {
        Genome {
            genes: vec![
                Gene {
                    coord: Axial::ORIGIN,
                    kind: AtomKind::Seed,
                    facing: Dir::from_index(rng.below(6)),
                    param: 0.0,
                },
                Gene {
                    coord: Axial::new(0, 1),
                    kind: AtomKind::Thruster,
                    facing: Dir::from_index(rng.below(6)),
                    param: 0.0,
                },
                Gene {
                    coord: Axial::new(0, 2),
                    kind: AtomKind::Eater,
                    facing: Dir::from_index(rng.below(6)),
                    param: 0.0,
                }
            ],
        }
    }

    /// Attempt to attach one new atom to a free neighbouring cell. Returns
    /// false if the structure has no exposed edge (should not happen in
    /// practice, but keeps growth loops safe).
    pub fn grow_one(&mut self, rng: &mut Rng) -> bool {
        let occ = self.occupied();
        // Collect (parent_index, direction, empty cell) candidates.
        let mut cands: Vec<(usize, Dir, Axial)> = Vec::new();
        for (i, gene) in self.genes.iter().enumerate() {
            for d in Dir::ALL {
                let c = gene.coord.step(d);
                if !occ.contains_key(&c) {
                    cands.push((i, d, c));
                }
            }
        }
        if cands.is_empty() {
            return false;
        }
        let (pi, d, cell) = cands[rng.below(cands.len())];
        let parent_kind = self.genes[pi].kind;
        let kind = Self::choose_kind(rng, parent_kind);
        let facing = Self::choose_facing(rng, kind, d);
        let param = Self::default_param(rng, kind);
        self.genes.push(Gene {
            coord: cell,
            kind,
            facing,
            param,
        });
        true
    }

    /// Apply reproduction-time mutations. Each operator fires independently.
    pub fn mutate(&mut self, rng: &mut Rng) {
        if rng.chance(MUT_POINT) {
            // Re-type a random non-seed atom, biased by its current neighbour.
            let n = self.genes.len();
            if n > 1 {
                let idx = 1 + rng.below(n - 1);
                // Pick a bias parent from an actual neighbour if one exists.
                let here = self.genes[idx].coord;
                let occ = self.occupied();
                let mut parent = AtomKind::Conductor;
                for d in Dir::ALL {
                    if let Some(&j) = occ.get(&here.step(d)) {
                        parent = self.genes[j].kind;
                        break;
                    }
                }
                let mut k = Self::choose_kind(rng, parent);
                // Occasionally just flip a gate's operator in place.
                if let AtomKind::Gate(_) = self.genes[idx].kind {
                    if rng.chance(0.5) {
                        k = AtomKind::Gate(GateOp::ALL[rng.below(6)]);
                    }
                }
                self.genes[idx].kind = k;
            }
        }
        if rng.chance(MUT_ROTATE) {
            let idx = rng.below(self.genes.len());
            self.genes[idx].facing = Dir::from_index(rng.below(6));
        }
        if rng.chance(MUT_PARAM) {
            let idx = rng.below(self.genes.len());
            let p = &mut self.genes[idx].param;
            *p = (*p + rng.range(-0.3, 0.3)).clamp(0.0, 1.0);
        }
        if rng.chance(MUT_ADD) && self.genes.len() < MAX_ATOMS {
            self.grow_one(rng);
        }
        if rng.chance(MUT_REMOVE) && self.genes.len() > 2 {
            let idx = 1 + rng.below(self.genes.len() - 1);
            self.genes.remove(idx);
            self.prune_disconnected();
        }
    }

    /// Keep only atoms reachable from the seed by bonds; drop islands created
    /// by a removal so the molecule stays a single construct.
    pub fn prune_disconnected(&mut self) {
        if self.genes.is_empty() {
            return;
        }
        let occ = self.occupied();
        let mut keep: FxHashSet<usize> = FxHashSet::default();
        let mut queue: VecDeque<usize> = VecDeque::new();
        keep.insert(0);
        queue.push_back(0);
        while let Some(i) = queue.pop_front() {
            let c = self.genes[i].coord;
            for d in Dir::ALL {
                if let Some(&j) = occ.get(&c.step(d)) {
                    if keep.insert(j) {
                        queue.push_back(j);
                    }
                }
            }
        }
        if keep.len() != self.genes.len() {
            let mut kept: Vec<Gene> = Vec::with_capacity(keep.len());
            for (i, g) in self.genes.iter().enumerate() {
                if keep.contains(&i) {
                    kept.push(*g);
                }
            }
            self.genes = kept;
        }
    }
}
