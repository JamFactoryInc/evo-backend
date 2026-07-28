//! The ecosystem: all molecules and free food, the per-tick spatial
//! interactions between them (sensing, eating), and the population dynamics
//! (reproduction, death, ambient food) that close the evolutionary loop.
//!
//! No explicit fitness function exists. Molecules that happen to gather energy
//! faster than they spend it cross the reproduction threshold and copy their
//! (mutated) genome; molecules that run dry dissolve into food. Selection is an
//! emergent consequence of the energy economy.

use std::collections::{HashMap, HashSet};

use godot::prelude::*;

use crate::atom::AtomKind;
use crate::config::*;
use crate::genome::Genome;
use crate::molecule::Molecule;
use crate::rng::Rng;

pub struct World {
    pub molecules: Vec<Molecule>,
    pub rng: Rng,
    pub size: f32,
    pub time: f32,
    hue_cursor: f32,
    /// Spatial hash: cell -> (molecule index, atom index). Reused each step.
    grid: HashMap<(i32, i32), Vec<(u32, u32)>>,
}

impl World {
    pub fn new(seed: u64, size: f32) -> World {
        World {
            molecules: Vec::new(),
            rng: Rng::new(seed),
            size,
            time: 0.0,
            hue_cursor: 0.0,
            grid: HashMap::new(),
        }
    }

    pub fn clear(&mut self) {
        self.molecules.clear();
        self.time = 0.0;
    }

    fn next_hue(&mut self) -> f32 {
        // Golden-ratio hop gives well-spread, distinct species tints.
        self.hue_cursor = (self.hue_cursor + 0.618_034) % 1.0;
        self.hue_cursor
    }

    pub fn living_count(&self) -> usize {
        self.molecules.iter().filter(|m| !m.is_food).count()
    }
    pub fn food_count(&self) -> usize {
        self.molecules.iter().filter(|m| m.is_food).count()
    }
    pub fn total_atoms(&self) -> usize {
        self.molecules.iter().map(|m| m.len()).sum()
    }

    /// Seed the world with `count` random molecules and a starter food supply.
    pub fn spawn_random_population(&mut self, count: usize) {
        for _ in 0..count {
            let a = self.rng.range(0.0, std::f32::consts::TAU);
            let rad = self.rng.range(0.0, WORLD_RADIUS * 0.8);
            let pos = Vector2::new(a.cos(), a.sin()) * rad;
            let genome = Genome::random(&mut self.rng, SEED_MIN_ATOMS, SEED_MAX_ATOMS);
            let theta = self.rng.range(0.0, std::f32::consts::TAU);
            let hue = self.next_hue();
            let m = Molecule::from_genome(genome, pos, theta, START_POOL, hue, self.size);
            self.molecules.push(m);
        }
        for _ in 0..AMBIENT_FOOD_TARGET {
            self.spawn_one_food();
        }
    }

    fn spawn_one_food(&mut self) {
        if self.food_count() >= MAX_FOOD {
            return;
        }
        let a = self.rng.range(0.0, std::f32::consts::TAU);
        let rad = self.rng.range(0.0, WORLD_RADIUS);
        let pos = Vector2::new(a.cos(), a.sin()) * rad;
        let drift = Vector2::new(self.rng.range(-8.0, 8.0), self.rng.range(-8.0, 8.0));
        self.molecules.push(Molecule::food(pos, drift, self.size));
    }

    #[inline]
    fn cell_of(&self, p: Vector2) -> (i32, i32) {
        ((p.x / self.size).floor() as i32, (p.y / self.size).floor() as i32)
    }

    fn rebuild_grid(&mut self) {
        self.grid.clear();
        for (mi, m) in self.molecules.iter().enumerate() {
            for ai in 0..m.len() {
                let c = self.cell_of(m.atom_world(ai));
                self.grid.entry(c).or_default().push((mi as u32, ai as u32));
            }
        }
    }

    /// Visit every atom whose cell lies within `cell_radius` of `center`,
    /// calling `f(molecule_index, atom_index)`.
    fn for_atoms_near<F: FnMut(usize, usize)>(&self, center: Vector2, cell_radius: i32, mut f: F) {
        let (cx, cy) = self.cell_of(center);
        for gx in (cx - cell_radius)..=(cx + cell_radius) {
            for gy in (cy - cell_radius)..=(cy + cell_radius) {
                if let Some(bucket) = self.grid.get(&(gx, gy)) {
                    for &(mi, ai) in bucket {
                        f(mi as usize, ai as usize);
                    }
                }
            }
        }
    }

    /// Advance the whole world by `dt`.
    pub fn step(&mut self, dt: f32) {
        self.time += dt;
        if self.molecules.is_empty() {
            return;
        }
        self.rebuild_grid();

        // --- Spatial scan: sensor readings + eat intents ---------------------
        let sensor_range = SENSOR_RANGE_HEX * self.size;
        let sensor_cellr = (sensor_range / self.size).ceil() as i32 + 1;
        let eat_range = EAT_RANGE_FRAC * self.size;

        let mut sensor_sets: Vec<(usize, usize, f32)> = Vec::new();
        // (eater_mol, eater_atom, victim_mol, victim_atom, dist2)
        let mut eat_intents: Vec<(usize, usize, usize, usize, f32)> = Vec::new();

        for (mi, m) in self.molecules.iter().enumerate() {
            if m.is_food {
                continue;
            }
            for ai in 0..m.len() {
                match m.kinds[ai] {
                    AtomKind::Sensor => {
                        let ahead = m.atom_world(ai) + m.dir_world(m.facing[ai]) * (sensor_range * 0.5);
                        let mut count = 0.0f32;
                        let r2 = (sensor_range * 0.6) * (sensor_range * 0.6);
                        self.for_atoms_near(ahead, sensor_cellr, |omi, oai| {
                            if omi == mi {
                                return;
                            }
                            let d2 = (self.molecules[omi].atom_world(oai) - ahead).length_squared();
                            if d2 <= r2 {
                                count += 1.0;
                            }
                        });
                        let reading = (count / SENSOR_SATURATION).min(1.0);
                        if reading > 0.0 {
                            sensor_sets.push((mi, ai, reading));
                        }
                    }
                    AtomKind::Eater => {
                        let mouth = m.atom_world(ai) + m.dir_world(m.facing[ai]) * (self.size * 0.5);
                        let er2 = eat_range * eat_range;
                        let mut best: Option<(usize, usize, f32)> = None;
                        self.for_atoms_near(mouth, 2, |omi, oai| {
                            if omi == mi {
                                return;
                            }
                            let d2 = (self.molecules[omi].atom_world(oai) - mouth).length_squared();
                            if d2 <= er2 && best.map_or(true, |(_, _, bd)| d2 < bd) {
                                best = Some((omi, oai, d2));
                            }
                        });
                        if let Some((omi, oai, d2)) = best {
                            eat_intents.push((mi, ai, omi, oai, d2));
                        }
                    }
                    _ => {}
                }
            }
        }

        // Apply sensor readings.
        for (mi, ai, r) in sensor_sets {
            self.molecules[mi].sensor[ai] = r;
        }

        // Resolve eats: closest bite wins each victim; a victim is consumed once.
        eat_intents.sort_by(|a, b| a.4.partial_cmp(&b.4).unwrap_or(std::cmp::Ordering::Equal));
        let mut claimed: HashSet<(usize, usize)> = HashSet::new();
        for (em, ea, vm, va, _) in eat_intents {
            if claimed.contains(&(vm, va)) {
                continue;
            }
            claimed.insert((vm, va));
            self.molecules[em].ate[ea] = true;
            self.molecules[em].pool += FOOD_VALUE;
        }

        // --- Tick every molecule (circuit + metabolism + motion) -------------
        for m in self.molecules.iter_mut() {
            m.tick(dt);
        }

        // --- Structural changes: consumed atoms, deaths, births --------------
        // Group claimed victims per molecule.
        let mut victims: HashMap<usize, HashSet<usize>> = HashMap::new();
        for (vm, va) in claimed {
            victims.entry(vm).or_default().insert(va);
        }

        let old = std::mem::take(&mut self.molecules);
        let mut survivors: Vec<Molecule> = Vec::with_capacity(old.len());
        let mut newborns: Vec<Molecule> = Vec::new();
        let mut new_food: Vec<Molecule> = Vec::new();

        for (mi, mut m) in old.into_iter().enumerate() {
            let vic = victims.get(&mi);

            if m.is_food {
                // Food is consumed if its single atom was claimed.
                if vic.map_or(true, |v| !v.contains(&0)) {
                    survivors.push(m);
                }
                continue;
            }

            // Bitten: seed loss is fatal, otherwise lose the eaten atoms.
            if let Some(v) = vic {
                if v.contains(&0) {
                    self.dissolve(&m, &mut new_food);
                    continue;
                }
                match self.rebuild_after_removal(&m, v) {
                    Some(rebuilt) => m = rebuilt,
                    None => {
                        self.dissolve(&m, &mut new_food);
                        continue;
                    }
                }
            }

            // Starvation death.
            if m.pool <= 0.0 {
                self.dissolve(&m, &mut new_food);
                continue;
            }

            // Reproduction.
            let population = survivors.iter().filter(|s| !s.is_food).count() + newborns.len();
            if m.pool > REPRO_THRESHOLD && population < MAX_MOLECULES {
                let child = self.reproduce(&m);
                m.pool *= 1.0 - REPRO_CHILD_FRAC;
                newborns.push(child);
            }

            survivors.push(m);
        }

        self.molecules = survivors;
        self.molecules.append(&mut newborns);
        for f in new_food {
            if self.food_count() < MAX_FOOD {
                self.molecules.push(f);
            }
        }

        // --- Ambient food rain -----------------------------------------------
        if self.food_count() < AMBIENT_FOOD_TARGET {
            for _ in 0..FOOD_SPAWN_PER_STEP {
                self.spawn_one_food();
            }
        }
    }

    /// Copy a molecule's genome, mutate it, and place a child nearby.
    fn reproduce(&mut self, parent: &Molecule) -> Molecule {
        let mut genome = parent.genome.clone();
        genome.mutate(&mut self.rng);
        genome.prune_disconnected();
        let a = self.rng.range(0.0, std::f32::consts::TAU);
        let offset = Vector2::new(a.cos(), a.sin()) * (self.size * (parent.len() as f32).sqrt() * 2.0);
        let hue = (parent.hue + self.rng.range(-0.04, 0.04)).rem_euclid(1.0);
        let mut child = Molecule::from_genome(
            genome,
            parent.pos + offset,
            self.rng.range(0.0, std::f32::consts::TAU),
            parent.pool * REPRO_CHILD_FRAC,
            hue,
            self.size,
        );
        child.vel = parent.vel;
        child
    }

    /// Rebuild a molecule after some atoms were eaten. Returns None if nothing
    /// (or only the seed) survives.
    fn rebuild_after_removal(&self, m: &Molecule, victims: &HashSet<usize>) -> Option<Molecule> {
        let mut genome = m.genome.clone();
        // Remove from the back so indices stay valid.
        let mut idxs: Vec<usize> = victims.iter().copied().collect();
        idxs.sort_unstable_by(|a, b| b.cmp(a));
        for i in idxs {
            if i < genome.genes.len() {
                genome.genes.remove(i);
            }
        }
        genome.prune_disconnected();
        if genome.genes.len() < 2 {
            return None;
        }
        let mut rebuilt =
            Molecule::from_genome(genome, m.pos, m.theta, m.pool, m.hue, self.size);
        rebuilt.vel = m.vel;
        rebuilt.omega = m.omega;
        Some(rebuilt)
    }

    /// Scatter a dead molecule's atoms into free food.
    fn dissolve(&mut self, m: &Molecule, out: &mut Vec<Molecule>) {
        for i in 0..m.len() {
            if out.len() + self.food_count() >= MAX_FOOD {
                break;
            }
            let drift = Vector2::new(self.rng.range(-14.0, 14.0), self.rng.range(-14.0, 14.0));
            out.push(Molecule::food(m.atom_world(i), m.vel + drift, self.size));
        }
    }
}
