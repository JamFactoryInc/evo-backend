//! The ecosystem: all molecules and free food, the per-tick spatial
//! interactions between them (sensing, eating), and the population dynamics
//! (reproduction, death, ambient food) that close the evolutionary loop.
//!
//! No explicit fitness function exists. Molecules that happen to gather energy
//! faster than they spend it cross the reproduction threshold and copy their
//! (mutated) genome; molecules that run dry dissolve into food. Selection is an
//! emergent consequence of the energy economy.

use std::collections::{HashMap, HashSet};
use std::mem;
use fxhash::{FxHashMap, FxHashSet};
use godot::prelude::*;
use kdtree::distance::squared_euclidean;
use kdtree::KdTree;
use crate::atom::AtomKind;
use crate::config::*;
use crate::genome::Genome;
use crate::molecule::Molecule;
use crate::rng::Rng;

pub struct World {
    pub molecules: Vec<Molecule>,
    pub rng: Rng,
    pub alive_count: usize,
    pub size: f32,
    pub size_inv: f32,
    pub time: f32,
    hue_cursor: f32,
    /// Spatial hash: cell -> (molecule index, atom index). Reused each step.
    grid: FxHashMap<(i32, i32), Vec<(u32, u32)>>,
    sensor_sets: Vec<(usize, usize, f32)>,
    eat_intents: Vec<(usize, usize, usize, usize, f32)>,
    
    // quad_tree: KdTree<f32, (u32, u32), [f32; 2]>
}

impl World {
    pub fn new(seed: u64, size: f32) -> World {
        // let a: ([f32; 2], (i32, i32)) = ([0f32, 0f32], (0, 0));
        // let mut x = KdTree::new(2);
        // x.add([0f32, 0f32], (0i32, 0i32));

        World {
            molecules: Vec::new(),
            rng: Rng::new(seed),
            alive_count: 0,
            size,
            size_inv: 1.0 / size,
            time: 0.0,
            hue_cursor: 0.0,
            grid: FxHashMap::default(),
            
            // quad_tree: KdTree::new(2),
            sensor_sets: vec![],
            eat_intents: vec![],
        }
    }

    pub fn clear(&mut self) {
        self.molecules.clear();
        self.time = 0.0;
        self.alive_count = 0;
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
            self.alive_count += 1;
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
        ((p.x * self.size_inv).floor() as i32, (p.y * self.size_inv).floor() as i32)
    }

    fn rebuild_grid(&mut self) {
        // self.quad_tree = KdTree::with_capacity(
        //     2,
        //     16
        // );
        self.grid.clear();
        for (_, m) in self.molecules.iter_mut().enumerate() {
            let sin_cos = m.theta.sin_cos();
            m.sin_cos = sin_cos;
        }
        for (mi, m) in self.molecules.iter().enumerate() {
            for ai in 0..m.len() {
                let coord = m.atom_world(ai);
                let c = self.cell_of(coord);
                // self.quad_tree.add([coord.x, coord.y], (mi as u32, ai as u32)).expect("TODO: panic message");
                self.grid.entry(c).or_default().push((mi as u32, ai as u32));
            }
        }
    }

    /// Visit every atom whose cell lies within `cell_radius` of `center`,
    /// calling `f(molecule_index, atom_index)`.
    #[inline(never)]
    fn for_atoms_near<F: FnMut(usize, usize)>(&self, center: Vector2, cell_radius: i32, mut f: F) {

        let (cx, cy) = self.cell_of(center);
        // let area = AreaBuilder::default()
        //     .anchor(Point {x: cx - cell_radius, y: cy - cell_radius})
        //     .dimensions((cell_radius * 2, cell_radius * 2))
        //     .build().unwrap();
        // let point = &[center.x, center.y];
        // let candidates = self.quad_tree.iter_nearest_within_radius(
        //     point, Some(128.0), &squared_euclidean
        // ).unwrap();
        // for (_, (mi, ai)) in candidates {
        //     f(*mi as usize, *ai as usize)
        // }

        if let Some(bucket) = self.grid.get(&(cx, cy)) {
            for &(mi, ai) in bucket {
                f(mi as usize, ai as usize);
            }
        }

        // for gx in (cx - cell_radius)..=(cx + cell_radius) {
        //     for gy in (cy - cell_radius)..=(cy + cell_radius) {
        //         if let Some(bucket) = self.grid.get(&(gx, gy)) {
        //             for &(mi, ai) in bucket {
        //                 f(mi as usize, ai as usize);
        //             }
        //         }
        //     }
        // }
    }

    #[inline(never)]
    fn process_molecule_atoms(
        &mut self,
        sensor_range: f32,
        sensor_cellr: i32,
        eat_range: f32
    ) {
        let mut sensor_sets = vec![];
        let mut eat_intents = vec![];
        mem::swap(&mut self.sensor_sets, &mut sensor_sets);
        mem::swap(&mut self.eat_intents, &mut eat_intents);

        for (mi, m) in self.molecules.iter().enumerate() {
            if m.is_food {
                continue;
            }
            for ai in 0..m.len() {
                self.process_atoms(
                    m,
                    &mut sensor_sets,
                    &mut eat_intents,
                    mi,
                    ai,
                    sensor_range,
                    sensor_cellr,
                    eat_range
                );
            }
        }

        mem::swap(&mut self.sensor_sets, &mut sensor_sets);
        mem::swap(&mut self.eat_intents, &mut eat_intents);
    }

    #[inline(never)]
    fn process_atoms(
        &self, m: &Molecule,
        sensor_sets: &mut Vec<(usize, usize, f32)>,
        eat_intents: &mut Vec<(usize, usize, usize, usize, f32)>,
        mi: usize,
        ai: usize,
        sensor_range: f32,
        sensor_cellr: i32,
        eat_range: f32
    ) {
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

        self.sensor_sets.clear();
        self.eat_intents.clear();
        // let mut sensor_sets: Vec<(usize, usize, f32)> = Vec::new();
        // // (eater_mol, eater_atom, victim_mol, victim_atom, dist2)
        // let mut eat_intents: Vec<(usize, usize, usize, usize, f32)> = Vec::new();

        self.process_molecule_atoms(
            sensor_range,
            sensor_cellr,
            eat_range,
        );

        // Apply sensor readings.
        for (mi, ai, r) in &self.sensor_sets {
            self.molecules[*mi].sensor[*ai] = *r;
        }

        // Resolve eats: closest bite wins each victim; a victim is consumed once.
        self.eat_intents.sort_by(|a, b| a.4.partial_cmp(&b.4).unwrap_or(std::cmp::Ordering::Equal));
        let mut claimed: FxHashSet<(usize, usize)> = FxHashSet::default();
        for (em, ea, vm, va, _) in &self.eat_intents {
            let em = *em;
            let ea = *ea;
            let vm = *vm;
            let va = *va;
            // if (!self.molecules[vm].is_food) {
            //     continue;
            // }
            if claimed.contains(&(vm, va)) || self.molecules[em].vel.length_squared() < 10.0 {
                continue;
            }
            claimed.insert((vm, va));
            self.molecules[em].ate[ea] = true;
            self.molecules[em].pool += FOOD_VALUE;

            let mass = self.molecules[em].mass;
            let vel = self.molecules[em].vel;
            self.molecules[em].vel -= vel / mass;
        }

        // --- Tick every molecule (circuit + metabolism + motion) -------------
        for m in self.molecules.iter_mut() {
            m.tick(dt);
        }

        // --- Structural changes: consumed atoms, deaths, births --------------
        // Group claimed victims per molecule.
        let mut victims: FxHashMap<usize, FxHashSet<usize>> = FxHashMap::default();
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

                self.alive_count += 1;
                newborns.push(child);
            }

            survivors.push(m);
        }

        // self.alive_count = survivors.len() + newborns.len();
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
    fn rebuild_after_removal(&self, m: &Molecule, victims: &FxHashSet<usize>) -> Option<Molecule> {
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
        self.alive_count -= 1;
        for i in 0..m.len() {
            if out.len() + self.food_count() >= MAX_FOOD {
                break;
            }
            let drift = Vector2::new(self.rng.range(-14.0, 14.0), self.rng.range(-14.0, 14.0));
            out.push(Molecule::food(m.atom_world(i), m.vel + drift, self.size));
        }
    }
}


#[test]
fn bench() {
    let mut world = World {
        molecules: Vec::new(),
        rng: Rng::new(0),
        alive_count: 0,
        size: 14.0,
        size_inv: 1.0 / 14.0,
        time: 0.0,
        hue_cursor: 0.0,
        grid: FxHashMap::default(),
        
        // quad_tree: KdTree::new(2),
        sensor_sets: vec![],
        eat_intents: vec![],
    };

    world.spawn_random_population(1000);

    for _ in 0..1000 {
        world.step(0.001)
    }

}