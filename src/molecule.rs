//! A molecule: one living construct. It couples two layers that advance in
//! lockstep each tick:
//!
//! * **Signal layer** — energy on bond sites, evaluated as a synchronous
//!   cellular automaton (all atoms read the previous tick and write the next),
//!   which is the circuit/nervous system.
//! * **Metabolic layer** — a single `pool` reserve that solar/eating fill and
//!   logic/thrust drain. Run out and the molecule dies. This scalar is the
//!   selection pressure that makes the genetic algorithm run without an
//!   explicit fitness function.
//!
//! The molecule is also a rigid body: thruster atoms inject force (and torque,
//! since they act off-centre) and it drifts through 2D space.

use std::cmp::Ordering;
use godot::prelude::*;

use crate::atom::AtomKind;
use crate::config::*;
use crate::genome::Genome;
use crate::hex_grid::{Axial, Dir};

pub struct Molecule {
    pub genome: Genome,

    // --- topology cache (derived from genome, rebuilt on structural change) ---
    pub coords: Vec<Axial>,
    pub kinds: Vec<AtomKind>,
    pub facing: Vec<Dir>,
    /// `neighbor[i][d]` = index of the atom bonded on direction `d`, or -1.
    pub neighbor: Vec<[i32; 6]>,
    /// Unrotated offset of each atom from the centre of mass (world units).
    pub local_off: Vec<Vector2>,

    // --- signal layer (double buffered via a scratch swap in `tick`) ---
    pub site: Vec<[f32; 6]>,
    pub batt: Vec<f32>,
    /// Set by the world before `tick`: eater fed on the previous scan.
    pub ate: Vec<bool>,
    /// Set by the world before `tick`: sensor reading in [0, 1].
    pub sensor: Vec<f32>,

    // --- metabolic + rigid-body state ---
    pub pool: f32,
    pub pos: Vector2,
    pub vel: Vector2,
    pub sin_cos: (f32, f32),
    pub theta: f32,
    pub omega: f32,
    pub mass: f32,
    pub inv_inertia: f32,

    pub age: f32,
    pub alive: bool,
    pub is_food: bool,
    /// Species tint (0..1 hue-ish) inherited by children, for rendering.
    pub hue: f32,
    pub size: f32,
    pub max_radius: f32,
}

fn rotate_2d(sin_cos: (f32, f32), vector2: Vector2) -> Vector2 {
    let (sin, cos) = sin_cos;
    Vector2::new(
        cos * vector2.x - sin * vector2.y,
        sin * vector2.x + cos * vector2.y,
    )
}

impl Molecule {
    /// Build a living molecule from a blueprint at a world pose.
    pub fn from_genome(
        genome: Genome,
        pos: Vector2,
        theta: f32,
        pool: f32,
        hue: f32,
        size: f32,
    ) -> Molecule {
        let n = genome.genes.len();
        let coords: Vec<Axial> = genome.genes.iter().map(|g| g.coord).collect();
        let kinds: Vec<AtomKind> = genome.genes.iter().map(|g| g.kind).collect();
        let facing: Vec<Dir> = genome.genes.iter().map(|g| g.facing).collect();
        let batt: Vec<f32> = genome
            .genes
            .iter()
            .map(|g| if matches!(g.kind, AtomKind::Battery) { g.param * BATT_CAP } else { 0.0 })
            .collect();

        // Neighbour table.
        let index: std::collections::HashMap<Axial, usize> =
            coords.iter().enumerate().map(|(i, c)| (*c, i)).collect();
        let mut neighbor = vec![[-1i32; 6]; n];
        for (i, c) in coords.iter().enumerate() {
            for d in Dir::ALL {
                if let Some(&j) = index.get(&c.step(d)) {
                    neighbor[i][d.index()] = j as i32;
                }
            }
        }

        // Centre of mass and moment of inertia (unit mass per atom).
        let raw: Vec<Vector2> = coords
            .iter()
            .map(|c| {
                let (x, y) = c.to_world(size);
                Vector2::new(x, y)
            })
            .collect();
        let mut centroid = Vector2::ZERO;
        for r in &raw {
            centroid += *r;
        }
        centroid /= n.max(1) as f32;
        let local_off: Vec<Vector2> = raw.iter().map(|r| *r - centroid).collect();
        let inertia: f32 = local_off.iter().map(|o| o.length_squared()).sum::<f32>().max(1e-3);
        let mass = n as f32;

        Molecule {
            genome,
            coords,
            kinds,
            facing,
            neighbor,
            local_off,
            site: vec![[0.0; 6]; n],
            batt,
            ate: vec![false; n],
            sensor: vec![0.0; n],
            pool,
            pos,
            vel: Vector2::ZERO,
            theta,
            sin_cos: theta.sin_cos(),
            omega: 0.0,
            mass,
            inv_inertia: 1.0 / inertia,
            age: 0.0,
            alive: true,
            is_food: false,
            hue,
            size,
            max_radius: 0.0f32,
        }
    }

    pub fn calculate_max_radius(&self) -> f32 {
        self.local_off.iter()
            .map(|v| v.length_squared())
            .max_by(|l, r| l.partial_cmp(r).unwrap_or(Ordering::Equal))
            .unwrap_or(0.0)
    }

    /// A free-floating single-atom nutrient.
    #[inline(never)]
    pub fn food(pos: Vector2, vel: Vector2, size: f32) -> Molecule {
        let genome = Genome {
            genes: vec![crate::genome::Gene {
                coord: Axial::ORIGIN,
                kind: AtomKind::Food,
                facing: Dir::East,
                param: 0.0,
            }],
        };
        let mut m = Molecule::from_genome(genome, pos, 0.0, FOOD_POOL, 0.08, size);
        m.vel = vel;
        m.is_food = true;
        m
    }

    pub fn len(&self) -> usize {
        self.coords.len()
    }
    pub fn is_empty(&self) -> bool {
        self.coords.is_empty()
    }

    /// World-space centre of atom `i`, accounting for the body's rotation.
    #[inline]
    pub fn atom_world(&self, i: usize) -> Vector2 {
        self.pos + rotate_2d(self.sin_cos, self.local_off[i])
    }

    /// World-space unit vector of a bond direction on this (rotated) body.
    #[inline]
    pub fn dir_world(&self, d: Dir) -> Vector2 {
        let (x, y) = d.unit();
        rotate_2d(self.sin_cos, Vector2::new(x, y))
    }

    /// Total signal energy on an atom's sites — used for render brightness.
    #[inline]
    pub fn activity(&self, i: usize) -> f32 {
        self.site[i].iter().sum()
    }

    #[inline]
    fn bonded(&self, i: usize, d: usize) -> bool {
        self.neighbor[i][d] >= 0
    }

    /// Deliver `e` from atom `i` out of direction `d` into the facing site of
    /// its neighbour's next-tick buffer. Energy on an unbonded site is lost.
    #[inline]
    fn emit(&self, next: &mut [[f32; 6]], i: usize, d: usize, e: f32) {
        let j = self.neighbor[i][d];
        if j >= 0 {
            let opp = (d + 3) % 6;
            next[j as usize][opp] += e;
        }
    }

    /// Advance one simulation tick: evaluate the circuit, apply metabolism, and
    /// integrate rigid-body motion. `sensor`/`ate` must be set by the world
    /// beforehand; pool credit from eating is added directly by the world.
    pub fn tick(&mut self, dt: f32) {
        let n = self.len();
        if self.is_food || n == 0 {
            self.integrate(Vector2::ZERO, 0.0, dt);
            return;
        }

        let mut next = vec![[0.0f32; 6]; n];
        let mut force = Vector2::ZERO;
        let mut torque = 0.0f32;
        let mut dpool = -UPKEEP * n as f32; // metabolism

        for i in 0..n {
            let inp = self.site[i];
            match self.kinds[i] {
                AtomKind::Seed | AtomKind::Conductor => {
                    // Diffuse: push each bond the average of the *other* bonds
                    // so energy travels across rather than reflecting back.
                    let mut bl = [0usize; 6];
                    let mut nb = 0;
                    let mut total = 0.0;
                    for d in 0..6 {
                        if self.bonded(i, d) {
                            bl[nb] = d;
                            nb += 1;
                            total += inp[d];
                        }
                    }
                    if nb == 1 {
                        self.emit(&mut next, i, bl[0], inp[bl[0]] * (1.0 - WIRE_LOSS));
                    } else if nb > 1 {
                        let denom = (nb - 1) as f32;
                        for k in 0..nb {
                            let d = bl[k];
                            let share = (total - inp[d]) / denom * (1.0 - WIRE_LOSS);
                            self.emit(&mut next, i, d, share);
                        }
                    }
                }
                AtomKind::Solar => {
                    dpool += SOLAR_RATE;
                    for d in 0..6 {
                        if self.bonded(i, d) {
                            self.emit(&mut next, i, d, SOLAR_SIGNAL);
                        }
                    }
                }
                AtomKind::Gate(op) => {
                    let out_d = self.facing[i].index();
                    let mut high = 0u32;
                    let mut cnt = 0u32;
                    for d in 0..6 {
                        if d == out_d || !self.bonded(i, d) {
                            continue;
                        }
                        cnt += 1;
                        if inp[d] >= HI {
                            high += 1;
                        }
                    }
                    if cnt > 0 && op.eval(high, cnt) {
                        self.emit(&mut next, i, out_d, QUANTUM);
                        dpool -= GATE_COST;
                    }
                }
                AtomKind::Battery => {
                    let out_d = self.facing[i].index();
                    let ctrl_d = (out_d + 3) % 6;
                    // Charge from every other bonded site.
                    for d in 0..6 {
                        if d != out_d && d != ctrl_d && self.bonded(i, d) {
                            self.batt[i] = (self.batt[i] + inp[d] * (1.0 - BATT_LOSS)).min(BATT_CAP);
                        }
                    }
                    // Release while control line is held high.
                    if self.bonded(i, ctrl_d) && inp[ctrl_d] >= HI && self.batt[i] > 0.0 {
                        let r = self.batt[i].min(BATT_DRAIN);
                        self.batt[i] -= r;
                        self.emit(&mut next, i, out_d, r);
                    }
                }
                AtomKind::Thruster => {
                    let mut intake = 0.0;
                    for d in 0..6 {
                        if self.bonded(i, d) {
                            intake += inp[d];
                        }
                    }
                    if intake <= 0.0 {
                        continue
                    }
                    let fuel = intake * THRUST_FUEL;

                    if self.pool + dpool > fuel {
                        dpool -= fuel;
                        let f = self.dir_world(self.facing[i]) * (intake * THRUST_FORCE);
                        force += f;
                        let r = rotate_2d(self.sin_cos, self.local_off[i]);
                        torque += r.x * f.y - r.y * f.x;
                    }
                }
                AtomKind::Eater => {
                    // World credited pool already; radiate a bite pulse inward.
                    if !self.ate[i] {
                        continue
                    }
                    for d in 0..6 {
                        if self.bonded(i, d) {
                            self.emit(&mut next, i, d, EAT_PULSE);
                        }
                    }
                }
                AtomKind::Sensor => {
                    // Senses ahead (facing); injects the reading behind it.
                    let s = self.sensor[i];
                    if s > 0.0 {
                        let back = (self.facing[i].index() + 3) % 6;
                        self.emit(&mut next, i, back, s);
                    }
                }
                AtomKind::Food => {}
            }
        }

        self.pool += dpool;
        self.site = next;
        // Reset transient interaction inputs; the world refills them next scan.
        for a in self.ate.iter_mut() {
            *a = false;
        }
        for s in self.sensor.iter_mut() {
            *s = 0.0;
        }

        self.integrate(force, torque, dt);
    }

    fn integrate(&mut self, force: Vector2, torque: f32, dt: f32) {
        self.vel += force / self.mass * dt;
        self.omega += torque * self.inv_inertia * dt;

        // Soft boundary spring keeps everything on-screen.
        let d = self.pos.length();
        if d > WORLD_RADIUS && d > 0.0 {
            let pull = -self.pos / d * (BOUNDARY_PULL * (d - WORLD_RADIUS) / WORLD_RADIUS);
            self.vel += pull * dt;
        }

        // Viscous space.
        self.vel *= (1.0 - LINEAR_DRAG * dt).clamp(0.0, 1.0);
        self.omega *= (1.0 - ANGULAR_DRAG * dt).clamp(0.0, 1.0);

        self.pos += self.vel * dt;
        self.theta += self.omega * dt;
        self.age += dt;
    }
}
