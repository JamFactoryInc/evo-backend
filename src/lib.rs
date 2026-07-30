//! evo-2 — a hex-grid electrical-circuit life simulator.
//!
//! Molecules are organic circuits built from hexagonal atoms. Energy flows on
//! bond sites as a synchronous cellular automaton (the circuit/nervous system)
//! while a metabolic pool drives selection. Thrusters move bodies, eaters prey,
//! sensors perceive, and a genetic algorithm tunes the whole thing with no
//! explicit fitness function — survival and reproduction do the selecting.
//!
//! `EvoWorld` is the Godot-facing node; the simulation proper lives in the
//! sibling modules and is engine-agnostic apart from using Godot's `Vector2`.

mod atom;
mod config;
mod genome;
mod hex_grid;
mod molecule;
mod rng;
mod world;

use godot::classes::mesh::PrimitiveType;
use godot::classes::multi_mesh::TransformFormat;
use godot::classes::{ArrayMesh, INode2D, MultiMesh, MultiMeshInstance2D, Node2D, Time};
use godot::prelude::*;
use crate::atom::AtomKind;
use crate::world::World;

struct MyExtension;

#[gdextension]
unsafe impl ExtensionLibrary for MyExtension {}

/// The simulation world as a Godot node. Drop it into a scene, hit play, and it
/// grows, feeds, mutates and dies on its own. Pan/zoom the parent Camera2D to
/// explore the population.
#[derive(GodotClass)]
#[class(base=Node2D)]
struct EvoWorld {
    /// Number of molecules spawned on start / reset.
    #[export]
    population: i32,
    /// World units between adjacent atom centres (visual + physical scale).
    #[export]
    hex_size: f32,
    /// Simulation ticks evaluated per rendered frame (raises circuit speed).
    #[export]
    ticks_per_frame: i32,
    /// RNG seed; 0 picks a time-based seed for a fresh run each launch.
    #[export]
    seed: i64,
    /// Freeze the simulation (rendering continues).
    #[export]
    paused: bool,

    #[export]
    alive_count: i32,

    world: Option<World>,
    multimesh: Option<Gd<MultiMesh>>,
    capacity: i32,

    base: Base<Node2D>,
}

#[godot_api]
impl INode2D for EvoWorld {
    fn init(base: Base<Node2D>) -> Self {
        Self {
            population: 250,
            hex_size: 14.0,
            ticks_per_frame: 1,
            seed: 0,
            paused: false,
            alive_count: 0,
            world: None,
            multimesh: None,
            capacity: 0,
            base,
        }
    }

    fn ready(&mut self) {
        self.build_renderer();
        self.reset();
        godot_print!("[evo-2] world ready!");
    }

    fn physics_process(&mut self, delta: f64) {
        if !self.paused {
            let ticks = self.ticks_per_frame.max(1);
            // Keep real-time motion regardless of tick count; clamp the first
            // huge frame so nothing explodes.
            let dt = (delta as f32 / ticks as f32).min(1.0 / 30.0);
            if let Some(w) = self.world.as_mut() {
                for _ in 0..ticks {
                    w.step(dt);
                }
            }
            self.alive_count = self.world
                .as_ref()
                .map(|w| w.alive_count as i32)
                .unwrap_or(0);
            if self.alive_count == 0 {
                self.reset()
            }
        }
        self.render();
    }
}

#[godot_api]
impl EvoWorld {
    /// Clear and repopulate the world from the current exported settings.
    #[func]
    fn reset(&mut self) {
        let seed = if self.seed == 0 {
            Time::singleton().get_ticks_usec()
        } else {
            self.seed as u64
        };
        let mut w = World::new(seed, self.hex_size);
        w.spawn_random_population(self.population.max(0) as usize);
        self.world = Some(w);
    }

    /// Inject `count` extra random molecules at any time.
    #[func]
    fn spawn(&mut self, count: i32) {
        if let Some(w) = self.world.as_mut() {
            w.spawn_random_population(count.max(0) as usize);
        }
    }

    #[func]
    fn living_count(&self) -> i64 {
        self.world.as_ref().map_or(0, |w| w.living_count() as i64)
    }
    #[func]
    fn food_count(&self) -> i64 {
        self.world.as_ref().map_or(0, |w| w.food_count() as i64)
    }
    #[func]
    fn atom_count(&self) -> i64 {
        self.world.as_ref().map_or(0, |w| w.total_atoms() as i64)
    }
    #[func]
    fn sim_time(&self) -> f64 {
        self.world.as_ref().map_or(0.0, |w| w.time as f64)
    }

    /// Create the MultiMesh + instance node and attach it as a child.
    fn build_renderer(&mut self) {
        let mesh = build_hex_mesh(self.hex_size * 0.56);

        let mut mm = MultiMesh::new_gd();
        mm.set_transform_format(TransformFormat::TRANSFORM_2D);
        mm.set_use_colors(true);
        mm.set_mesh(&mesh);
        mm.set_instance_count(0);

        let mut mmi = MultiMeshInstance2D::new_alloc();
        mmi.set_multimesh(&mm);
        self.base_mut().add_child(&mmi);

        self.multimesh = Some(mm);
        self.capacity = 0;
    }

    /// Push every atom's transform + colour into the MultiMesh buffer.
    fn render(&mut self) {
        let (Some(w), Some(mm)) = (self.world.as_ref(), self.multimesh.as_mut()) else {
            return;
        };

        let total = w.total_atoms() as i32;
        if total > self.capacity {
            // Grow with headroom so we reallocate rarely.
            self.capacity = total + total / 2 + 64;
            mm.set_instance_count(self.capacity);
        }
        mm.set_visible_instance_count(total);

        let mut idx: i32 = 0;
        for m in &w.molecules {
            for i in 0..m.len() {
                let p = m.atom_world(i);
                mm.set_instance_transform_2d(idx, Transform2D::from_angle_origin(m.theta, p));

                let (r, g, b) = m.kinds[i].color();
                // Brighten with signal activity; dim when the cell is starving.
                let act = m.activity(i).clamp(0.0, 2.0);
                let mut bright = 0.45 + 0.55 * (act * 0.5);
                if !m.is_food {
                    let vitality = (m.pool / config::START_POOL).clamp(0.25, 1.0);
                    bright *= vitality;
                }
                let col = Color::from_rgba(
                    (r * bright).min(1.0),
                    (g * bright).min(1.0),
                    (b * bright).min(1.0),
                    1.0,
                );
                mm.set_instance_color(idx, col);
                idx += 1;
            }
        }
    }
}

/// Build a flat, pointy-top hexagon (a 6-triangle fan) as an `ArrayMesh`.
fn build_hex_mesh(radius: f32) -> Gd<ArrayMesh> {
    use std::f32::consts::{FRAC_PI_2, FRAC_PI_3};

    let mut ring = [Vector2::ZERO; 6];
    for (k, v) in ring.iter_mut().enumerate() {
        let a = FRAC_PI_2 + k as f32 * FRAC_PI_3;
        *v = Vector2::new(a.cos() * radius, a.sin() * radius);
    }

    let mut verts: Vec<Vector3> = Vec::with_capacity(18);
    for k in 0..6 {
        let n = (k + 1) % 6;
        verts.push(Vector3::new(0.0, 0.0, 0.0));
        verts.push(Vector3::new(ring[k].x, ring[k].y, 0.0));
        verts.push(Vector3::new(ring[n].x, ring[n].y, 0.0));
    }

    // Surface array: index ARRAY_VERTEX (0) of a length-ARRAY_MAX (13) array.
    let mut arrays = VarArray::new();
    arrays.resize(13, &Variant::nil());
    arrays.set(0, &PackedVector3Array::from(verts).to_variant());

    let mut mesh = ArrayMesh::new_gd();
    mesh.add_surface_from_arrays(PrimitiveType::TRIANGLES, &arrays);
    mesh
}

// Silence the unused-warning for the palette's convenience alias in builds that
// don't touch every atom kind constant directly.
#[allow(dead_code)]
fn _kind_palette_ref() -> usize {
    AtomKind::PALETTE.len()
}
