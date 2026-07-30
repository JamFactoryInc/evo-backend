//! Central tuning knobs for the simulation.
//!
//! Everything the "physics of life" depends on lives here so behaviour can be
//! retuned without hunting through the logic. Energy is a single currency that
//! flows on bond sites (the CA/circuit layer) while `pool` is a molecule's
//! metabolic reserve (its life force). Logic converts pool -> signal; actuators
//! burn pool; solar/eating refill pool.

/// Signal threshold: a bond-site energy >= HI counts as logic "high".
pub const HI: f32 = 0.5;
/// Energy emitted by a gate/actuator when it fires.
pub const QUANTUM: f32 = 1.0;

/// Fractional loss when a conductor re-emits energy (resistance).
pub const WIRE_LOSS: f32 = 0.02;

/// Pool energy harvested per Solar atom per tick (photosynthesis).
pub const SOLAR_RATE: f32 = 0.14;
/// Steady signal a Solar atom radiates on its bonds (an always-on power line).
pub const SOLAR_SIGNAL: f32 = 1.0;

/// Pool cost for a gate to drive its output high (control is not free).
pub const GATE_COST: f32 = 0.015;

/// Battery storage capacity.
pub const BATT_CAP: f32 = 24.0;
/// Max energy a battery releases per tick when its control line is high.
pub const BATT_DRAIN: f32 = 3.0;
/// Loss when charging a battery.
pub const BATT_LOSS: f32 = 0.03;

/// World force produced per unit of intake signal by a thruster.
pub const THRUST_FORCE: f32 = 2000.0;
/// Pool burned per unit of thrust (fuel). No fuel -> no thrust.
pub const THRUST_FUEL: f32 = 0.01;

/// Pool drained per atom per tick just to stay alive (metabolism).
pub const UPKEEP: f32 = 0.001;

/// Pool gained by an Eater per atom it consumes.
pub const FOOD_VALUE: f32 = 7.0;
/// Signal an Eater radiates on the tick after it feeds (a "bite" pulse).
pub const EAT_PULSE: f32 = 1.0;
/// Eat reach as a fraction of hex size, measured from the mouth point.
pub const EAT_RANGE_FRAC: f32 = 1.05;

/// Sensor look-ahead distance, in hex units.
pub const SENSOR_RANGE_HEX: f32 = 5.0;
/// Foreign-atom count that saturates a sensor to signal 1.0.
pub const SENSOR_SATURATION: f32 = 4.0;

/// Pool at which a molecule reproduces (splits off a mutated child).
pub const REPRO_THRESHOLD: f32 = 42.0;
/// Fraction of pool handed to the child on reproduction.
pub const REPRO_CHILD_FRAC: f32 = 0.45;
/// Pool a freshly spawned molecule starts with.
pub const START_POOL: f32 = 22.0;
/// Energy stored in a free-floating Food atom.
pub const FOOD_POOL: f32 = 5.0;

/// Per-second velocity retained is `1 - LINEAR_DRAG*dt` (space is viscous so
/// motion settles and the sim stays bounded).
pub const LINEAR_DRAG: f32 = 0.55;
pub const ANGULAR_DRAG: f32 = 1.6;

/// Soft world radius (world units). Beyond it a gentle spring pulls molecules
/// back so the population stays on-screen.
pub const WORLD_RADIUS: f32 = 1400.0;
pub const BOUNDARY_PULL: f32 = 12.0;

/// Population / food caps to bound the work per frame.
pub const MAX_MOLECULES: usize = 400;
pub const MAX_FOOD: usize = 1000;

/// Target ambient food count; the world tops this up over time (a food rain
/// that seeds the ecosystem so early molecules have something to eat).
pub const AMBIENT_FOOD_TARGET: usize = 200;
pub const FOOD_SPAWN_PER_STEP: usize = 3;

/// Genome growth bounds for the initial random population.
pub const SEED_MIN_ATOMS: usize = 5;
pub const SEED_MAX_ATOMS: usize = 30;
pub const MAX_ATOMS: usize = 48;

/// Mutation probabilities applied per reproduction.
pub const MUT_POINT: f32 = 0.18; // change an atom's kind
pub const MUT_ROTATE: f32 = 0.20; // change an atom's facing
pub const MUT_PARAM: f32 = 0.15; // tweak a numeric param
pub const MUT_ADD: f32 = 0.45; // grow a new atom
pub const MUT_REMOVE: f32 = 0.12; // prune an atom
