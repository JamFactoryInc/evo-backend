//! Atom taxonomy: the "elements" a molecule is built from.
//!
//! Each atom occupies one hex cell and has up to six bond sites (one per
//! [`crate::hex_grid::Dir`]). What an atom *does* with the energy arriving on
//! those sites is decided here; the synchronous evaluation lives in
//! `molecule.rs`. Every kind is orientation-aware via a `facing` direction so
//! the same element can be wired into a circuit many ways.

/// Boolean logic operators for gate atoms, generalised to N connected inputs.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum GateOp {
    And,  // fires when every connected input is high (>=1 input)
    Or,   // fires when any input is high
    Nor,  // fires when no input is high (self-powered inverter / oscillator)
    Nand, // fires unless all inputs are high
    Xor,  // fires when an odd number of inputs are high
    Xnor, // fires when an even number of inputs are high (includes zero)
}

impl GateOp {
    pub const ALL: [GateOp; 6] = [
        GateOp::And,
        GateOp::Or,
        GateOp::Nor,
        GateOp::Nand,
        GateOp::Xor,
        GateOp::Xnor,
    ];

    /// Evaluate the gate from the number of high inputs and the input count.
    #[inline]
    pub fn eval(self, high: u32, inputs: u32) -> bool {
        match self {
            GateOp::And => inputs > 0 && high == inputs,
            GateOp::Or => high > 0,
            GateOp::Nor => high == 0,
            GateOp::Nand => inputs > 0 && high < inputs,
            GateOp::Xor => high % 2 == 1,
            GateOp::Xnor => high % 2 == 0,
        }
    }
}

/// The kind of an atom. This is the heritable "type gene".
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum AtomKind {
    /// The mandatory core of every molecule. Behaves as a conductor hub; if it
    /// is destroyed the whole molecule dies.
    Seed,
    /// Passes/diffuses energy between its bonds (a wire).
    Conductor,
    /// Harvests ambient energy into the pool and radiates a steady power signal.
    Solar,
    /// Boolean logic element (see [`GateOp`]).
    Gate(GateOp),
    /// Stores energy; releases it on the front site while its control (back)
    /// site is held high. Charges from its other sites.
    Battery,
    /// Converts an intake signal into world force along its facing direction,
    /// burning pool as fuel.
    Thruster,
    /// A mouth: consumes a foreign atom it touches, feeding the pool and firing
    /// a bite pulse.
    Eater,
    /// A sense organ: emits a signal proportional to foreign matter ahead.
    Sensor,
    /// Inert free-floating nutrient. Not part of living molecules' behaviour;
    /// exists to be eaten.
    Food,
}

impl AtomKind {
    /// The palette used for the initial random population's structural atoms.
    /// (`Food` and `Seed` are handled separately.)
    pub const PALETTE: [AtomKind; 10] = [
        AtomKind::Conductor,
        // AtomKind::Solar,
        AtomKind::Gate(GateOp::And),
        AtomKind::Gate(GateOp::Or),
        AtomKind::Gate(GateOp::Nor),
        AtomKind::Gate(GateOp::Xor),
        AtomKind::Battery,
        AtomKind::Thruster,
        AtomKind::Eater,
        AtomKind::Sensor,
        AtomKind::Conductor, // weight conductors up a touch
    ];

    /// Whether this atom actively senses/eats and therefore needs the world's
    /// spatial index during a step.
    #[inline]
    pub fn is_interactive(self) -> bool {
        matches!(self, AtomKind::Eater | AtomKind::Sensor)
    }

    /// RGB base colour (0..1) for instanced rendering. Activity brightening is
    /// applied on top at render time.
    pub fn color(self) -> (f32, f32, f32) {
        match self {
            AtomKind::Seed => (0.95, 0.95, 0.80),
            AtomKind::Conductor => (0.45, 0.48, 0.55),
            AtomKind::Solar => (0.20, 0.80, 0.35),
            AtomKind::Gate(op) => match op {
                GateOp::And => (0.30, 0.55, 0.95),
                GateOp::Or => (0.30, 0.75, 0.95),
                GateOp::Nor => (0.55, 0.35, 0.95),
                GateOp::Nand => (0.40, 0.45, 0.95),
                GateOp::Xor => (0.70, 0.30, 0.95),
                GateOp::Xnor => (0.85, 0.35, 0.85),
            },
            AtomKind::Battery => (0.95, 0.80, 0.20),
            AtomKind::Thruster => (0.95, 0.35, 0.20),
            AtomKind::Eater => (0.90, 0.15, 0.30),
            AtomKind::Sensor => (0.20, 0.90, 0.90),
            AtomKind::Food => (0.55, 0.45, 0.30),
        }
    }
}
