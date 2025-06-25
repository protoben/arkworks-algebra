use ark_ec::{
    bn,
    bn::{Bn, BnConfig, TwistType},
};
use ark_ff::MontFp;

use crate::*;

pub mod g1;
pub mod g2;

#[cfg(test)]
mod tests;

pub struct Config;

impl BnConfig for Config {
    const X: &'static [u64] = &[4965661367192848881];
    /// `x` is positive.
    const X_IS_NEGATIVE: bool = false;
    const ATE_LOOP_COUNT: &'static [i8] = &[
        0, 0, 0, 1, 0, 1, 0, -1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1, 0, -1, 0, 0, 0, 1, 0, -1, 0, 0, 0,
        0, -1, 0, 0, 1, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, -1, 0, 0, -1, 0, 1, 0, -1, 0, 0, 0, -1, 0,
        -1, 0, 0, 0, 1, 0, 1, 1,
    ];

    const TWIST_MUL_BY_Q_X: Fq2 = Fq2::new(MontFp!("99"), MontFp!("99"));
    const TWIST_MUL_BY_Q_Y: Fq2 = Fq2::new(MontFp!("99"), MontFp!("99"));
    const TWIST_TYPE: TwistType = TwistType::D;
    type Fp = Fq;
    type Fp2Config = Fq2Config;
    type Fp6Config = Fq6Config;
    type Fp12Config = Fq12Config;
    type G1Config = g1::Config;
    type G2Config = g2::Config;
}

pub type Bn63 = Bn<Config>;

pub type G1Affine = bn::G1Affine<Config>;
pub type G1Projective = bn::G1Projective<Config>;
pub type G2Affine = bn::G2Affine<Config>;
pub type G2Projective = bn::G2Projective<Config>;
