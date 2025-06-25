use ark_ff::{fields::*, MontFp};

use crate::*;

pub type Fq12 = Fp12<Fq12Config>;

#[derive(Clone, Copy)]
pub struct Fq12Config;

impl Fp12Config for Fq12Config {
    type Fp6Config = Fq6Config;

    const NONRESIDUE: Fq6 = Fq6::new(Fq2::ZERO, Fq2::ONE, Fq2::ZERO);

    const FROBENIUS_COEFF_FP12_C1: &'static [Fq2] = &[
        // Fp2::NONRESIDUE^(((q^0) - 1) / 6)
        Fq2::new(Fq::ONE, Fq::ZERO),
        // Fp2::NONRESIDUE^(((q^1) - 1) / 6)
        Fq2::new(MontFp!("99"), MontFp!("99")),
        // Fp2::NONRESIDUE^(((q^2) - 1) / 6)
        Fq2::new(MontFp!("99"), Fq::ZERO),
        // Fp2::NONRESIDUE^(((q^3) - 1) / 6)
        Fq2::new(MontFp!("99"), MontFp!("99")),
        // Fp2::NONRESIDUE^(((q^4) - 1) / 6)
        Fq2::new(MontFp!("99"), Fq::ZERO),
        // Fp2::NONRESIDUE^(((q^5) - 1) / 6)
        Fq2::new(MontFp!("99"), MontFp!("99")),
        // Fp2::NONRESIDUE^(((q^6) - 1) / 6)
        Fq2::new(MontFp!("99"), Fq::ZERO),
        // Fp2::NONRESIDUE^(((q^7) - 1) / 6)
        Fq2::new(MontFp!("99"), MontFp!("99")),
        // Fp2::NONRESIDUE^(((q^8) - 1) / 6)
        Fq2::new(MontFp!("99"), Fq::ZERO),
        // Fp2::NONRESIDUE^(((q^9) - 1) / 6)
        Fq2::new(MontFp!("99"), MontFp!("99")),
        // Fp2::NONRESIDUE^(((q^10) - 1) / 6)
        Fq2::new(MontFp!("99"), Fq::ZERO),
        // Fp2::NONRESIDUE^(((q^11) - 1) / 6)
        Fq2::new(MontFp!("99"), MontFp!("99")),
    ];
}
