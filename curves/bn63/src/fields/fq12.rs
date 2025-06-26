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
        Fq2::new(
            MontFp!("1934652787673387003"),
            MontFp!("10883804381893750551"),
        ),
        // Fp2::NONRESIDUE^(((q^2) - 1) / 6)
        Fq2::new(MontFp!("15577975247367842572"), Fq::ZERO),
        // Fp2::NONRESIDUE^(((q^3) - 1) / 6)
        Fq2::new(
            MontFp!("2815240724822615819"),
            MontFp!("16798630968913295944"),
        ),
        // Fp2::NONRESIDUE^(((q^4) - 1) / 6)
        Fq2::new(MontFp!("11043637364960361766"), Fq::ZERO),
        // Fp2::NONRESIDUE^(((q^5) - 1) / 6)
        Fq2::new(
            MontFp!("16976002852497441255"),
            MontFp!("3194425091572361385"),
        ),
        // Fp2::NONRESIDUE^(((q^6) - 1) / 6)
        Fq2::new(MontFp!("3011743889503200494"), Fq::ZERO),
        // Fp2::NONRESIDUE^(((q^7) - 1) / 6)
        Fq2::new(
            MontFp!("217034790608001361"),
            MontFp!("17770040737513226044"),
        ),
        // Fp2::NONRESIDUE^(((q^8) - 1) / 6)
        Fq2::new(MontFp!("2949797105425105092"), Fq::ZERO),
        // Fp2::NONRESIDUE^(((q^9) - 1) / 6)
        Fq2::new(
            MontFp!("3905326557003353764"),
            MontFp!("13772959849570479790"),
        ),
        // Fp2::NONRESIDUE^(((q^10) - 1) / 6)
        Fq2::new(MontFp!("3397693575357255426"), Fq::ZERO),
        // Fp2::NONRESIDUE^(((q^11) - 1) / 6)
        Fq2::new(
            MontFp!("427051393067369148"),
            MontFp!("2654847796777835667"),
        ),
    ];
}
