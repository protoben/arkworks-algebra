use ark_ff::{fields::*, MontFp};

use crate::*;

pub type Fq6 = Fp6<Fq6Config>;

#[derive(Clone, Copy)]
pub struct Fq6Config;

impl Fp6Config for Fq6Config {
    type Fp2Config = Fq2Config;

    /// NONRESIDUE = U+9
    const NONRESIDUE: Fq2 = Fq2::new(MontFp!("9"), Fq::ONE);

    const FROBENIUS_COEFF_FP6_C1: &'static [Fq2] = &[
        // Fp2::NONRESIDUE^(((q^0) - 1) / 3)
        Fq2::new(Fq::ONE, Fq::ZERO),
        // Fp2::NONRESIDUE^(((q^1) - 1) / 3)
        Fq2::new(
            MontFp!("9447033008302988816"),
            MontFp!("4993688792287293668"),
        ),
        // Fp2::NONRESIDUE^(((q^2) - 1) / 3)
        Fq2::new(MontFp!("10977246212795789640"), Fq::ZERO),
        // Fp2::NONRESIDUE^(((q^3) - 1) / 3)
        Fq2::new(MontFp!("14282140240228910547"), MontFp!("99")),
        // Fp2::NONRESIDUE^(((q^4) - 1) / 3)
        Fq2::new(MontFp!("14899400062266438237"), Fq::ZERO),
        // Fp2::NONRESIDUE^(((q^5) - 1) / 3)
        Fq2::new(
            MontFp!("16426983657807468186"),
            MontFp!("8808657312839593921"),
        ),
    ];

    const FROBENIUS_COEFF_FP6_C2: &'static [Fq2] = &[
        // Fp2::NONRESIDUE^((2*(q^0) - 2) / 3)
        Fq2::new(Fq::ONE, Fq::ZERO),
        // Fp2::NONRESIDUE^((2*(q^1) - 2) / 3)
        Fq2::new(
            MontFp!("3169860291736066552"),
            MontFp!("1568466356636742399"),
        ),
        // Fp2::NONRESIDUE^((2*(q^2) - 2) / 3)
        Fq2::new(MontFp!("6094353107138948651"), Fq::ZERO),
        // Fp2::NONRESIDUE^((2*(q^3) - 2) / 3)
        Fq2::new(
            MontFp!("3696619507225245745"),
            MontFp!("2639269374417157822"),
        ),
        // Fp2::NONRESIDUE^((2*(q^4) - 2) / 3)
        Fq2::new(MontFp!("150229326907837"), Fq::ZERO),
        // Fp2::NONRESIDUE^((2*(q^5) - 2) / 3)
        Fq2::new(
            MontFp!("5322526873970400681"),
            MontFp!("1886767605411956268"),
        ),
    ];

    //#[inline(always)]
    //fn mul_fp2_by_nonresidue_in_place(fe: &mut Fq2) -> &mut Fq2 {
    //    // (c0+u*c1)*(9+u) = (9*c0-c1)+u*(9*c1+c0)
    //    let mut f = *fe;
    //    f.double_in_place().double_in_place().double_in_place();
    //    let mut c0 = fe.c1;
    //    Fq2Config::mul_fp_by_nonresidue_in_place(&mut c0);
    //    c0 += &f.c0;
    //    c0 += &fe.c0;
    //    let c1 = f.c1 + fe.c1 + fe.c0;
    //    *fe = Fq2::new(c0, c1);
    //    fe
    //}
}
