use super::{Fp, FpConfig};
use crate::{
    biginteger::arithmetic as fa, BigInt, BigInteger, PrimeField, SqrtPrecomputation, Zero,
};
use ark_ff_macros::unroll_for_loops;
use ark_std::marker::PhantomData;

pub const PRECOMP_TABLE_SIZE: usize = 1 << 14;

/// A trait that specifies the constants and arithmetic procedures
/// for Montgomery arithmetic over the prime field defined by `MODULUS`.
///
/// # Note
/// Manual implementation of this trait is not recommended unless one wishes
/// to specialize arithmetic methods. Instead, the
/// [`MontConfig`][`ark_ff_macros::MontConfig`] derive macro should be used.
pub trait MontConfig<const N: usize>: 'static + Sync + Send + Sized {
    /// The modulus of the field.
    const MODULUS: BigInt<N>;

    /// Let `M` be the power of 2^64 nearest to `Self::MODULUS_BITS`. Then
    /// `R = M % Self::MODULUS`.
    const R: BigInt<N> = Self::MODULUS.montgomery_r();

    /// R2 = R^2 % Self::MODULUS
    const R2: BigInt<N> = Self::MODULUS.montgomery_r2();

    /// INV = -MODULUS^{-1} mod 2^64
    const INV: u64 = inv::<Self, N>();

    /// A multiplicative generator of the field.
    /// `Self::GENERATOR` is an element having multiplicative order
    /// `Self::MODULUS - 1`.
    const GENERATOR: Fp<MontBackend<Self, N>, N>;

    /// Can we use the no-carry optimization for multiplication
    /// outlined [here](https://hackmd.io/@gnark/modular_multiplication)?
    ///
    /// This optimization applies if
    /// (a) `Self::MODULUS[N-1] < u64::MAX >> 1`, and
    /// (b) the bits of the modulus are not all 1.
    #[doc(hidden)]
    const CAN_USE_NO_CARRY_MUL_OPT: bool = can_use_no_carry_mul_optimization::<Self, N>();

    /// Can we use the no-carry optimization for squaring
    /// outlined [here](https://hackmd.io/@gnark/modular_multiplication)?
    ///
    /// This optimization applies if
    /// (a) `Self::MODULUS[N-1] < u64::MAX >> 2`, and
    /// (b) the bits of the modulus are not all 1.
    #[doc(hidden)]
    const CAN_USE_NO_CARRY_SQUARE_OPT: bool = can_use_no_carry_mul_optimization::<Self, N>();

    /// Does the modulus have a spare unused bit
    ///
    /// This condition applies if
    /// (a) `Self::MODULUS[N-1] >> 63 == 0`
    #[doc(hidden)]
    const MODULUS_HAS_SPARE_BIT: bool = modulus_has_spare_bit::<Self, N>();

    /// 2^s root of unity computed by GENERATOR^t
    const TWO_ADIC_ROOT_OF_UNITY: Fp<MontBackend<Self, N>, N>;

    /// An integer `b` such that there exists a multiplicative subgroup
    /// of size `b^k` for some integer `k`.
    const SMALL_SUBGROUP_BASE: Option<u32> = None;

    /// The integer `k` such that there exists a multiplicative subgroup
    /// of size `Self::SMALL_SUBGROUP_BASE^k`.
    const SMALL_SUBGROUP_BASE_ADICITY: Option<u32> = None;

    /// GENERATOR^((MODULUS-1) / (2^s *
    /// SMALL_SUBGROUP_BASE^SMALL_SUBGROUP_BASE_ADICITY)).
    /// Used for mixed-radix FFT.
    const LARGE_SUBGROUP_ROOT_OF_UNITY: Option<Fp<MontBackend<Self, N>, N>> = None;

    /// Precomputed material for use when computing square roots.
    /// The default is to use the standard Tonelli-Shanks algorithm.
    const SQRT_PRECOMP: Option<SqrtPrecomputation<Fp<MontBackend<Self, N>, N>>> =
        sqrt_precomputation::<N, Self>();

    #[allow(long_running_const_eval)]
    const SMALL_ELEMENT_MONTGOMERY_PRECOMP: [Fp<MontBackend<Self, N>, N>; PRECOMP_TABLE_SIZE] =
        small_element_montgomery_precomputation::<N, Self>();

    /// (MODULUS + 1) / 4 when MODULUS % 4 == 3. Used for square root precomputations.
    #[doc(hidden)]
    const MODULUS_PLUS_ONE_DIV_FOUR: Option<BigInt<N>> = {
        match Self::MODULUS.mod_4() == 3 {
            true => {
                let (modulus_plus_one, carry) =
                    Self::MODULUS.const_add_with_carry(&BigInt::<N>::one());
                let mut result = modulus_plus_one.divide_by_2_round_down();
                // Since modulus_plus_one is even, dividing by 2 results in a MSB of 0.
                // Thus we can set MSB to `carry` to get the correct result of (MODULUS + 1) // 2:
                result.0[N - 1] |= (carry as u64) << 63;
                Some(result.divide_by_2_round_down())
            },
            false => None,
        }
    };

    /// Number of spare bits (i.e. significant bits equal to 0) in the modulus `p`
    #[doc(hidden)]
    const MODULUS_NUM_SPARE_BITS: u32 = Self::MODULUS.num_spare_bits();

    /// 2 * MODULUS. Last limb is represented as a u64, but is always 0 or 1.
    #[doc(hidden)]
    const MODULUS_TIMES_2: ([u64; N], u64) = {
        let (modulus_times_2, carry) = Self::MODULUS.const_mul2_with_carry();
        (modulus_times_2.0, carry as u64)
    };

    /// Barrett reduction constant: $R' = 2^{\text{MODULUS_BITS}}$.
    #[doc(hidden)]
    const BARRETT_RPRIME: (BigInt<N>, bool) = {
        let num_spare_bits = Self::MODULUS.num_spare_bits();
        assert!(num_spare_bits <= 64);
        if num_spare_bits == 0 {
            (BigInt::<N>::zero(), true)
        } else {
            (BigInt::<N>::pow_2(64 - num_spare_bits), false)
        }
    };

    /// Barrett reduction constant: mu = floor( r * R' / (2 * MODULUS) )
    /// = floor( 2^(64 * (N + 1) - num_spare_bits(MODULUS) - 1) / MODULUS )
    #[doc(hidden)]
    const BARRETT_MU: u64 = {
        assert!(Self::MODULUS.num_spare_bits() < 64);
        let r_times_r_prime_over_2 =
            crate::const_helpers::RBuffer::<N>([0u64; N],
                1 << (63 - Self::MODULUS.num_spare_bits()));
        let result: BigInt<N> = const_modulo!(r_times_r_prime_over_2, &Self::MODULUS);
        // Result should be a u64
        result.0[0]
    };

    /// Sets `a = a + b`.
    #[inline(always)]
    fn add_assign(a: &mut Fp<MontBackend<Self, N>, N>, b: &Fp<MontBackend<Self, N>, N>) {
        // This cannot exceed the backing capacity.
        let c = a.0.add_with_carry(&b.0);
        // However, it may need to be reduced
        if Self::MODULUS_HAS_SPARE_BIT {
            a.subtract_modulus()
        } else {
            a.subtract_modulus_with_carry(c)
        }
    }

    /// Sets `a = a - b`.
    #[inline(always)]
    fn sub_assign(a: &mut Fp<MontBackend<Self, N>, N>, b: &Fp<MontBackend<Self, N>, N>) {
        // If `other` is larger than `self`, add the modulus to self first.
        if b.0 > a.0 {
            a.0.add_with_carry(&Self::MODULUS);
        }
        a.0.sub_with_borrow(&b.0);
    }

    /// Sets `a = 2 * a`.
    #[inline(always)]
    fn double_in_place(a: &mut Fp<MontBackend<Self, N>, N>) {
        // This cannot exceed the backing capacity.
        let c = a.0.mul2();
        // However, it may need to be reduced.
        if Self::MODULUS_HAS_SPARE_BIT {
            a.subtract_modulus()
        } else {
            a.subtract_modulus_with_carry(c)
        }
    }

    /// Sets `a = -a`.
    #[inline(always)]
    fn neg_in_place(a: &mut Fp<MontBackend<Self, N>, N>) {
        if !a.is_zero() {
            let mut tmp = Self::MODULUS;
            tmp.sub_with_borrow(&a.0);
            a.0 = tmp;
        }
    }

    /// This modular multiplication algorithm uses Montgomery
    /// reduction for efficient implementation. It also additionally
    /// uses the "no-carry optimization" outlined
    /// [here](https://hackmd.io/@gnark/modular_multiplication) if
    /// `Self::MODULUS` has (a) a non-zero MSB, and (b) at least one
    /// zero bit in the rest of the modulus.
    #[unroll_for_loops(12)]
    #[inline(always)]
    fn mul_assign(a: &mut Fp<MontBackend<Self, N>, N>, b: &Fp<MontBackend<Self, N>, N>) {
        // No-carry optimisation applied to CIOS
        if Self::CAN_USE_NO_CARRY_MUL_OPT {
            if N <= 6
                && N > 1
                && cfg!(all(
                    feature = "asm",
                    target_feature = "bmi2",
                    target_feature = "adx",
                    target_arch = "x86_64"
                ))
            {
                #[cfg(
                    all(
                        feature = "asm",
                        target_feature = "bmi2",
                        target_feature = "adx",
                        target_arch = "x86_64"
                    )
                )]
                #[allow(unsafe_code, unused_mut)]
                #[rustfmt::skip]

                // Tentatively avoid using assembly for `N == 1`.
                match N {
                    2 => { ark_ff_asm::x86_64_asm_mul!(2, (a.0).0, (b.0).0); },
                    3 => { ark_ff_asm::x86_64_asm_mul!(3, (a.0).0, (b.0).0); },
                    4 => { ark_ff_asm::x86_64_asm_mul!(4, (a.0).0, (b.0).0); },
                    5 => { ark_ff_asm::x86_64_asm_mul!(5, (a.0).0, (b.0).0); },
                    6 => { ark_ff_asm::x86_64_asm_mul!(6, (a.0).0, (b.0).0); },
                    _ => unsafe { ark_std::hint::unreachable_unchecked() },
                };
            } else {
                let mut r = [0u64; N];

                for i in 0..N {
                    let mut carry1 = 0u64;
                    r[0] = fa::mac(r[0], (a.0).0[0], (b.0).0[i], &mut carry1);

                    let k = r[0].wrapping_mul(Self::INV);

                    let mut carry2 = 0u64;
                    fa::mac_discard(r[0], k, Self::MODULUS.0[0], &mut carry2);

                    for j in 1..N {
                        r[j] = fa::mac_with_carry(r[j], (a.0).0[j], (b.0).0[i], &mut carry1);
                        r[j - 1] = fa::mac_with_carry(r[j], k, Self::MODULUS.0[j], &mut carry2);
                    }
                    r[N - 1] = carry1 + carry2;
                }
                (a.0).0.copy_from_slice(&r);
            }
            a.subtract_modulus();
        } else {
            // Alternative implementation
            // Implements CIOS.
            let (carry, res) = a.mul_without_cond_subtract(b);
            *a = res;

            if Self::MODULUS_HAS_SPARE_BIT {
                a.subtract_modulus_with_carry(carry);
            } else {
                a.subtract_modulus();
            }
        }
    }

    #[inline(always)]
    #[unroll_for_loops(12)]
    fn square_in_place(a: &mut Fp<MontBackend<Self, N>, N>) {
        if N == 1 {
            // We default to multiplying with `a` using the `Mul` impl
            // for the N == 1 case
            *a *= *a;
            return;
        }
        if Self::CAN_USE_NO_CARRY_SQUARE_OPT
            && (2..=6).contains(&N)
            && cfg!(all(
                feature = "asm",
                target_feature = "bmi2",
                target_feature = "adx",
                target_arch = "x86_64"
            ))
        {
            #[cfg(all(
                feature = "asm",
                target_feature = "bmi2",
                target_feature = "adx",
                target_arch = "x86_64"
            ))]
            #[allow(unsafe_code, unused_mut)]
            #[rustfmt::skip]
            match N {
                2 => { ark_ff_asm::x86_64_asm_square!(2, (a.0).0); },
                3 => { ark_ff_asm::x86_64_asm_square!(3, (a.0).0); },
                4 => { ark_ff_asm::x86_64_asm_square!(4, (a.0).0); },
                5 => { ark_ff_asm::x86_64_asm_square!(5, (a.0).0); },
                6 => { ark_ff_asm::x86_64_asm_square!(6, (a.0).0); },
                _ => unsafe { ark_std::hint::unreachable_unchecked() },
            };
            a.subtract_modulus();
            return;
        }

        let mut r = crate::const_helpers::MulBuffer::<N>::zeroed();

        let mut carry = 0;
        for i in 0..(N - 1) {
            for j in (i + 1)..N {
                r[i + j] = fa::mac_with_carry(r[i + j], (a.0).0[i], (a.0).0[j], &mut carry);
            }
            r.b1[i] = carry;
            carry = 0;
        }

        r.b1[N - 1] = r.b1[N - 2] >> 63;
        for i in 2..(2 * N - 1) {
            r[2 * N - i] = (r[2 * N - i] << 1) | (r[2 * N - (i + 1)] >> 63);
        }
        r.b0[1] <<= 1;

        for i in 0..N {
            r[2 * i] = fa::mac_with_carry(r[2 * i], (a.0).0[i], (a.0).0[i], &mut carry);
            carry = fa::adc(&mut r[2 * i + 1], 0, carry);
        }
        // Montgomery reduction
        let mut carry2 = 0;
        for i in 0..N {
            let k = r[i].wrapping_mul(Self::INV);
            carry = 0;
            fa::mac_discard(r[i], k, Self::MODULUS.0[0], &mut carry);
            for j in 1..N {
                r[j + i] = fa::mac_with_carry(r[j + i], k, Self::MODULUS.0[j], &mut carry);
            }
            carry2 = fa::adc(&mut r.b1[i], carry, carry2);
        }
        (a.0).0.copy_from_slice(&r.b1);
        if Self::MODULUS_HAS_SPARE_BIT {
            a.subtract_modulus();
        } else {
            a.subtract_modulus_with_carry(carry2 != 0);
        }
    }

    fn inverse(a: &Fp<MontBackend<Self, N>, N>) -> Option<Fp<MontBackend<Self, N>, N>> {
        if a.is_zero() {
            return None;
        }
        // Guajardo Kumar Paar Pelzl
        // Efficient Software-Implementation of Finite Fields with Applications to
        // Cryptography
        // Algorithm 16 (BEA for Inversion in Fp)

        let one = BigInt::from(1u64);

        let mut u = a.0;
        let mut v = Self::MODULUS;
        let mut b = Fp::new_unchecked(Self::R2); // Avoids unnecessary reduction step.
        let mut c = Fp::zero();

        while u != one && v != one {
            while u.is_even() {
                u.div2();

                if b.0.is_even() {
                    b.0.div2();
                } else {
                    let carry = b.0.add_with_carry(&Self::MODULUS);
                    b.0.div2();
                    if !Self::MODULUS_HAS_SPARE_BIT && carry {
                        (b.0).0[N - 1] |= 1 << 63;
                    }
                }
            }

            while v.is_even() {
                v.div2();

                if c.0.is_even() {
                    c.0.div2();
                } else {
                    let carry = c.0.add_with_carry(&Self::MODULUS);
                    c.0.div2();
                    if !Self::MODULUS_HAS_SPARE_BIT && carry {
                        (c.0).0[N - 1] |= 1 << 63;
                    }
                }
            }

            if v < u {
                u.sub_with_borrow(&v);
                b -= &c;
            } else {
                v.sub_with_borrow(&u);
                c -= &b;
            }
        }

        if u == one {
            Some(b)
        } else {
            Some(c)
        }
    }

    fn from_u64(r: u64) -> Option<Fp<MontBackend<Self, N>, N>> {
        if r < PRECOMP_TABLE_SIZE as u64 {
            Some(Self::SMALL_ELEMENT_MONTGOMERY_PRECOMP[r as usize])
        } else if BigInt::from(r) >= <MontBackend<Self, N>>::MODULUS {
            None
        } else {
            Some(Fp::new_unchecked(Self::R2).mul_u64(r))
        }
    }

    fn from_bigint(r: BigInt<N>) -> Option<Fp<MontBackend<Self, N>, N>> {
        let mut r = Fp::new_unchecked(r);
        if r.is_zero() {
            Some(r)
        } else if r.is_geq_modulus() {
            None
        } else {
            r *= &Fp::new_unchecked(Self::R2);
            Some(r)
        }
    }

    #[inline]
    #[cfg_attr(not(target_family = "wasm"), unroll_for_loops(12))]
    #[cfg_attr(target_family = "wasm", unroll_for_loops(6))]
    #[allow(clippy::modulo_one)]
    fn into_bigint(a: Fp<MontBackend<Self, N>, N>) -> BigInt<N> {
        let mut r = (a.0).0;
        // Montgomery Reduction
        for i in 0..N {
            let k = r[i].wrapping_mul(Self::INV);
            let mut carry = 0;

            fa::mac_with_carry(r[i], k, Self::MODULUS.0[0], &mut carry);
            for j in 1..N {
                r[(j + i) % N] =
                    fa::mac_with_carry(r[(j + i) % N], k, Self::MODULUS.0[j], &mut carry);
            }
            r[i % N] = carry;
        }

        BigInt::new(r)
    }

    #[unroll_for_loops(12)]
    fn sum_of_products<const M: usize>(
        a: &[Fp<MontBackend<Self, N>, N>; M],
        b: &[Fp<MontBackend<Self, N>, N>; M],
    ) -> Fp<MontBackend<Self, N>, N> {
        // Adapted from https://github.com/zkcrypto/bls12_381/pull/84 by @str4d.

        // For a single `a x b` multiplication, operand scanning (schoolbook) takes each
        // limb of `a` in turn, and multiplies it by all of the limbs of `b` to compute
        // the result as a double-width intermediate representation, which is then fully
        // reduced at the carry. Here however we have pairs of multiplications (a_i, b_i),
        // the results of which are summed.
        //
        // The intuition for this algorithm is two-fold:
        // - We can interleave the operand scanning for each pair, by processing the jth
        //   limb of each `a_i` together. As these have the same offset within the overall
        //   operand scanning flow, their results can be summed directly.
        // - We can interleave the multiplication and reduction steps, resulting in a
        //   single bitshift by the limb size after each iteration. This means we only
        //   need to store a single extra limb overall, instead of keeping around all the
        //   intermediate results and eventually having twice as many limbs.

        let modulus_size = Self::MODULUS.const_num_bits() as usize;
        if modulus_size >= 64 * N - 1 {
            a.iter().zip(b).map(|(a, b)| *a * b).sum()
        } else if M == 2 {
            // Algorithm 2, line 2
            let result = (0..N).fold(BigInt::zero(), |mut result, j| {
                // Algorithm 2, line 3
                let mut carry_a = 0;
                let mut carry_b = 0;
                for (a, b) in a.iter().zip(b) {
                    let a = &a.0;
                    let b = &b.0;
                    let mut carry2 = 0;
                    result.0[0] = fa::mac(result.0[0], a.0[j], b.0[0], &mut carry2);
                    for k in 1..N {
                        result.0[k] = fa::mac_with_carry(result.0[k], a.0[j], b.0[k], &mut carry2);
                    }
                    carry_b = fa::adc(&mut carry_a, carry_b, carry2);
                }

                let k = result.0[0].wrapping_mul(Self::INV);
                let mut carry2 = 0;
                fa::mac_discard(result.0[0], k, Self::MODULUS.0[0], &mut carry2);
                for i in 1..N {
                    result.0[i - 1] =
                        fa::mac_with_carry(result.0[i], k, Self::MODULUS.0[i], &mut carry2);
                }
                result.0[N - 1] = fa::adc_no_carry(carry_a, carry_b, &mut carry2);
                result
            });
            let mut result = Fp::new_unchecked(result);
            result.subtract_modulus();
            debug_assert_eq!(
                a.iter().zip(b).map(|(a, b)| *a * b).sum::<Fp<_, N>>(),
                result
            );
            result
        } else {
            let chunk_size = 2 * (N * 64 - modulus_size) - 1;
            // chunk_size is at least 1, since MODULUS_BIT_SIZE is at most N * 64 - 1.
            a.chunks(chunk_size)
                .zip(b.chunks(chunk_size))
                .map(|(a, b)| {
                    // Algorithm 2, line 2
                    let result = (0..N).fold(BigInt::zero(), |mut result, j| {
                        // Algorithm 2, line 3
                        let (temp, carry) = a.iter().zip(b).fold(
                            (result, 0),
                            |(mut temp, mut carry), (Fp(a, _), Fp(b, _))| {
                                let mut carry2 = 0;
                                temp.0[0] = fa::mac(temp.0[0], a.0[j], b.0[0], &mut carry2);
                                for k in 1..N {
                                    temp.0[k] =
                                        fa::mac_with_carry(temp.0[k], a.0[j], b.0[k], &mut carry2);
                                }
                                carry = fa::adc_no_carry(carry, 0, &mut carry2);
                                (temp, carry)
                            },
                        );

                        let k = temp.0[0].wrapping_mul(Self::INV);
                        let mut carry2 = 0;
                        fa::mac_discard(temp.0[0], k, Self::MODULUS.0[0], &mut carry2);
                        for i in 1..N {
                            result.0[i - 1] =
                                fa::mac_with_carry(temp.0[i], k, Self::MODULUS.0[i], &mut carry2);
                        }
                        result.0[N - 1] = fa::adc_no_carry(carry, 0, &mut carry2);
                        result
                    });
                    let mut result = Fp::new_unchecked(result);
                    result.subtract_modulus();
                    debug_assert_eq!(
                        a.iter().zip(b).map(|(a, b)| *a * b).sum::<Fp<_, N>>(),
                        result
                    );
                    result
                })
                .sum()
        }
    }
}

/// Compute -M^{-1} mod 2^64.
pub const fn inv<T: MontConfig<N>, const N: usize>() -> u64 {
    // We compute this as follows.
    // First, MODULUS mod 2^64 is just the lower 64 bits of MODULUS.
    // Hence MODULUS mod 2^64 = MODULUS.0[0] mod 2^64.
    //
    // Next, computing the inverse mod 2^64 involves exponentiating by
    // the multiplicative group order, which is euler_totient(2^64) - 1.
    // Now, euler_totient(2^64) = 1 << 63, and so
    // euler_totient(2^64) - 1 = (1 << 63) - 1 = 1111111... (63 digits).
    // We compute this powering via standard square and multiply.
    let mut inv = 1u64;
    crate::const_for!((_i in 0..63) {
        // Square
        inv = inv.wrapping_mul(inv);
        // Multiply
        inv = inv.wrapping_mul(T::MODULUS.0[0]);
    });
    inv.wrapping_neg()
}

#[inline]
pub const fn can_use_no_carry_mul_optimization<T: MontConfig<N>, const N: usize>() -> bool {
    // Checking the modulus at compile time
    let mut all_remaining_bits_are_one = T::MODULUS.0[N - 1] == u64::MAX >> 1;
    crate::const_for!((i in 1..N) {
        all_remaining_bits_are_one  &= T::MODULUS.0[N - i - 1] == u64::MAX;
    });
    modulus_has_spare_bit::<T, N>() && !all_remaining_bits_are_one
}

#[inline]
pub const fn modulus_has_spare_bit<T: MontConfig<N>, const N: usize>() -> bool {
    T::MODULUS.0[N - 1] >> 63 == 0
}

#[inline]
pub const fn can_use_no_carry_square_optimization<T: MontConfig<N>, const N: usize>() -> bool {
    // Checking the modulus at compile time
    let top_two_bits_are_zero = T::MODULUS.0[N - 1] >> 62 == 0;
    let mut all_remaining_bits_are_one = T::MODULUS.0[N - 1] == u64::MAX >> 2;
    crate::const_for!((i in 1..N) {
        all_remaining_bits_are_one  &= T::MODULUS.0[N - i - 1] == u64::MAX;
    });
    top_two_bits_are_zero && !all_remaining_bits_are_one
}

pub const fn sqrt_precomputation<const N: usize, T: MontConfig<N>>(
) -> Option<SqrtPrecomputation<Fp<MontBackend<T, N>, N>>> {
    match T::MODULUS.mod_4() {
        3 => match T::MODULUS_PLUS_ONE_DIV_FOUR.as_ref() {
            Some(BigInt(modulus_plus_one_div_four)) => Some(SqrtPrecomputation::Case3Mod4 {
                modulus_plus_one_div_four,
            }),
            None => None,
        },
        _ => Some(SqrtPrecomputation::TonelliShanks {
            two_adicity: <MontBackend<T, N>>::TWO_ADICITY,
            quadratic_nonresidue_to_trace: T::TWO_ADIC_ROOT_OF_UNITY,
            trace_of_modulus_minus_one_div_two:
                &<Fp<MontBackend<T, N>, N>>::TRACE_MINUS_ONE_DIV_TWO.0,
        }),
    }
}

/// Adapted the `bn256-table` feature from `halo2curves`:
/// https://github.com/privacy-scaling-explorations/halo2curves/blob/main/script/bn256.py
pub const fn small_element_montgomery_precomputation<const N: usize, T: MontConfig<N>>(
) -> [Fp<MontBackend<T, N>, N>; PRECOMP_TABLE_SIZE] {
    let mut lookup_table: [Fp<MontBackend<T, N>, N>; PRECOMP_TABLE_SIZE] =
        [Fp::new_unchecked(BigInt::zero()); PRECOMP_TABLE_SIZE];

    let mut i: usize = 1;
    while i < PRECOMP_TABLE_SIZE {
        let mut limbs = [0u64; N];
        limbs[0] = i as u64;
        lookup_table[i] = <Fp<MontBackend<T, N>, N>>::new(BigInt::new(limbs));
        i += 1;
    }
    lookup_table
}

/// Construct a [`Fp<MontBackend<T, N>, N>`] element from a literal string. This
/// should be used primarily for constructing constant field elements; in a
/// non-const context, [`Fp::from_str`](`ark_std::str::FromStr::from_str`) is
/// preferable.
///
/// # Panics
///
/// If the integer represented by the string cannot fit in the number
/// of limbs of the `Fp`, this macro results in a
/// * compile-time error if used in a const context
/// * run-time error otherwise.
///
/// # Usage
///
/// ```rust
/// # use ark_test_curves::MontFp;
/// # use ark_test_curves::bls12_381 as ark_bls12_381;
/// # use ark_std::{One, str::FromStr};
/// use ark_bls12_381::Fq;
/// const ONE: Fq = MontFp!("1");
/// const NEG_ONE: Fq = MontFp!("-1");
///
/// fn check_correctness() {
///     assert_eq!(ONE, Fq::one());
///     assert_eq!(Fq::from_str("1").unwrap(), ONE);
///     assert_eq!(NEG_ONE, -Fq::one());
/// }
/// ```
#[macro_export]
macro_rules! MontFp {
    ($c0:expr) => {{
        let (is_positive, limbs) = $crate::ark_ff_macros::to_sign_and_limbs!($c0);
        $crate::Fp::from_sign_and_limbs(is_positive, &limbs)
    }};
}

pub use ark_ff_macros::MontConfig;

pub use MontFp;

pub struct MontBackend<T: MontConfig<N>, const N: usize>(PhantomData<T>);

impl<T: MontConfig<N>, const N: usize> FpConfig<N> for MontBackend<T, N> {
    /// The modulus of the field.
    const MODULUS: crate::BigInt<N> = T::MODULUS;

    /// A multiplicative generator of the field.
    /// `Self::GENERATOR` is an element having multiplicative order
    /// `Self::MODULUS - 1`.
    const GENERATOR: Fp<Self, N> = T::GENERATOR;

    /// Additive identity of the field, i.e. the element `e`
    /// such that, for all elements `f` of the field, `e + f = f`.
    const ZERO: Fp<Self, N> = Fp::new_unchecked(BigInt([0u64; N]));

    /// Multiplicative identity of the field, i.e. the element `e`
    /// such that, for all elements `f` of the field, `e * f = f`.
    const ONE: Fp<Self, N> = Fp::new_unchecked(T::R);

    const TWO_ADICITY: u32 = Self::MODULUS.two_adic_valuation();
    const TWO_ADIC_ROOT_OF_UNITY: Fp<Self, N> = T::TWO_ADIC_ROOT_OF_UNITY;
    const SMALL_SUBGROUP_BASE: Option<u32> = T::SMALL_SUBGROUP_BASE;
    const SMALL_SUBGROUP_BASE_ADICITY: Option<u32> = T::SMALL_SUBGROUP_BASE_ADICITY;
    const LARGE_SUBGROUP_ROOT_OF_UNITY: Option<Fp<Self, N>> = T::LARGE_SUBGROUP_ROOT_OF_UNITY;
    const SQRT_PRECOMP: Option<crate::SqrtPrecomputation<Fp<Self, N>>> = T::SQRT_PRECOMP;
    const SMALL_ELEMENT_MONTGOMERY_PRECOMP: [Fp<Self, N>; PRECOMP_TABLE_SIZE] =
        T::SMALL_ELEMENT_MONTGOMERY_PRECOMP;

    fn add_assign(a: &mut Fp<Self, N>, b: &Fp<Self, N>) {
        T::add_assign(a, b)
    }

    fn sub_assign(a: &mut Fp<Self, N>, b: &Fp<Self, N>) {
        T::sub_assign(a, b)
    }

    fn double_in_place(a: &mut Fp<Self, N>) {
        T::double_in_place(a)
    }

    fn neg_in_place(a: &mut Fp<Self, N>) {
        T::neg_in_place(a)
    }

    /// This modular multiplication algorithm uses Montgomery
    /// reduction for efficient implementation. It also additionally
    /// uses the "no-carry optimization" outlined
    /// [here](https://hackmd.io/@zkteam/modular_multiplication) if
    /// `P::MODULUS` has (a) a non-zero MSB, and (b) at least one
    /// zero bit in the rest of the modulus.
    #[inline]
    fn mul_assign(a: &mut Fp<Self, N>, b: &Fp<Self, N>) {
        T::mul_assign(a, b)
    }

    fn sum_of_products<const M: usize>(a: &[Fp<Self, N>; M], b: &[Fp<Self, N>; M]) -> Fp<Self, N> {
        T::sum_of_products(a, b)
    }

    #[inline]
    #[allow(unused_braces, clippy::absurd_extreme_comparisons)]
    fn square_in_place(a: &mut Fp<Self, N>) {
        T::square_in_place(a)
    }

    fn inverse(a: &Fp<Self, N>) -> Option<Fp<Self, N>> {
        T::inverse(a)
    }

    fn from_bigint(r: BigInt<N>) -> Option<Fp<Self, N>> {
        T::from_bigint(r)
    }

    #[inline]
    #[allow(clippy::modulo_one)]
    fn into_bigint(a: Fp<Self, N>) -> BigInt<N> {
        T::into_bigint(a)
    }

    fn from_u64(r: u64) -> Option<Fp<Self, N>> {
        if BigInt::from(r) >= T::MODULUS { // Access MODULUS via T
            None
        } else {
            // Use the standard From<u64> implementation for Fp
            Some(Fp::<Self, N>::from(r))
        }
    }
}

impl<T: MontConfig<N>, const N: usize> Fp<MontBackend<T, N>, N> {
    #[doc(hidden)]
    pub const R: BigInt<N> = T::R;
    #[doc(hidden)]
    pub const R2: BigInt<N> = T::R2;
    #[doc(hidden)]
    pub const INV: u64 = T::INV;

    /// Construct a new field element from its underlying
    /// [`struct@BigInt`] data type.
    #[inline]
    pub const fn new(element: BigInt<N>) -> Self {
        let mut r = Self(element, PhantomData);
        if r.const_is_zero() {
            r
        } else {
            r = r.mul(&Fp(T::R2, PhantomData));
            r
        }
    }

    /// Construct a new field element from its underlying
    /// [`struct@BigInt`] data type.
    ///
    /// Unlike [`Self::new`], this method does not perform Montgomery reduction.
    /// Thus, this method should be used only when constructing
    /// an element from an integer that has already been put in
    /// Montgomery form.
    #[inline]
    pub const fn new_unchecked(element: BigInt<N>) -> Self {
        Self(element, PhantomData)
    }

    const fn const_is_zero(&self) -> bool {
        self.0.const_is_zero()
    }

    #[doc(hidden)]
    const fn const_neg(self) -> Self {
        if !self.const_is_zero() {
            Self::new_unchecked(Self::sub_with_borrow(&T::MODULUS, &self.0))
        } else {
            self
        }
    }

    /// Interpret a set of limbs (along with a sign) as a field element.
    /// For *internal* use only; please use the `ark_ff::MontFp` macro instead
    /// of this method
    #[doc(hidden)]
    pub const fn from_sign_and_limbs(is_positive: bool, limbs: &[u64]) -> Self {
        let mut repr = BigInt::<N>([0; N]);
        assert!(limbs.len() <= N);
        crate::const_for!((i in 0..(limbs.len())) {
            repr.0[i] = limbs[i];
        });
        let res = Self::new(repr);
        if is_positive {
            res
        } else {
            res.const_neg()
        }
    }

    const fn mul_without_cond_subtract(mut self, other: &Self) -> (bool, Self) {
        let (mut lo, mut hi) = ([0u64; N], [0u64; N]);
        crate::const_for!((i in 0..N) {
            let mut carry = 0;
            crate::const_for!((j in 0..N) {
                let k = i + j;
                if k >= N {
                    hi[k - N] = mac_with_carry!(hi[k - N], (self.0).0[i], (other.0).0[j], &mut carry);
                } else {
                    lo[k] = mac_with_carry!(lo[k], (self.0).0[i], (other.0).0[j], &mut carry);
                }
            });
            hi[i] = carry;
        });
        // Montgomery reduction
        let mut carry2 = 0;
        crate::const_for!((i in 0..N) {
            let tmp = lo[i].wrapping_mul(T::INV);
            let mut carry;
            mac!(lo[i], tmp, T::MODULUS.0[0], &mut carry);
            crate::const_for!((j in 1..N) {
                let k = i + j;
                if k >= N {
                    hi[k - N] = mac_with_carry!(hi[k - N], tmp, T::MODULUS.0[j], &mut carry);
                }  else {
                    lo[k] = mac_with_carry!(lo[k], tmp, T::MODULUS.0[j], &mut carry);
                }
            });
            hi[i] = adc!(hi[i], carry, &mut carry2);
        });

        crate::const_for!((i in 0..N) {
            (self.0).0[i] = hi[i];
        });
        (carry2 != 0, self)
    }

    const fn mul(self, other: &Self) -> Self {
        let (carry, res) = self.mul_without_cond_subtract(other);
        if T::MODULUS_HAS_SPARE_BIT {
            res.const_subtract_modulus()
        } else {
            res.const_subtract_modulus_with_carry(carry)
        }
    }

    pub fn mul_u64(self, other: u64) -> Self {
        // Stage 1: Bignum Multiplication
        // Compute c = self.0 * other. Result c has N+1 limbs.
        let (c_lo, c_hi): ([u64; N], u64) = bigint_mul_by_u64(&self.0.0, other);

        // Compute tilde_c = floor(c / R') = floor(c / 2^MODULUS_BITS)
        // If there is a spare bit, then we will need to construct tilde_c as the _concatenation_
        // of c_hi with the high spare bits of c_lo[N-1]
        // Otherwise, tilde_c = c_hi
        let tilde_c: u64 = if T::MODULUS_HAS_SPARE_BIT {
            (c_hi << T::MODULUS_NUM_SPARE_BITS) + (c_lo[N - 1] >> (64 - T::MODULUS_NUM_SPARE_BITS))
        } else {
            c_hi
        };

        // Stage 2: Barrett Reduction
        // Compute m = floor( (tilde_c * BARRETT_MU) / r ) = (tilde_c * BARRETT_MU as u128) >> 64
        let m: u64 = ((tilde_c as u128 * T::BARRETT_MU as u128) >> 64) as u64;

        // Compute m * 2p
        let (m2p_lo, m2p_hi, _m2p_carry) =
            bigint_plus_one_mul_by_u64(&T::MODULUS_TIMES_2.0, &T::MODULUS_TIMES_2.1, m);

        // Should not have a carry out of the high limb
        debug_assert!(_m2p_carry == false);

        // Compute r = c - m * 2p
        // Have either r = (c mod 2p) or r = (c mod 2p) + 2p
        let (mut r_lo, mut r_hi, _r_borrow) =
            sub_bigint_plus_one((c_lo, c_hi), (m2p_lo, m2p_hi));

        // Conditional subtraction: r := r - 2p if r >= 2p
        let cmp_2p = compare_bigint_plus_one((r_lo, r_hi), T::MODULUS_TIMES_2);
        if cmp_2p != core::cmp::Ordering::Less { // if r >= 2p
            (r_lo, r_hi, _) = sub_bigint_plus_one((r_lo, r_hi), T::MODULUS_TIMES_2);
        }
        // Now r = (r_lo, r_hi) = c mod 2p

        // Compute c' = r - p if r >= p else r_lo
        let cmp_p = compare_bigint_plus_one((r_lo, r_hi), (T::MODULUS.0, 0));
        
        // if r >= p
        let result_bigint = if cmp_p != core::cmp::Ordering::Less {
            sub_bigint_plus_one((r_lo, r_hi), (T::MODULUS.0, 0)).0
        } else {
            r_lo
        };

        Self::new_unchecked(BigInt::<N>::new(result_bigint))
    }

    const fn const_is_valid(&self) -> bool {
        crate::const_for!((i in 0..N) {
            if (self.0).0[N - i - 1] < T::MODULUS.0[N - i - 1] {
                return true
            } else if (self.0).0[N - i - 1] > T::MODULUS.0[N - i - 1] {
                return false
            }
        });
        false
    }

    #[inline]
    const fn const_subtract_modulus(mut self) -> Self {
        if !self.const_is_valid() {
            self.0 = Self::sub_with_borrow(&self.0, &T::MODULUS);
        }
        self
    }

    #[inline]
    const fn const_subtract_modulus_with_carry(mut self, carry: bool) -> Self {
        if carry || !self.const_is_valid() {
            self.0 = Self::sub_with_borrow(&self.0, &T::MODULUS);
        }
        self
    }

    const fn sub_with_borrow(a: &BigInt<N>, b: &BigInt<N>) -> BigInt<N> {
        a.const_sub_with_borrow(b).0
    }
}

#[cfg(test)]
mod test {
    use ark_std::{str::FromStr, vec::*};
    use ark_test_curves::secp256k1::Fr;
    use num_bigint::{BigInt, BigUint, Sign};
    use crate::ark_std::rand::RngCore;

    // #[test]
    // fn test_mul_u64() {
    //     use crate::{BigInt, UniformRand};
    //     use ark_test_curves::secp256k1::Fr;

    //     let mut rng = ark_std::test_rng();
    //     const N: usize = 4; // For Fr (secp256k1 scalar field)

    //     // Generate random field element
    //     let random_bigint = BigInt::<N>::rand(&mut rng);
    //     let value1_fp = Fr::new(random_bigint); // Converts to Montgomery form

    //     // Generate random u64
    //     let value2_u64 = u64::rand(&mut rng);

    //     // Compute using the optimized mul_u64
    //     let result_mul_u64 = value1_fp.mul_u64(value2_u64);

    //     // Compute the expected result using standard field multiplication
    //     // Fr::from(u64) correctly converts the u64 into Montgomery form
    //     let expected_fp = value1_fp * Fr::from(value2_u64);

    //     assert_eq!(result_mul_u64, expected_fp, "mul_u64 did not match standard field multiplication");

    //     // Test with zero
    //     let zero_fp = Fr::zero();
    //     assert_eq!(zero_fp.mul_u64(value2_u64), zero_fp, "0.mul_u64(x) != 0");
    //     assert_eq!(value1_fp.mul_u64(0), zero_fp, "x.mul_u64(0) != 0");
    // }

    #[test]
    fn test_mont_macro_correctness() {
        let (is_positive, limbs) = str_to_limbs_u64(
            "111192936301596926984056301862066282284536849596023571352007112326586892541694",
        );
        let t = Fr::from_sign_and_limbs(is_positive, &limbs);

        let result: BigUint = t.into();
        let expected = BigUint::from_str(
            "111192936301596926984056301862066282284536849596023571352007112326586892541694",
        )
        .unwrap();

        assert_eq!(result, expected);
    }

    fn str_to_limbs_u64(num: &str) -> (bool, Vec<u64>) {
        let (sign, digits) = BigInt::from_str(num)
            .expect("could not parse to bigint")
            .to_radix_le(16);
        let limbs = digits
            .chunks(16)
            .map(|chunk| {
                let mut this = 0u64;
                for (i, hexit) in chunk.iter().enumerate() {
                    this += (*hexit as u64) << (4 * i);
                }
                this
            })
            .collect::<Vec<_>>();

        let sign_is_positive = sign != Sign::Minus;
        (sign_is_positive, limbs)
    }
}

/// Multiply a N-limb big integer with a u64, producing a N+1 limb result, 
/// represented as a tuple of an array of N limbs and a u64 high limb
#[inline(always)]
fn bigint_mul_by_u64<const N: usize>(val: &[u64; N], other: u64) -> ([u64; N], u64) { 
    let (mut lo, mut hi) = ([0u64; N], 0u64);

    for i in 0..N - 1 {
        lo[i] = mac_with_carry!(lo[i], val[i], other, &mut lo[i + 1]);
    }
    lo[N - 1] = mac_with_carry!(lo[N - 1], val[N - 1], other, &mut hi);

    (lo, hi)
}

/// Multiply a N+1 limb big integer with a u64, producing a N+1 limb result,
/// represented as a tuple of an array of N limbs and a u64 high limb
/// Also returns a boolean indicating if there was a carry out of the high limb (for debugging)
#[inline(always)]
fn bigint_plus_one_mul_by_u64<const N: usize>(val_lo: &[u64; N], val_hi: &u64, other: u64) -> ([u64; N], u64, bool) {
    let (mut lo, mut hi, mut carry) = ([0u64; N], *val_hi, 0u64);

    for i in 0..N - 1 {
        lo[i] = mac_with_carry!(lo[i], val_lo[i], other, &mut lo[i + 1]);
    }
    lo[N - 1] = mac_with_carry!(lo[N - 1], val_lo[N - 1], other, &mut hi);
    hi = mac_with_carry!(hi, *val_hi, other, &mut carry);

    (lo, hi, carry != 0)
}

/// Subtract two N+1 limb big integers, represented as a tuple of an array of N limbs and a u64 high limb
#[inline(always)]
fn sub_bigint_plus_one<const N: usize>(
    a: ([u64; N], u64),
    b: ([u64; N], u64),
) -> ([u64; N], u64, bool) {
    let (mut a_lo, mut a_hi) = a;
    let (b_lo, b_hi) = b;
    let mut borrow = 0u64; // sbb uses u64 for borrow

    // Subtract low N limbs
    for i in 0..N {
        // Updates a_lo[i] in place and returns the new borrow
        borrow = fa::sbb(&mut a_lo[i], b_lo[i], borrow);
    }

    // Subtract high limb
    // Need to calculate a_hi - b_hi - borrow and get the final borrow out
    let tmp = (1u128 << 64) + (a_hi as u128) - (b_hi as u128) - (borrow as u128);
    let final_borrow_occurred = (tmp >> 64) == 0;
    a_hi = tmp as u64; // Update the high limb result

    (a_lo, a_hi, final_borrow_occurred)
}

/// Compare two N+1 limb big integers, represented as a tuple of an array of N limbs and a u64 high limb
#[inline(always)]
fn compare_bigint_plus_one<const N: usize>(a: ([u64; N], u64), b: ([u64; N], u64)) -> core::cmp::Ordering {
    if a.1 > b.1 {
        return core::cmp::Ordering::Greater;
    } else if a.1 < b.1 {
        return core::cmp::Ordering::Less;
    } else {
        a.0.cmp(&b.0)
    }
}