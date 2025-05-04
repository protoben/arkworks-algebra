use ark_ff::UniformRand;
use ark_std::rand::{rngs::StdRng, Rng, SeedableRng};
use ark_test_curves::secp256k1::{Fr, FqConfig, FrConfig};
use criterion::{criterion_group, criterion_main, Criterion};
use ark_ff::BigInteger;
use ark_ff::BigInt;
use ark_ff::MontConfig;
use ark_ff_macros::unroll_for_loops;

const N: usize = 4;

// Hack: copy over the helper functions from the Montgomery backend to be benched

/// Subtract two N+1 limb big integers (represented as low u64 and high N limbs).
/// Returns the N+1 limb result and a boolean indicating if a borrow occurred.
#[unroll_for_loops(8)]
#[inline(always)]
fn sub_bigint_plus_one<const N: usize>(
    a: (u64, [u64; N]),
    b: (u64, [u64; N]),
) -> ((u64, [u64; N]), bool) {
    let (mut a_lo, mut a_hi) = a;
    let (b_lo, b_hi) = b;
    let mut borrow_bool: bool = false;

    // Subtract low u64 limb using overflowing_sub
    // Equivalent to (result, borrow_out) = a_lo.borrowing_sub(b_lo, borrow_bool)
    let (diff_lo, borrow1_lo) = a_lo.overflowing_sub(b_lo);
    let (res_lo, borrow2_lo) = diff_lo.overflowing_sub(borrow_bool as u64);
    a_lo = res_lo;
    borrow_bool = borrow1_lo || borrow2_lo; // Update borrow for next limb

    // Subtract high N limbs
    for i in 0..N {
        // Equivalent to (res_hi, borrow_out_hi) = a_hi[i].borrowing_sub(b_hi[i], borrow_bool)
        let (diff_hi, borrow1_hi) = a_hi[i].overflowing_sub(b_hi[i]);
        let (res_hi, borrow2_hi) = diff_hi.overflowing_sub(borrow_bool as u64);
        a_hi[i] = res_hi;
        borrow_bool = borrow1_hi || borrow2_hi; // Update borrow for next iteration
    }

    // Final borrow indicates if the result is negative (b > a)
    let final_borrow_occurred = borrow_bool;

    ((a_lo, a_hi), final_borrow_occurred)
}

/// Compare two N+1 limb big integers (represented as low u64 and high N limbs).
#[unroll_for_loops(8)]
#[inline(always)]
fn compare_bigint_plus_one<const N: usize>(
    a: (u64, [u64; N]),
    b: (u64, [u64; N]),
) -> core::cmp::Ordering {
    // Compare high N limbs first, from most significant (N-1) down to 0
    for i in (0..N).rev() {
        if a.1[i] > b.1[i] {
            return core::cmp::Ordering::Greater;
        } else if a.1[i] < b.1[i] {
            return core::cmp::Ordering::Less;
        }
    }
    // High limbs are equal, compare the low u64 limb
    if a.0 > b.0 {
        return core::cmp::Ordering::Greater;
    } else if a.0 < b.0 {
        return core::cmp::Ordering::Less;
    }
    // All limbs are equal
    return core::cmp::Ordering::Equal;
}

/// Helper to extract N limbs from an N+1 limb value, asserting the high limb is zero.
#[unroll_for_loops(4)]
#[inline(always)]
fn get_n_limbs_from_n_plus_one<const N: usize>(val: (u64, [u64; N])) -> BigInt<N> {
    debug_assert!(val.1[N-1] == 0, "High limb must be zero to extract N limbs");
    let mut limbs = [0u64; N];
    limbs[0] = val.0;
    if N > 1 {
        for i in 0..N-1 {
            limbs[i + 1] = val.1[i];
        }
    }
    BigInt::<N>(limbs)
}

/// Original conditional subtraction logic for Barrett reduction.
/// Takes an N+1 limb intermediate result `r_tmp` and returns the N-limb final result.
/// Performs up to 2 conditional subtractions, instead of 1 in the optimized version.
#[inline(always)]
fn _barrett_cond_subtract<T: MontConfig<N>, const N: usize>(r_tmp: (u64, [u64; N])) -> [u64; N] {
     // Final conditional subtractions (optimized based on spare bits)
    let final_limbs: [u64; N];

    if T::MODULUS_NUM_SPARE_BITS >= 1 {
        // Case S >= 1: 2P fits in N limbs (T::MODULUS_TIMES_2_NPLUS1.1[N-1] == 0)
        let p2_n_limbs = T::MODULUS_TIMES_2_N;

        if T::MODULUS_NUM_SPARE_BITS >= 2 {
            // Optimization for S >= 2: r_tmp = c - m*2p < 4p already fits in N limbs
            let mut r_n_limbs = get_n_limbs_from_n_plus_one::<N>(r_tmp);

            // Conditional subtraction 1 (if r >= 2P) using N limbs
            if r_n_limbs >= p2_n_limbs {
                r_n_limbs.sub_with_borrow(&p2_n_limbs); // Ignore borrow
            }
            // Conditional subtraction 2 (if r >= P) using N limbs
            if r_n_limbs >= T::MODULUS {
                r_n_limbs.sub_with_borrow(&T::MODULUS); // Ignore borrow
            }
            final_limbs = r_n_limbs.0;
        } else {
            // Case S == 1: r_tmp = c - m*2p might temporarily exceed N limbs
            let r_geq_2p = compare_bigint_plus_one(r_tmp, T::MODULUS_TIMES_2_NPLUS1) != core::cmp::Ordering::Less;
            let mut temp_r_n_limbs_arr = [0u64; N];

            if r_geq_2p {
                // Since 2P fits in N limbs, subtracting it from r_tmp might use the high limb r_tmp.1[N-1]
                let (sub_res, sub_borrow) = sub_bigint_plus_one(r_tmp, T::MODULUS_TIMES_2_NPLUS1);
                // After subtracting 2P, the result MUST fit in N limbs
                debug_assert!(sub_res.1[N-1] == 0, "High limb must be 0 after 2P subtraction when S=1");
                debug_assert!(!sub_borrow, "Borrow should not occur when subtracting 2P for S=1");
                temp_r_n_limbs_arr[0] = sub_res.0;
                 if N > 1 {
                    // Use loop for potential const compatibility
                    let mut i = 0;
                    while i < N - 1 {
                        temp_r_n_limbs_arr[i + 1] = sub_res.1[i];
                        i += 1;
                    }
                 }
            } else {
                // r_tmp was already < 2P.
                // If r_geq_2p is false, r_tmp < 2P. Since 2P fits in N limbs, r_tmp must also fit.
                 temp_r_n_limbs_arr = get_n_limbs_from_n_plus_one::<N>(r_tmp).0;
            }
            let mut r_n_limbs = BigInt::<N>(temp_r_n_limbs_arr);

            // Conditional subtraction 2 (if r >= P) using N limbs
            // At this point, r_n_limbs holds the value r < 2P fitting in N limbs
            if r_n_limbs >= T::MODULUS {
                r_n_limbs.sub_with_borrow(&T::MODULUS); // Ignore borrow
            }
            final_limbs = r_n_limbs.0;
        }
    } else {
        // Case S == 0: Use (N+1)-limb helpers throughout
        let mut current_r = r_tmp; // (u64, [u64; N])

        // Conditional subtraction 1: if r >= 2p
        if compare_bigint_plus_one(current_r, T::MODULUS_TIMES_2_NPLUS1) != core::cmp::Ordering::Less {
            current_r = sub_bigint_plus_one(current_r, T::MODULUS_TIMES_2_NPLUS1).0;
        }
        // Now current_r = c mod 2p, represented as (lo, hi)

        // Conditional subtraction 2: if r >= p
        if compare_bigint_plus_one(current_r, T::MODULUS_NPLUS1) != core::cmp::Ordering::Less { // if r >= p
             let (sub_res, sub_borrow) = sub_bigint_plus_one(current_r, T::MODULUS_NPLUS1);
             // Result MUST fit in N limbs now
             final_limbs = get_n_limbs_from_n_plus_one::<N>(sub_res).0;
             debug_assert!(!sub_borrow, "Borrow should not occur when subtracting P for S=0");
        } else {
             // r was already < P
             final_limbs = get_n_limbs_from_n_plus_one::<N>(current_r).0;
        }
    }
    final_limbs
}

/// Optimized conditional subtraction logic for Barrett reduction using comparisons.
/// Takes an N+1 limb intermediate result `r_tmp` and returns the N-limb final result.
#[unroll_for_loops(4)]
#[inline(always)]
fn barrett_cond_subtract_optimized<T: MontConfig<N>, const N: usize>(r_tmp: (u64, [u64; N])) -> [u64; N] {
    let final_limbs: [u64; N];
    let mut r_n = get_n_limbs_from_n_plus_one::<N>(r_tmp); // Make r_n mutable

    // Compare with 2p (N+1 limbs)
    let compare_2p = if T::MODULUS_NUM_SPARE_BITS >= 2 {
        compare_bigint_plus_one(r_tmp, T::MODULUS_TIMES_2_NPLUS1)
    } else {
        r_n.cmp(&T::MODULUS_TIMES_2_N)
    };

    if compare_2p != core::cmp::Ordering::Less { // r_tmp >= 2p
        // Compare with 3p (N+1 limbs)
        let compare_3p = if T::MODULUS_NUM_SPARE_BITS >= 2 {
            compare_bigint_plus_one(r_tmp, T::MODULUS_TIMES_3_NPLUS1)
        } else {
            r_n.cmp(&T::MODULUS_TIMES_3_N)
        };

        if compare_3p != core::cmp::Ordering::Less { // r_tmp >= 3p
            // Subtract 3p
            if T::MODULUS_NUM_SPARE_BITS >= 2 { // 3p fits in N limbs
                let borrow_n = r_n.sub_with_borrow(&T::MODULUS_TIMES_3_N); // Call on mutable r_n, assign return to borrow_n
                debug_assert!(!borrow_n, "Borrow should not occur subtracting 3p (S>=2)");
                final_limbs = r_n.0; // Use r_n directly
            } else { // Use N+1 limb subtraction
                let (res_n1, borrow_n1) = sub_bigint_plus_one(r_tmp, T::MODULUS_TIMES_3_NPLUS1);
                debug_assert!(!borrow_n1, "Borrow should not occur subtracting 3p (S<2)");
                final_limbs = get_n_limbs_from_n_plus_one::<N>(res_n1).0;
            }
        } else { // 2p <= r_tmp < 3p
            // Subtract 2p
            if T::MODULUS_NUM_SPARE_BITS >= 1 { // 2p fits in N limbs
                let borrow_n = r_n.sub_with_borrow(&T::MODULUS_TIMES_2_N); // Call on mutable r_n, assign return to borrow_n
                 debug_assert!(!borrow_n, "Borrow should not occur subtracting 2p (S>=1)");
                final_limbs = r_n.0; // Use r_n directly
            } else { // s == 0, use N+1 limb subtraction
                let (res_n1, borrow_n1) = sub_bigint_plus_one(r_tmp, T::MODULUS_TIMES_2_NPLUS1);
                debug_assert!(!borrow_n1, "Borrow should not occur subtracting 2p (S=0)");
                final_limbs = get_n_limbs_from_n_plus_one::<N>(res_n1).0;
            }
        }
    } else { // r_tmp < 2p
        // Compare with p (N+1 limbs)
        let compare_p = if T::MODULUS_NUM_SPARE_BITS >= 1 {
            compare_bigint_plus_one(r_tmp, T::MODULUS_NPLUS1)
        } else {
            r_n.cmp(&T::MODULUS)
        };

        if compare_p != core::cmp::Ordering::Less { // p <= r_tmp < 2p
            // Subtract p
            // P always fits in N limbs if S>=1. If S=0, use N+1.
            if T::MODULUS_NUM_SPARE_BITS >= 1 { // N limb subtraction suffices
                let borrow_n = r_n.sub_with_borrow(&T::MODULUS); // Call on mutable r_n, assign return to borrow_n
                debug_assert!(!borrow_n, "Borrow should not occur subtracting p (S>=1)");
                final_limbs = r_n.0; // Use r_n directly
            } else { // s == 0, use N+1 limb subtraction
                 let (res_n1, borrow_n1) = sub_bigint_plus_one(r_tmp, T::MODULUS_NPLUS1);
                 debug_assert!(!borrow_n1, "Borrow should not occur subtracting p (S=0)");
                 final_limbs = get_n_limbs_from_n_plus_one::<N>(res_n1).0;
            }
        } else { // r_tmp < p
            // Subtract 0 (No-op)
            // Result must already fit in N limbs
            final_limbs = get_n_limbs_from_n_plus_one::<N>(r_tmp).0;
        }
    }
    final_limbs
}

fn mul_small_bench(c: &mut Criterion) {
    const SAMPLES: usize = 1000;
    // Use a fixed seed for reproducibility
    let mut rng = StdRng::seed_from_u64(0u64);

    let a_s = (0..SAMPLES)
        .map(|_| Fr::rand(&mut rng))
        .collect::<Vec<_>>();
    let a_limbs_s = a_s.iter().map(|a| a.0.0).collect::<Vec<_>>();

    let b_u64_s = (0..SAMPLES)
        .map(|_| rng.gen::<u64>())
        .collect::<Vec<_>>();
    // Convert u64 to Fr for standard multiplication benchmark
    let b_fr_s = b_u64_s.iter().map(|&b| Fr::from(b)).collect::<Vec<_>>();

    let b_u64_as_u128_s = b_u64_s.iter().map(|&b| b as u128).collect::<Vec<_>>();

    let b_i64_s = (0..SAMPLES)
        .map(|_| rng.gen::<i64>())
        .collect::<Vec<_>>();

    let b_u128_s = (0..SAMPLES)
        .map(|_| rng.gen::<u128>())
        .collect::<Vec<_>>();

    let b_i128_s = (0..SAMPLES)
        .map(|_| rng.gen::<i128>())
        .collect::<Vec<_>>();

    // Generate another set of random Fr elements for addition
    let c_s = (0..SAMPLES)
        .map(|_| Fr::rand(&mut rng))
        .collect::<Vec<_>>();

    let barrett_inputs = (0..SAMPLES)
        .map(|_| {
            let lo = rng.gen::<u64>();
            let mut hi = [0u64; N];
            for i in 0..N {
                hi[i] = rng.gen::<u64>();
            }
            (lo, hi)
        })
        .collect::<Vec<_>>();

    let mut group = c.benchmark_group("Fr Arithmetic Comparison");

    // --- Conditional Subtraction Benchmarks ---
    group.bench_function("barrett_cond_subtract", |bench| {
        let mut i = 0;
        bench.iter(|| {
            i = (i + 1) % SAMPLES;
            // Call using the FrConfig type explicitly
            criterion::black_box(_barrett_cond_subtract::<FrConfig, N>(barrett_inputs[i]))
        })
    });

    group.bench_function("barrett_cond_subtract_optimized", |bench| {
        let mut i = 0;
        bench.iter(|| {
            i = (i + 1) % SAMPLES;
            // Call using the FrConfig type explicitly
            criterion::black_box(barrett_cond_subtract_optimized::<FrConfig, N>(barrett_inputs[i]))
        })
    });
    // --- End Conditional Subtraction Benchmarks ---

    group.bench_function("mul_u64 (full)", |bench| {
        let mut i = 0;
        bench.iter(|| {
            i = (i + 1) % SAMPLES;
            criterion::black_box(a_s[i].mul_u64(b_u64_s[i]))
        })
    });

    // Note: results might be worse than in real applications due to branch prediction being wrong
    // 50% of the time
    group.bench_function("mul_u128 (full)", |bench| {
        let mut i = 0;
        bench.iter(|| {
            i = (i + 1) % SAMPLES;
            criterion::black_box(a_s[i].mul_u128(b_u128_s[i]))
        })
    });

    group.bench_function("mul_i128", |bench| {
        let mut i = 0;
        bench.iter(|| {
            i = (i + 1) % SAMPLES;
            criterion::black_box(a_s[i].mul_i128(b_i128_s[i]))
        })
    });

    group.bench_function("standard mul (Fr * Fr)", |bench| {
        let mut i = 0;
        bench.iter(|| {
            i = (i + 1) % SAMPLES;
            criterion::black_box(a_s[i] * b_fr_s[i])
        })
    });

    // Benchmark mul_u128 specifically with inputs known to fit in u64
    group.bench_function("mul_u128 (u64 inputs)", |bench| {
        let mut i = 0;
        bench.iter(|| {
            i = (i + 1) % SAMPLES;
            // Call mul_u128 but provide a u64 input cast to u128
            criterion::black_box(a_s[i].mul_u128(b_u64_as_u128_s[i]))
        })
    });

    // Benchmark the auxiliary function directly (assuming it's made public)
    // Note: Requires mul_u128_aux to be pub in montgomery_backend.rs
    // Need to import it if not already done via wildcard/specific import
    // Let's assume it's accessible via a_s[i].mul_u128_aux(...) for now
    group.bench_function("mul_u128_aux (u128 inputs)", |bench| {
        let mut i = 0;
        bench.iter(|| {
            i = (i + 1) % SAMPLES;
            criterion::black_box(a_s[i].mul_u128_aux(b_u128_s[i]))
        })
    });

    group.bench_function("Addition (Fr + Fr)", |bench| {
        let mut i = 0;
        bench.iter(|| {
            i = (i + 1) % SAMPLES;
            criterion::black_box(a_s[i] + c_s[i])
        })
    });

    group.finish();
}

criterion_group!(benches, mul_small_bench);
criterion_main!(benches); 