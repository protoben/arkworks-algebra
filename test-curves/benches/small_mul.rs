use ark_ff::UniformRand;
use ark_std::rand::{rngs::StdRng, Rng, SeedableRng};
use ark_test_curves::bn254::{Fr, FqConfig};
use criterion::{criterion_group, criterion_main, Criterion};
use ark_ff::BigInteger;
use ark_ff::BigInt;
use ark_ff::MontConfig;
use ark_ff_macros::unroll_for_loops;
use ark_ff::biginteger::arithmetic as fa;

const N: usize = 4;

// Hack: copy over the helper functions from the Montgomery backend to be benched

/// Multiply a N-limb big integer with a u64, producing a N+1 limb result,
/// represented as a tuple of a u64 low limb and an array of N high limbs.
#[unroll_for_loops(8)]
#[inline(always)]
fn bigint_mul_by_u64<const N: usize>(val: &[u64; N], other: u64) -> (u64, [u64; N]) {
    let mut result_hi = [0u64; N];
    let mut carry: u64; // Start with carry = 0

    // Calculate the full 128-bit product of the lowest limb
    let prod128_0: u128 = (val[0] as u128) * (other as u128);
    let result_lo = prod128_0 as u64; // Lowest limb of the result
    carry = (prod128_0 >> 64) as u64; // Carry into the high part

    // Iterate through the remaining limbs of the input BigInt
    for i in 1..N {
        // Calculate the full 128-bit product of the current limb and the u64 multiplier
        let prod128: u128 = (val[i] as u128) * (other as u128);

        // Add the carry from the previous limb's computation
        let sum128: u128 = prod128 + (carry as u128);

        // The lower 64 bits of the sum become the current result limb (in the high part)
        result_hi[i - 1] = sum128 as u64; // Store in result_hi[0] to result_hi[N-2]

        // The upper 64 bits of the sum become the carry for the next limb
        carry = (sum128 >> 64) as u64;
    }

    // After the loop, the final carry is the highest limb (N-th limb of the high part)
    result_hi[N - 1] = carry;

    (result_lo, result_hi)
}

/// Multiply a N+1 limb big integer (represented as low u64 and high N limbs) with a u64,
/// producing a N+1 limb result in the same format.
/// Also returns a boolean indicating if there was a carry out (overflow).
#[unroll_for_loops(8)]
#[inline(always)]
fn bigint_plus_one_mul_by_u64<const N: usize>(
    val_lo: &u64,
    val_hi: &[u64; N],
    other: u64,
) -> (u64, [u64; N], bool) {
    let mut result_hi = [0u64; N];

    // Stage 1: Multiply the low limb
    let prod_lo: u128 = (*val_lo as u128) * (other as u128);
    let result_lo = prod_lo as u64;
    let mut carry = (prod_lo >> 64) as u64;

    // Stage 2: Multiply the high N limbs
    for i in 0..N {
        let prod_hi: u128 = (val_hi[i] as u128) * (other as u128) + (carry as u128);
        result_hi[i] = prod_hi as u64;
        carry = (prod_hi >> 64) as u64;
    }

    // Final carry indicates overflow
    let overflow = carry != 0;
    (result_lo, result_hi, overflow)
}

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

/// Multiply a N-limb big integer with a u128, producing a N+2 limb result,
/// represented as a tuple of an array of 2 low limbs and an array of N high limbs.
#[unroll_for_loops(8)]
#[inline(always)]
fn bigint_mul_by_u128<const N: usize>(val: &BigInt<N>, other: u128) -> ([u64; 2], [u64; N]) {
    let other_lo = other as u64;
    let other_hi = (other >> 64) as u64;

    // Compute partial products
    // p1 = val * other_lo -> (N+1) limbs: (p1_lo: u64, p1_hi: [u64; N])
    let (p1_lo, p1_hi) = bigint_mul_by_u64(&val.0, other_lo);
    // p2 = val * other_hi -> (N+1) limbs: (p2_lo: u64, p2_hi: [u64; N])
    let (p2_lo, p2_hi) = bigint_mul_by_u64(&val.0, other_hi);

    // Calculate the final result r = p1 + (p2 << 64) limb by limb.
    // p1       : [p1_lo, p1_hi[0], ..., p1_hi[N-1]]
    // p2 << 64 : [0, p2_lo, p2_hi[0], ..., p2_hi[N-1]]
    // Sum (r)  : [r_lo[0], r_lo[1], r_hi[0], ..., r_hi[N-1]] (N+2 limbs)

    let mut r_lo = [0u64; 2];
    let mut r_hi = [0u64; N];
    let mut carry: u64 = 0;

    // r_lo[0] = p1_lo + 0 + carry (carry is initially 0)
    r_lo[0] = p1_lo;
    // carry = 0; // Initial carry is 0

    // Calculate r_lo[1] = p1_hi[0] + p2_lo + carry (limb 1)
    r_lo[1] = p1_hi[0]; // Initialize with p1 limb
    carry = fa::adc(&mut r_lo[1], p2_lo, carry); // Add p2 limb and carry

    // Calculate r_hi[0] to r_hi[N-1] (limbs 2 to N+1)
    for i in 0..N {
        let p1_limb = if i + 1 < N { p1_hi[i + 1] } else { 0 }; // Limb p1[i+2]
        let p2_limb = p2_hi[i]; // Limb p2[i+1]

        // r_hi[i] = p1_limb + p2_limb + carry
        r_hi[i] = p1_limb; // Initialize with p1 limb
        carry = fa::adc(&mut r_hi[i], p2_limb, carry); // Add p2 limb and carry
    }

    // The final carry MUST be zero for the result to fit in N+2 limbs.
    debug_assert!(carry == 0, "Overflow in bigint_mul_by_u128");

    (r_lo, r_hi)
}

/// Helper function to perform Barrett reduction from N+1 limbs to N limbs.
/// Input `c` is represented as `(u64, [u64; N])`.
/// Output is the N-limb result `[u64; N]`.
#[unroll_for_loops(4)]
#[inline(always)]
fn barrett_reduce_nplus1_to_n<T: MontConfig<N>, const N: usize>(c: (u64, [u64; N])) -> [u64; N] {
    let (c_lo, c_hi) = c; // c_lo is the lowest limb, c_hi holds the top N limbs

    // Compute tilde_c = floor(c / R') = floor(c / 2^MODULUS_BITS)
    // This involves the top two limbs of the N+1 limb number `c`.
    // The highest limb is c_hi[N-1]. The second highest is c_hi[N-2].
    // Assume that `N >= 1`
    let tilde_c: u64 = if T::MODULUS_HAS_SPARE_BIT {
        let high_limb = c_hi[N-1];
        let second_high_limb = if N > 1 { c_hi[N-2] } else { c_lo }; // Use c_lo if N=1
        (high_limb << T::MODULUS_NUM_SPARE_BITS) + (second_high_limb >> (64 - T::MODULUS_NUM_SPARE_BITS))
    } else {
        c_hi[N-1] // If no spare bits, tilde_c is just the highest limb
    };

    // Estimate m = floor( (tilde_c * BARRETT_MU) / r )
    let m: u64 = ((tilde_c as u128 * T::BARRETT_MU as u128) >> 64) as u64;

    // Compute m * 2p (N+1 limbs)
    let (m2p_lo, m2p_hi, _m2p_carry) = bigint_plus_one_mul_by_u64::<N>(
        &T::MODULUS_TIMES_2_NPLUS1.0, // Low limb of 2p
        &T::MODULUS_TIMES_2_NPLUS1.1, // High N limbs of 2p
        m,
    );
    debug_assert!(_m2p_carry == false, "Overflow calculating m * 2p");
    let m_times_2p = (m2p_lo, m2p_hi);

    // Compute r_tmp = c - m * 2p (N+1 limbs)
    let (r_tmp, _) = sub_bigint_plus_one(c, m_times_2p); // r_tmp = (r_tmp_lo, r_tmp_hi)

    // Final conditional subtractions (optimized based on spare bits)
    let final_limbs: [u64; N];

    if T::MODULUS_NUM_SPARE_BITS >= 1 {
        // Case S >= 1: 2P fits in N limbs (T::MODULUS_TIMES_2_NPLUS1.1[N-1] == 0)
        let mut p2_n_limbs_arr = [0u64; N];
        p2_n_limbs_arr[0] = T::MODULUS_TIMES_2_NPLUS1.0;
        if N > 1 {
            p2_n_limbs_arr[1..N].copy_from_slice(&T::MODULUS_TIMES_2_NPLUS1.1[0..(N-1)]);
        }
        let p2_n_limbs = BigInt::<N>(p2_n_limbs_arr);

        if T::MODULUS_NUM_SPARE_BITS >= 2 {
            // Optimization for S >= 2: r_tmp = c - m*2p < 4p already fits in N limbs
            debug_assert!(r_tmp.1[N - 1] == 0, "High limb of r_tmp should be 0 for S >= 2 before subtractions");

            let mut r_n_limbs_arr = [0u64; N];
            r_n_limbs_arr[0] = r_tmp.0; // Low limb
            if N > 1 {
                r_n_limbs_arr[1..N].copy_from_slice(&r_tmp.1[0..(N - 1)]); // Lower N-1 limbs of high part
            }
            let mut r_n_limbs = BigInt::<N>(r_n_limbs_arr);

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
                    temp_r_n_limbs_arr[1..N].copy_from_slice(&sub_res.1[0..(N-1)]);
                 }
            } else {
                // r_tmp was already < 2P.
                // If r_geq_2p is false, r_tmp < 2P. Since 2P fits in N limbs, r_tmp must also fit.
                debug_assert!(r_tmp.1[N - 1] == 0, "High limb must be 0 if r < 2P and S=1");
                temp_r_n_limbs_arr[0] = r_tmp.0;
                if N > 1 {
                    temp_r_n_limbs_arr[1..N].copy_from_slice(&r_tmp.1[0..(N - 1)]);
                }
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
             debug_assert!(sub_res.1[N-1] == 0, "High limb must be 0 after P subtraction when S=0");
             debug_assert!(!sub_borrow, "Borrow should not occur when subtracting P for S=0");
             let mut r_n_limbs_arr = [0u64; N];
             r_n_limbs_arr[0] = sub_res.0;
             if N > 1 {
                r_n_limbs_arr[1..N].copy_from_slice(&sub_res.1[0..(N-1)]);
             }
             final_limbs = r_n_limbs_arr;
        } else {
             // r was already < P
             debug_assert!(current_r.1[N - 1] == 0, "High limb must be 0 if r < P and S=0");
             let mut r_n_limbs_arr = [0u64; N];
             r_n_limbs_arr[0] = current_r.0;
             if N > 1 {
                r_n_limbs_arr[1..N].copy_from_slice(&current_r.1[0..(N - 1)]);
             }
             final_limbs = r_n_limbs_arr;
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

    group.bench_function("mul_u64 (full)", |bench| {
        let mut i = 0;
        bench.iter(|| {
            i = (i + 1) % SAMPLES;
            criterion::black_box(a_s[i].mul_u64(b_u64_s[i]))
        })
    });

    group.bench_function("bigint_mul_by_u64 (helper)", |bench| {
        let mut i = 0;
        bench.iter(|| {
            i = (i + 1) % SAMPLES;
            criterion::black_box(bigint_mul_by_u64::<N>(&a_limbs_s[i], b_u64_s[i]))
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

    group.bench_function("bigint_mul_by_u128 (helper)", |bench| {
        let mut i = 0;
        bench.iter(|| {
            i = (i + 1) % SAMPLES;
            let val_bigint = BigInt::<N>(a_limbs_s[i]);
            criterion::black_box(bigint_mul_by_u128::<N>(&val_bigint, b_u128_s[i]))
        })
    });

    // group.bench_function("mul_i128", |bench| {
    //     let mut i = 0;
    //     bench.iter(|| {
    //         i = (i + 1) % SAMPLES;
    //         criterion::black_box(a_s[i].mul_i128(b_i128_s[i]))
    //     })
    // });

    group.bench_function("barrett_reduce_nplus1_to_n (helper)", |bench| {
        let mut i = 0;
        bench.iter(|| {
            i = (i + 1) % SAMPLES;
            criterion::black_box(
                barrett_reduce_nplus1_to_n::<FqConfig, N>(barrett_inputs[i])
            )
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