use ark_ff::{Field, UniformRand};
use ark_std::rand::{rngs::StdRng, Rng, SeedableRng};
use ark_test_curves::secp256k1::Fr;
use criterion::{criterion_group, criterion_main, Criterion};

fn mul_u64_bench(c: &mut Criterion) {
    const SAMPLES: usize = 1000;
    // Use a fixed seed for reproducibility
    let mut rng = StdRng::seed_from_u64(0u64);

    let a_s = (0..SAMPLES)
        .map(|_| Fr::rand(&mut rng))
        .collect::<Vec<_>>();
    let b_s = (0..SAMPLES)
        .map(|_| rng.gen::<u64>())
        .collect::<Vec<_>>();
    // Convert u64 to Fr for standard multiplication benchmark
    let b_fr_s = b_s.iter().map(|&b| Fr::from(b)).collect::<Vec<_>>();

    // Generate another set of random Fr elements for addition
    let c_s = (0..SAMPLES)
        .map(|_| Fr::rand(&mut rng))
        .collect::<Vec<_>>();

    let mut group = c.benchmark_group("Fr Arithmetic Comparison");

    group.bench_function("mul_u64", |bench| {
        let mut i = 0;
        bench.iter(|| {
            i = (i + 1) % SAMPLES;
            // Make sure the computation is not optimized away
            criterion::black_box(a_s[i].mul_u64(b_s[i]))
        })
    });

    group.bench_function("standard mul (Fr * Fr::from(u64))", |bench| {
        let mut i = 0;
        bench.iter(|| {
            i = (i + 1) % SAMPLES;
            // Make sure the computation is not optimized away
            criterion::black_box(a_s[i] * b_fr_s[i])
        })
    });

    group.bench_function("Addition (Fr + Fr)", |bench| {
        let mut i = 0;
        bench.iter(|| {
            i = (i + 1) % SAMPLES;
            // Make sure the computation is not optimized away
            criterion::black_box(a_s[i] + c_s[i])
        })
    });

    group.bench_function("Square (Fr)", |bench| {
        let mut i = 0;
        bench.iter(|| {
            i = (i + 1) % SAMPLES;
            // Make sure the computation is not optimized away
            criterion::black_box(a_s[i].square())
        })
    });

    group.finish();
}

criterion_group!(benches, mul_u64_bench);
criterion_main!(benches); 