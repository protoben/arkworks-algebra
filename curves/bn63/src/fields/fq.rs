use ark_ff::fields::{Fp64, MontBackend, MontConfig};

#[derive(MontConfig)]
#[modulus = "6094503336465856489"]
#[generator = "26"]
pub struct FqConfig;
pub type Fq = Fp64<MontBackend<FqConfig, 1>>;

#[test]
fn foo() {
    const N: ark_ff::Fp<ark_ff::MontBackend<FqConfig, 1>, 1> =
        <FqConfig as MontConfig<1>>::TWO_ADIC_ROOT_OF_UNITY;
    let sz =
        <Fq as ark_serialize::CanonicalSerialize>::serialized_size(&N, ark_serialize::Compress::No);
    assert_eq!(sz, sz);
}
