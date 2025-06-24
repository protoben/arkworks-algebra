use ark_ff::fields::{Fp64, MontBackend, MontConfig};

#[derive(MontConfig)]
#[modulus = "6094503336465856489"]
#[generator = "26"]
pub struct FqConfig;
pub type Fq = Fp64<MontBackend<FqConfig, 1>>;
