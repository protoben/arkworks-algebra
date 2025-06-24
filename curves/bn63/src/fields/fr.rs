use ark_ff::fields::{Fp64, MontBackend, MontConfig};

#[derive(MontConfig)]
#[modulus = "6094503333997212553"]
#[generator = "5"]
pub struct FrConfig;
pub type Fr = Fp64<MontBackend<FrConfig, 1>>;
