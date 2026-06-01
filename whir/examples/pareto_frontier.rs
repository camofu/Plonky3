//! FRI vs WHIR Pareto frontier: prover cost vs argument size.
//!
//! # What this traces
//!
//! PR #1607 measured FRI and WHIR at a single (default) parameterisation and
//! reported one point each. The interesting comparison is the *frontier*: WHIR
//! exposes more knobs (per-round inverse-rate schedule + folding factor), so it
//! can slide along a prover-cost-vs-argument-size curve that vanilla FRI cannot.
//!
//! This example sweeps each protocol over its own knobs, holding the *claim*
//! fixed (the §1.1 univariate/multilinear bridge of PR #1607: 2^m elements as
//! 256 polynomials of size 2^(m-8), opened at one common point), and plots two
//! views with the same y-axis (argument size = postcard proof bytes):
//!
//!   - x = total committed oracle length (Σ codeword sizes) — analytic proxy for
//!     prover work (LDE FFTs + Merkle hashing). This is the "proof-length" axis.
//!   - x = measured prover wall-clock (commit + open).
//!
//! The first view is the theoretical frontier where WHIR should envelope FRI.
//! The second shows how WHIR's fixed per-round sumcheck overhead distorts that
//! envelope near the cheap-prover corner (FRI does no sumcheck).
//!
//! # Knobs swept
//!
//! WHIR: folding factor `k`, rate-schedule slope `Δ` (per-round log-inv-rate
//! increment, valid range 0..=k), and starting log-inv-rate. `Δ=0` → flat rate,
//! oracles decay by 2^k (cheap prover, many queries); `Δ=k-1` → the PR default
//! decay-by-2; `Δ=k` → constant-size oracles (dearest prover, fewest queries).
//!
//! FRI: `log_blowup` (rate) and `max_log_arity`. `num_queries` is derived as the
//! minimum reaching the target soundness under the same capacity-regime formula.
//!
//! # Run
//!
//! ```bash
//! cargo run -p p3-whir --release --example pareto_frontier
//! # optional: PARETO_M=22 to match the PR's message size (slower).
//! ```
//!
//! Outputs `pareto_frontier.svg` and `pareto_data.csv` in the crate root.

use std::fmt::Write as _;
use std::panic::{self, AssertUnwindSafe};
use std::time::Instant;

use p3_challenger::{
    CanObserve, CanSampleUniformBits, DuplexChallenger, FieldChallenger, GrindingChallenger,
};
use p3_commit::{BatchOpening, ExtensionMmcs, Mmcs, MultilinearPcs, Pcs};
use p3_dft::Radix2DFTSmallBatch;
use p3_field::Field;
use p3_field::coset::TwoAdicMultiplicativeCoset;
use p3_field::extension::BinomialExtensionField;
use p3_fri::{FriParameters, TwoAdicFriPcs};
use p3_koala_bear::{
    KoalaBear, Poseidon1KoalaBear, default_koalabear_poseidon1_16, default_koalabear_poseidon1_24,
};
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_multilinear_util::poly::Poly;
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use p3_whir::fiat_shamir::domain_separator::DomainSeparator;
use p3_whir::parameters::{FoldingFactor, ProtocolParameters, SecurityAssumption, WhirConfig};
use p3_whir::pcs::proof::PcsProof;
use p3_whir::pcs::prover::WhirProver;
use p3_whir::sumcheck::layout::{Layout, SuffixProver, Table, Witness};
use p3_whir::sumcheck::{OpeningProtocol, TableShape, TableSpec};
use rand::SeedableRng;
use rand::rngs::SmallRng;

// ---------------------------------------------------------------------------
// Shared substrate (matched on both sides), lifted from benches/fri_vs_whir.rs.
// ---------------------------------------------------------------------------

/// Base field used for the committed message.
type F = KoalaBear;
/// Challenge field used for Fiat-Shamir challenges and out-of-domain samples.
type EF = BinomialExtensionField<F, 4>;
/// DFT backend used by both protocols.
type Dft = Radix2DFTSmallBatch<F>;

/// Target soundness in bits for both protocols (capacity-regime conjecture).
const SECURITY_LEVEL: usize = 100;
/// Proof-of-work grinding budget, shared by both protocols.
const POW_BITS: usize = 20;
/// Common batching log: 2^8 = 256 polynomials opened at one common point.
const LOG_FRI_BATCH_WIDTH: usize = 8;
/// RNG seed making the sweep deterministic across runs.
const BENCH_SEED: u64 = 0xA17_5C0DE;
/// FRI final-polynomial truncation log (open the final poly at one point).
const FRI_LOG_FINAL_POLY_LEN: usize = 0;

/// Minimum FRI query count reaching `SECURITY_LEVEL` under the capacity formula
/// `log_blowup * queries + query_pow >= security_level`.
fn fri_min_queries(log_blowup: usize, query_pow_bits: usize) -> usize {
    SECURITY_LEVEL
        .saturating_sub(query_pow_bits)
        .div_ceil(log_blowup)
        .max(1)
}

trait WhirMmcs: Mmcs<F> + Clone + Send + Sync {}
impl<T: Mmcs<F> + Clone + Send + Sync> WhirMmcs for T {}

trait WhirChallenger<MT: Mmcs<F>>:
    FieldChallenger<F>
    + GrindingChallenger<Witness = F>
    + CanSampleUniformBits<F>
    + CanObserve<MT::Commitment>
    + Clone
{
}
impl<MT: Mmcs<F>, T> WhirChallenger<MT> for T where
    T: FieldChallenger<F>
        + GrindingChallenger<Witness = F>
        + CanSampleUniformBits<F>
        + CanObserve<MT::Commitment>
        + Clone
{
}

/// Layout binding mode: suffix binds trailing variables first, matching a single
/// common opening point shared across all columns.
type WhirLayout = SuffixProver<F, EF>;
type WhirPcsTy<MT, Ch> = WhirProver<EF, F, Dft, MT, Ch, WhirLayout>;
type WhirProofTy<MT> = PcsProof<F, EF, MT>;
type WhirCommitTy<MT> = <MT as Mmcs<F>>::Commitment;

// ---------------------------------------------------------------------------
// WHIR rig — parameterised over (k, delta, starting_log_inv_rate).
// ---------------------------------------------------------------------------

struct WhirRig<MT: Mmcs<F>, Ch> {
    pcs: WhirPcsTy<MT, Ch>,
    witness: Witness<F>,
    protocol: OpeningProtocol,
    domain_separator: DomainSeparator<EF, F>,
    challenger: Ch,
}

/// One swept WHIR data point: the runnable rig plus analytic metadata.
struct WhirBuilt<MT: Mmcs<F>, Ch> {
    rig: WhirRig<MT, Ch>,
    /// Total committed oracle length in field elements (prover-cost proxy).
    oracle_len: u128,
    /// Per-round query counts followed by the final-phase query count.
    queries: Vec<usize>,
}

/// Build a WHIR rig for `(m, log_width)` at folding factor `k`, rate-schedule
/// slope `delta`, and starting log-inv-rate `starting`.
///
/// Returns `None` if the resulting `WhirConfig` is invalid (panics during
/// construction, e.g. a rate step that would grow the RS domain or a two-adicity
/// overflow) or cannot reach the soundness target within the PoW budget.
fn whir_build<MT, Ch, const DIGEST_ELEMS: usize>(
    num_variables: usize,
    log_width: usize,
    k: usize,
    delta: usize,
    starting: usize,
    mmcs: MT,
    base_challenger: Ch,
) -> Option<WhirBuilt<MT, Ch>>
where
    MT: WhirMmcs,
    Ch: WhirChallenger<MT>,
{
    let log_height = num_variables - log_width;
    let width = 1 << log_width;
    let folding_factor = FoldingFactor::Constant(k);

    // One inverse-rate entry per intermediate round: rate += delta each round,
    // starting from `starting` (the rate of the first committed codeword).
    let (num_rounds, _) = folding_factor.compute_number_of_rounds(num_variables);
    let round_log_inv_rates: Vec<usize> = (0..num_rounds)
        .map(|round| starting + (round + 1) * delta)
        .collect();

    let params = ProtocolParameters {
        security_level: SECURITY_LEVEL,
        pow_bits: POW_BITS,
        round_log_inv_rates,
        folding_factor,
        soundness_type: SecurityAssumption::CapacityBound,
        starting_log_inv_rate: starting,
    };

    // `WhirConfig::new` asserts on invalid knob combinations; catch and skip.
    let config = panic::catch_unwind(AssertUnwindSafe(|| {
        WhirConfig::<EF, F, Ch>::new(num_variables, params)
    }))
    .ok()?;

    // Reject configs whose per-round grinding would exceed the shared PoW budget
    // (they cannot honestly claim the target soundness).
    if !config.check_pow_bits() {
        return None;
    }

    // Total committed oracle length = initial codeword + every per-round codeword.
    // The trailing direct-send phase commits no codeword, so it is excluded.
    let mut oracle_len: u128 = 1u128 << (num_variables + starting);
    let mut queries = Vec::with_capacity(config.round_parameters.len() + 1);
    for r in &config.round_parameters {
        oracle_len += 1u128 << (r.num_variables + r.log_inv_rate);
        queries.push(r.num_queries);
    }
    queries.push(config.final_queries);

    let seed = BENCH_SEED
        ^ ((num_variables as u64) << 16)
        ^ ((log_width as u64) << 8)
        ^ ((k as u64) << 4)
        ^ (delta as u64);
    let mut rng = SmallRng::seed_from_u64(seed);

    let columns = (0..width)
        .map(|_| Poly::<F>::rand(&mut rng, log_height))
        .collect();
    let table = Table::new(columns);
    let witness = WhirLayout::new_witness(vec![table], k);

    let protocol = OpeningProtocol::new(vec![TableSpec::new(
        TableShape::new(log_height, width),
        vec![(0..width).collect()],
    )])
    .pad_to_min_num_variables(k);

    let dft = Dft::new(1 << config.max_fft_size());
    let pcs = WhirPcsTy::<MT, Ch>::new(config, dft, mmcs);

    let mut domain_separator = DomainSeparator::new(vec![]);
    pcs.add_domain_separator::<DIGEST_ELEMS>(&mut domain_separator);

    Some(WhirBuilt {
        rig: WhirRig {
            pcs,
            witness,
            protocol,
            domain_separator,
            challenger: base_challenger,
        },
        oracle_len,
        queries,
    })
}

/// Run one full WHIR proving cycle; returns the commitment, proof, and the
/// commit + open wall-clock in milliseconds.
fn whir_prove_full<MT, Ch>(rig: &WhirRig<MT, Ch>) -> (WhirCommitTy<MT>, WhirProofTy<MT>, u128, u128)
where
    MT: WhirMmcs,
    Ch: WhirChallenger<MT>,
{
    let mut prover_challenger = rig.challenger.clone();
    rig.domain_separator
        .observe_domain_separator(&mut prover_challenger);

    let t = Instant::now();
    let (commitment, prover_data) = <WhirPcsTy<MT, Ch> as MultilinearPcs<EF, Ch>>::commit(
        &rig.pcs,
        rig.witness.clone(),
        &mut prover_challenger,
    );
    let commit_ms = t.elapsed().as_millis();

    let t = Instant::now();
    let proof = <WhirPcsTy<MT, Ch> as MultilinearPcs<EF, Ch>>::open(
        &rig.pcs,
        prover_data,
        rig.protocol.clone(),
        &mut prover_challenger,
    );
    let open_ms = t.elapsed().as_millis();

    (commitment, proof, commit_ms, open_ms)
}

/// Run one full WHIR verification cycle and assert acceptance; returns microseconds.
fn whir_verify_full<MT, Ch>(
    rig: &WhirRig<MT, Ch>,
    commitment: &WhirCommitTy<MT>,
    proof: &WhirProofTy<MT>,
) -> u128
where
    MT: WhirMmcs,
    Ch: WhirChallenger<MT>,
{
    let mut verifier_challenger = rig.challenger.clone();
    rig.domain_separator
        .observe_domain_separator(&mut verifier_challenger);

    let t = Instant::now();
    <WhirPcsTy<MT, Ch> as MultilinearPcs<EF, Ch>>::verify(
        &rig.pcs,
        commitment,
        proof,
        &mut verifier_challenger,
        rig.protocol.clone(),
    )
    .expect("WHIR verify failed");
    t.elapsed().as_micros()
}

// ---------------------------------------------------------------------------
// FRI rig — parameterised over (log_blowup, max_log_arity, num_queries).
// ---------------------------------------------------------------------------

trait FriInputMmcs: Mmcs<F, Proof: Sync, Error: Sync> + Clone + Send + Sync {}
impl<T: Mmcs<F, Proof: Sync, Error: Sync> + Clone + Send + Sync> FriInputMmcs for T {}

trait FriChallengeMmcs: Mmcs<EF> + Clone + Send + Sync {}
impl<T: Mmcs<EF> + Clone + Send + Sync> FriChallengeMmcs for T {}

trait FriChal<InMmcs: Mmcs<F>, ChMmcs: Mmcs<EF>>:
    FieldChallenger<F>
    + GrindingChallenger<Witness = F>
    + CanObserve<<InMmcs as Mmcs<F>>::Commitment>
    + CanObserve<<ChMmcs as Mmcs<EF>>::Commitment>
    + Clone
{
}
impl<InMmcs: Mmcs<F>, ChMmcs: Mmcs<EF>, T> FriChal<InMmcs, ChMmcs> for T where
    T: FieldChallenger<F>
        + GrindingChallenger<Witness = F>
        + CanObserve<<InMmcs as Mmcs<F>>::Commitment>
        + CanObserve<<ChMmcs as Mmcs<EF>>::Commitment>
        + Clone
{
}

struct FriRig<InMmcs, ChMmcs, Ch>
where
    InMmcs: Mmcs<F>,
    ChMmcs: Mmcs<EF>,
{
    pcs: TwoAdicFriPcs<F, Dft, InMmcs, ChMmcs>,
    domain: TwoAdicMultiplicativeCoset<F>,
    message: RowMajorMatrix<F>,
    challenger: Ch,
}

type FriProofTy<InMmcs, ChMmcs> = p3_fri::FriProof<EF, ChMmcs, F, Vec<BatchOpening<F, InMmcs>>>;
type FriCommitTy<InMmcs> = <InMmcs as Mmcs<F>>::Commitment;

/// Total committed FRI oracle length in field elements: the input-matrix LDE
/// (the big Merkle commit) plus the FFT-free commit-phase codewords.
fn fri_oracle_len(num_variables: usize, log_width: usize, log_blowup: usize, max_log_arity: usize) -> u128 {
    // Input matrix LDE: 2^log_width columns of 2^(log_height + log_blowup) rows.
    let mut total: u128 = 1u128 << (num_variables + log_blowup);
    // Commit phase folds a single reduced codeword starting at the per-column LDE
    // height, shrinking by 2^arity per round until it reaches the final-poly floor.
    let floor = log_blowup + FRI_LOG_FINAL_POLY_LEN;
    let mut h = num_variables - log_width + log_blowup;
    while h > floor {
        total += 1u128 << h;
        h -= max_log_arity.min(h - floor);
    }
    total
}

fn fri_build<InMmcs, ChMmcs, Ch>(
    num_variables: usize,
    log_width: usize,
    log_blowup: usize,
    max_log_arity: usize,
    num_queries: usize,
    val_mmcs: InMmcs,
    challenge_mmcs: ChMmcs,
    base_challenger: Ch,
) -> FriRig<InMmcs, ChMmcs, Ch>
where
    InMmcs: FriInputMmcs,
    ChMmcs: FriChallengeMmcs,
    Ch: FriChal<InMmcs, ChMmcs>,
{
    let fri_params = FriParameters {
        log_blowup,
        log_final_poly_len: FRI_LOG_FINAL_POLY_LEN,
        max_log_arity,
        num_queries,
        commit_proof_of_work_bits: 0,
        query_proof_of_work_bits: POW_BITS,
        mmcs: challenge_mmcs,
    };

    let log_height = num_variables - log_width;
    let width = 1 << log_width;

    let dft = Dft::new(1 << (log_height + log_blowup));
    let pcs = TwoAdicFriPcs::new(dft, val_mmcs, fri_params);

    let mut rng = SmallRng::seed_from_u64(
        BENCH_SEED ^ ((num_variables as u64) << 16) ^ ((log_width as u64) << 8) ^ 0xF1,
    );
    let message = RowMajorMatrix::<F>::rand(&mut rng, 1 << log_height, width);

    let domain = <TwoAdicFriPcs<F, Dft, InMmcs, ChMmcs> as Pcs<EF, Ch>>::natural_domain_for_degree(
        &pcs,
        1 << log_height,
    );

    FriRig {
        pcs,
        domain,
        message,
        challenger: base_challenger,
    }
}

#[allow(clippy::type_complexity)]
fn fri_prove_full<InMmcs, ChMmcs, Ch>(
    rig: &FriRig<InMmcs, ChMmcs, Ch>,
) -> (
    FriCommitTy<InMmcs>,
    FriProofTy<InMmcs, ChMmcs>,
    EF,
    Vec<EF>,
    u128,
    u128,
)
where
    InMmcs: FriInputMmcs,
    ChMmcs: FriChallengeMmcs,
    Ch: FriChal<InMmcs, ChMmcs>,
{
    let mut prover_challenger = rig.challenger.clone();

    let t = Instant::now();
    let (commit, prover_data) = <TwoAdicFriPcs<F, Dft, InMmcs, ChMmcs> as Pcs<EF, Ch>>::commit(
        &rig.pcs,
        [(rig.domain, rig.message.clone())],
    );
    let commit_ms = t.elapsed().as_millis();

    prover_challenger.observe(commit.clone());
    let zeta: EF = prover_challenger.sample_algebra_element();

    let data_and_points = vec![(&prover_data, vec![vec![zeta]])];

    let t = Instant::now();
    let (openings, proof) = <TwoAdicFriPcs<F, Dft, InMmcs, ChMmcs> as Pcs<EF, Ch>>::open(
        &rig.pcs,
        data_and_points,
        &mut prover_challenger,
    );
    let open_ms = t.elapsed().as_millis();

    let values = openings[0][0][0].clone();

    (commit, proof, zeta, values, commit_ms, open_ms)
}

fn fri_verify_full<InMmcs, ChMmcs, Ch>(
    rig: &FriRig<InMmcs, ChMmcs, Ch>,
    commit: &FriCommitTy<InMmcs>,
    proof: &FriProofTy<InMmcs, ChMmcs>,
    zeta: EF,
    values: &[EF],
) -> u128
where
    InMmcs: FriInputMmcs,
    ChMmcs: FriChallengeMmcs,
    Ch: FriChal<InMmcs, ChMmcs>,
{
    let mut verifier_challenger = rig.challenger.clone();
    verifier_challenger.observe(commit.clone());
    let derived: EF = verifier_challenger.sample_algebra_element();
    assert_eq!(derived, zeta, "verifier challenger drifted from prover");

    let claims = vec![(
        commit.clone(),
        vec![(rig.domain, vec![(zeta, values.to_vec())])],
    )];

    let t = Instant::now();
    <TwoAdicFriPcs<F, Dft, InMmcs, ChMmcs> as Pcs<EF, Ch>>::verify(
        &rig.pcs,
        claims,
        proof,
        &mut verifier_challenger,
    )
    .expect("FRI verify failed");
    t.elapsed().as_micros()
}

/// Poseidon1-backed Merkle + duplex challenger (the PR's headline hash).
mod poseidon1 {
    use super::*;

    pub type Perm16 = Poseidon1KoalaBear<16>;
    pub type Perm24 = Poseidon1KoalaBear<24>;
    pub type MerkleHash = PaddingFreeSponge<Perm24, 24, 16, 8>;
    pub type MerkleCompress = TruncatedPermutation<Perm16, 2, 8, 16>;
    pub type PackedF = <F as Field>::Packing;
    pub type ValMmcs = MerkleTreeMmcs<PackedF, PackedF, MerkleHash, MerkleCompress, 2, 8>;
    pub type ChallengeMmcs = ExtensionMmcs<F, EF, ValMmcs>;
    pub type Challenger = DuplexChallenger<F, Perm16, 16, 8>;
    pub const DIGEST_ELEMS: usize = 8;

    pub fn build_kit() -> (Challenger, ValMmcs, ChallengeMmcs) {
        let perm16 = default_koalabear_poseidon1_16();
        let perm24 = default_koalabear_poseidon1_24();
        let merkle_hash = MerkleHash::new(perm24);
        let merkle_compress = MerkleCompress::new(perm16.clone());
        let val_mmcs = ValMmcs::new(merkle_hash, merkle_compress, 0);
        let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());
        let challenger = Challenger::new(perm16);
        (challenger, val_mmcs, challenge_mmcs)
    }
}

// ---------------------------------------------------------------------------
// Sweep driver + record collection.
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct Record {
    protocol: &'static str,
    label: String,
    /// Total committed oracle length in field elements (prover-cost proxy).
    oracle_len: f64,
    /// Argument size: postcard-serialised proof bytes.
    proof_bytes: f64,
    /// Measured prover wall-clock (commit + open), milliseconds.
    prove_ms: f64,
    /// Measured verifier wall-clock, microseconds.
    verify_us: f64,
    /// Whether this is the protocol's PR-default parameterisation.
    is_default: bool,
}

fn main() {
    let m: usize = std::env::var("PARETO_M")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(20);
    let log_width = LOG_FRI_BATCH_WIDTH;

    println!(
        "=== FRI vs WHIR Pareto frontier ===\n\
         m = {m} (2^{m} elements), width = 2^{log_width} = {} polys of 2^{},\n\
         {SECURITY_LEVEL}-bit capacity-regime soundness, pow_bits = {POW_BITS}, Poseidon1.\n",
        1 << log_width,
        m - log_width,
    );

    // Silence panic backtraces while probing invalid WHIR configs.
    panic::set_hook(Box::new(|_| {}));

    let mut records: Vec<Record> = Vec::new();

    // ---- WHIR sweep: folding factor k, schedule slope delta, starting rate ----
    // Prover time is the min of `PROVE_REPS` runs to suppress scheduler noise.
    const PROVE_REPS: usize = 3;
    println!("WHIR sweep (k, delta, starting): proving...");
    for starting in [1usize, 2, 3] {
        for k in [3usize, 4, 5] {
            for delta in 0..=k {
                let (challenger, val_mmcs, _) = poseidon1::build_kit();
                let Some(built) = whir_build::<_, _, { poseidon1::DIGEST_ELEMS }>(
                    m, log_width, k, delta, starting, val_mmcs, challenger,
                ) else {
                    println!("  k={k} delta={delta} starting={starting}: skipped (invalid)");
                    continue;
                };

                let (commit, proof, c0, o0) = whir_prove_full(&built.rig);
                let mut prove_ms = c0 + o0;
                for _ in 1..PROVE_REPS {
                    let (_, _, c, o) = whir_prove_full(&built.rig);
                    prove_ms = prove_ms.min(c + o);
                }
                let verify_us = (0..3)
                    .map(|_| whir_verify_full(&built.rig, &commit, &proof))
                    .min()
                    .unwrap();
                let proof_bytes = postcard::to_allocvec(&proof).expect("postcard WHIR").len();

                // PR default: starting=1, k=4, and the decay-by-2 schedule (delta=k-1).
                let is_default = starting == 1 && k == 4 && delta == k - 1;
                let q: Vec<String> = built.queries.iter().map(|q| q.to_string()).collect();
                println!(
                    "  k={k} delta={delta} starting={starting}: \
                     oracle=2^{:.2} bytes={proof_bytes} prove={prove_ms}ms queries=[{}]{}",
                    (built.oracle_len as f64).log2(),
                    q.join(","),
                    if is_default { "  <- PR default" } else { "" },
                );

                records.push(Record {
                    protocol: "WHIR",
                    label: format!("k={k} Δ={delta} s={starting}"),
                    oracle_len: built.oracle_len as f64,
                    proof_bytes: proof_bytes as f64,
                    prove_ms: prove_ms as f64,
                    verify_us: verify_us as f64,
                    is_default,
                });
            }
        }
    }

    // WHIR probing done; restore the default panic hook so any FRI panic surfaces.
    let _ = panic::take_hook();

    // ---- FRI sweep: rate (log_blowup) and folding arity (max_log_arity) ----
    println!("\nFRI sweep (log_blowup, max_log_arity): proving...");
    for log_blowup in [1usize, 2, 3, 4] {
        for max_log_arity in [1usize, 2, 3] {
            let num_queries = fri_min_queries(log_blowup, POW_BITS);
            let (challenger, val_mmcs, challenge_mmcs) = poseidon1::build_kit();
            let rig = fri_build(
                m, log_width, log_blowup, max_log_arity, num_queries, val_mmcs, challenge_mmcs,
                challenger,
            );
            let (commit, proof, zeta, values, c0, o0) = fri_prove_full(&rig);
            let mut prove_ms = c0 + o0;
            for _ in 1..PROVE_REPS {
                let (_, _, _, _, c, o) = fri_prove_full(&rig);
                prove_ms = prove_ms.min(c + o);
            }
            let verify_us = (0..3)
                .map(|_| fri_verify_full(&rig, &commit, &proof, zeta, &values))
                .min()
                .unwrap();
            let proof_bytes = postcard::to_allocvec(&proof).expect("postcard FRI").len();
            let oracle_len = fri_oracle_len(m, log_width, log_blowup, max_log_arity);

            let is_default = log_blowup == 1 && max_log_arity == 1;
            println!(
                "  blowup={log_blowup} arity=2^{max_log_arity} queries={num_queries}: \
                 oracle=2^{:.2} bytes={proof_bytes} prove={prove_ms}ms{}",
                (oracle_len as f64).log2(),
                if is_default { "  <- PR default" } else { "" },
            );

            records.push(Record {
                protocol: "FRI",
                label: format!("ρ⁻¹=2^{log_blowup} a=2^{max_log_arity} q={num_queries}"),
                oracle_len: oracle_len as f64,
                proof_bytes: proof_bytes as f64,
                prove_ms: prove_ms as f64,
                verify_us: verify_us as f64,
                is_default,
            });
        }
    }

    let _ = panic::take_hook();

    // ---- Emit artifacts ----
    write_csv(&records);
    write_svg(m, log_width, &records);

    println!(
        "\nWrote {}\n      {}",
        concat!(env!("CARGO_MANIFEST_DIR"), "/pareto_data.csv"),
        concat!(env!("CARGO_MANIFEST_DIR"), "/pareto_frontier.svg"),
    );
}

fn write_csv(records: &[Record]) {
    let mut s = String::new();
    s.push_str("protocol,label,oracle_len_elems,proof_bytes,prove_ms,verify_us,is_default\n");
    for r in records {
        let _ = writeln!(
            s,
            "{},{},{:.0},{:.0},{:.0},{:.0},{}",
            r.protocol, r.label, r.oracle_len, r.proof_bytes, r.prove_ms, r.verify_us, r.is_default,
        );
    }
    std::fs::write(concat!(env!("CARGO_MANIFEST_DIR"), "/pareto_data.csv"), s)
        .expect("write csv");
}

// ---------------------------------------------------------------------------
// Self-contained SVG plotting (log-log scatter + Pareto frontier, no deps).
// ---------------------------------------------------------------------------

const WHIR_COLOR: &str = "#1f77b4";
const FRI_COLOR: &str = "#d62728";

/// Lower-left Pareto frontier (minimise both x and y): indices into `pts`,
/// sorted by x ascending.
fn pareto_front(pts: &[(f64, f64)]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..pts.len()).collect();
    idx.sort_by(|&a, &b| {
        pts[a].0
            .partial_cmp(&pts[b].0)
            .unwrap()
            .then(pts[a].1.partial_cmp(&pts[b].1).unwrap())
    });
    let mut front = Vec::new();
    let mut best_y = f64::INFINITY;
    for &i in &idx {
        if pts[i].1 < best_y - 1e-9 {
            best_y = pts[i].1;
            front.push(i);
        }
    }
    front
}

struct Axis {
    lo: f64,
    hi: f64,
}

impl Axis {
    /// Build a log10 axis spanning the data with a small margin.
    fn new(values: impl Iterator<Item = f64>) -> Self {
        let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
        for v in values {
            let l = v.log10();
            lo = lo.min(l);
            hi = hi.max(l);
        }
        let pad = ((hi - lo) * 0.08).max(0.05);
        Self {
            lo: lo - pad,
            hi: hi + pad,
        }
    }
    fn frac(&self, v: f64) -> f64 {
        (v.log10() - self.lo) / (self.hi - self.lo)
    }
}

/// 1-2-5 decade tick positions (raw values) covering the axis range.
fn log_ticks(ax: &Axis) -> Vec<f64> {
    let mut ticks = Vec::new();
    let start = ax.lo.floor() as i32;
    let end = ax.hi.ceil() as i32;
    for e in start..=end {
        for m in [1.0, 2.0, 5.0] {
            let v = m * 10f64.powi(e);
            let l = v.log10();
            if l >= ax.lo && l <= ax.hi {
                ticks.push(v);
            }
        }
    }
    ticks
}

fn fmt_count(v: f64) -> String {
    if v >= 1e9 {
        format!("{:.1}G", v / 1e9)
    } else if v >= 1e6 {
        format!("{:.1}M", v / 1e6)
    } else if v >= 1e3 {
        format!("{:.0}K", v / 1e3)
    } else {
        format!("{v:.0}")
    }
}

fn fmt_bytes(v: f64) -> String {
    if v >= 1024.0 * 1024.0 {
        format!("{:.1}MiB", v / (1024.0 * 1024.0))
    } else {
        format!("{:.0}KiB", v / 1024.0)
    }
}

fn fmt_ms(v: f64) -> String {
    if v >= 1000.0 {
        format!("{:.1}s", v / 1000.0)
    } else {
        format!("{v:.0}ms")
    }
}

#[allow(clippy::too_many_arguments)]
fn draw_panel(
    out: &mut String,
    ox: f64,
    oy: f64,
    w: f64,
    h: f64,
    title: &str,
    x_label: &str,
    xs: &dyn Fn(&Record) -> f64,
    fmt_x: &dyn Fn(f64) -> String,
    records: &[Record],
) {
    let (pl, pr, pt, pb) = (78.0, 24.0, 44.0, 56.0);
    let (gx, gy, gw, gh) = (ox + pl, oy + pt, w - pl - pr, h - pt - pb);

    let xax = Axis::new(records.iter().map(xs));
    let yax = Axis::new(records.iter().map(|r| r.proof_bytes));
    let px = |v: f64| gx + xax.frac(v) * gw;
    let py = |v: f64| gy + gh - yax.frac(v) * gh;

    let _ = writeln!(
        out,
        r##"<text x="{:.1}" y="{:.1}" font-size="17" font-weight="bold" text-anchor="middle">{}</text>"##,
        ox + w / 2.0,
        oy + 22.0,
        title,
    );
    // Plot frame.
    let _ = writeln!(
        out,
        r##"<rect x="{gx:.1}" y="{gy:.1}" width="{gw:.1}" height="{gh:.1}" fill="#fafafa" stroke="#bbb"/>"##,
    );

    // Gridlines + tick labels.
    for t in log_ticks(&xax) {
        let x = px(t);
        let _ = writeln!(
            out,
            r##"<line x1="{x:.1}" y1="{gy:.1}" x2="{x:.1}" y2="{:.1}" stroke="#e6e6e6"/>"##,
            gy + gh,
        );
        let _ = writeln!(
            out,
            r##"<text x="{x:.1}" y="{:.1}" font-size="11" text-anchor="middle" fill="#555">{}</text>"##,
            gy + gh + 16.0,
            fmt_x(t),
        );
    }
    for t in log_ticks(&yax) {
        let y = py(t);
        let _ = writeln!(
            out,
            r##"<line x1="{gx:.1}" y1="{y:.1}" x2="{:.1}" y2="{y:.1}" stroke="#e6e6e6"/>"##,
            gx + gw,
        );
        let _ = writeln!(
            out,
            r##"<text x="{:.1}" y="{:.1}" font-size="11" text-anchor="end" fill="#555">{}</text>"##,
            gx - 6.0,
            y + 4.0,
            fmt_bytes(t),
        );
    }

    // Axis titles.
    let _ = writeln!(
        out,
        r##"<text x="{:.1}" y="{:.1}" font-size="13" text-anchor="middle">{}</text>"##,
        gx + gw / 2.0,
        oy + h - 6.0,
        x_label,
    );
    let _ = writeln!(
        out,
        r##"<text x="{:.1}" y="{:.1}" font-size="13" text-anchor="middle" transform="rotate(-90 {:.1} {:.1})">argument size (log scale)</text>"##,
        ox + 16.0,
        gy + gh / 2.0,
        ox + 16.0,
        gy + gh / 2.0,
    );

    // Per-protocol frontier + points.
    for (proto, color) in [("FRI", FRI_COLOR), ("WHIR", WHIR_COLOR)] {
        let pts: Vec<(f64, f64)> = records
            .iter()
            .filter(|r| r.protocol == proto)
            .map(|r| (xs(r), r.proof_bytes))
            .collect();
        if pts.is_empty() {
            continue;
        }
        let front = pareto_front(&pts);
        let poly: String = front
            .iter()
            .map(|&i| format!("{:.1},{:.1}", px(pts[i].0), py(pts[i].1)))
            .collect::<Vec<_>>()
            .join(" ");
        let _ = writeln!(
            out,
            r##"<polyline points="{poly}" fill="none" stroke="{color}" stroke-width="2.2" opacity="0.85"/>"##,
        );
    }
    // Draw points on top, defaults emphasised.
    for r in records {
        let color = if r.protocol == "WHIR" { WHIR_COLOR } else { FRI_COLOR };
        let (x, y) = (px(xs(r)), py(r.proof_bytes));
        if r.is_default {
            let _ = writeln!(
                out,
                r##"<circle cx="{x:.1}" cy="{y:.1}" r="6.5" fill="white" stroke="{color}" stroke-width="2.5"/>"##,
            );
            let _ = writeln!(
                out,
                r##"<circle cx="{x:.1}" cy="{y:.1}" r="2.5" fill="{color}"/>"##,
            );
            let _ = writeln!(
                out,
                r##"<text x="{:.1}" y="{:.1}" font-size="10" fill="{color}">{} default</text>"##,
                x + 9.0,
                y - 6.0,
                r.protocol,
            );
        } else {
            let _ = writeln!(
                out,
                r##"<circle cx="{x:.1}" cy="{y:.1}" r="3.6" fill="{color}" opacity="0.75"/>"##,
            );
        }
    }
}

fn write_svg(m: usize, log_width: usize, records: &[Record]) {
    let (w, h) = (1440.0, 660.0);
    let mut out = String::new();
    let _ = writeln!(
        out,
        r##"<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" font-family="sans-serif" viewBox="0 0 {w} {h}">"##,
    );
    let _ = writeln!(out, r##"<rect width="{w}" height="{h}" fill="white"/>"##);
    let _ = writeln!(
        out,
        r##"<text x="{:.1}" y="26" font-size="20" font-weight="bold" text-anchor="middle">FRI vs WHIR: prover cost vs argument size (Pareto frontier)</text>"##,
        w / 2.0,
    );
    let _ = writeln!(
        out,
        r##"<text x="{:.1}" y="46" font-size="12" text-anchor="middle" fill="#666">m=2^{m}, 256 polys of 2^{}, {SECURITY_LEVEL}-bit capacity-regime, Poseidon1 — each marker is one parameterisation; lines are lower-left Pareto frontiers (lower &amp; left is better)</text>"##,
        w / 2.0,
        m - log_width,
    );

    // Legend.
    let _ = writeln!(
        out,
        r##"<circle cx="{:.1}" cy="60" r="5" fill="{WHIR_COLOR}"/><text x="{:.1}" y="64" font-size="13">WHIR (sweep k, Δ, starting rate)</text>"##,
        w / 2.0 - 230.0,
        w / 2.0 - 220.0,
    );
    let _ = writeln!(
        out,
        r##"<circle cx="{:.1}" cy="60" r="5" fill="{FRI_COLOR}"/><text x="{:.1}" y="64" font-size="13">FRI (sweep log_blowup, arity)</text>"##,
        w / 2.0 + 20.0,
        w / 2.0 + 30.0,
    );

    let panel_w = w / 2.0;
    let panel_top = 74.0;
    let panel_h = h - panel_top - 8.0;
    draw_panel(
        &mut out,
        0.0,
        panel_top,
        panel_w,
        panel_h,
        "Theoretical proxy",
        "total committed oracle length, elements (log scale)",
        &|r: &Record| r.oracle_len,
        &fmt_count,
        records,
    );
    draw_panel(
        &mut out,
        panel_w,
        panel_top,
        panel_w,
        panel_h,
        "Measured",
        "prover wall-clock, commit + open (log scale)",
        &|r: &Record| r.prove_ms,
        &fmt_ms,
        records,
    );

    out.push_str("</svg>\n");
    std::fs::write(
        concat!(env!("CARGO_MANIFEST_DIR"), "/pareto_frontier.svg"),
        out,
    )
    .expect("write svg");
}
