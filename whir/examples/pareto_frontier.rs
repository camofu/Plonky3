//! FRI vs WHIR Pareto frontier: prover cost vs argument size.
//!
//! # What this traces
//!
//! PR #1607 measured FRI and WHIR at a single (default) parameterisation and
//! reported one point each. The interesting comparison is the *frontier*: WHIR
//! exposes more knobs (arbitrary folding strategy + per-round inverse-rate
//! schedule), so it can slide along a prover-cost-vs-argument-size curve that
//! vanilla FRI cannot.
//!
//! This example sweeps each protocol over its own knobs, holding the *claim*
//! fixed (the §1.1 univariate/multilinear bridge of PR #1607: 2^m elements as
//! 256 polynomials of size 2^(m-8), opened at one common point), and plots two
//! log-log views with the same y-axis (argument size = postcard proof bytes):
//!
//!   - x = a **modelled prover cost** (base-field multiply-equivalents): per
//!     committed codeword, the encode (FFT) and Merkle-hash terms, plus WHIR's
//!     open-phase sumcheck and the claim-batching term. This extends the older
//!     "total committed oracle length" proxy, which priced only the commit
//!     phase and so under-counted WHIR (see the cost-model constants below).
//!   - x = measured prover wall-clock (commit + open).
//!
//! Each panel draws the full sample cloud plus one per-protocol Pareto frontier
//! (lower-left is better). The raw oracle length is still emitted to the CSV.
//!
//! # Knobs swept
//!
//! WHIR: folding strategy, starting log-inv-rate, and inner per-round
//! log-inv-rate schedule. By default this covers several constant and
//! first-round-special folding strategies and affine schedules; set
//! `PARETO_EXHAUSTIVE_RATES=1` to enumerate every legal inner-rate sequence.
//! By default the folding sweep does a randomized grid search over per-round
//! folding sequences with entries in `1..=10` and at most 4 intermediate WHIR
//! rounds. The default samples random points in the full grid of
//! `(folding vector, starting rate, inner-rate vector)`, plus a few
//! constant-folding baselines, to keep the example maintainable as WHIR's best
//! region moves. Override the total sample cap with `PARETO_WHIR_SAMPLES`
//! (capped at 10000), force the initial folding/leaf arity with
//! `PARETO_WHIR_FIRST_FOLDING` (e.g. `8` for 256-wide first leaves), set the
//! maximum folding entry with `PARETO_WHIR_MAX_FOLDING`, and starting rates with
//! `PARETO_WHIR_STARTS`.
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

use std::collections::HashSet;
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
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

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
/// Extension degree of `EF` over `F`; used when normalising analytic oracle
/// lengths to base-field elements.
const EXT_DEGREE: u128 = 4;

// --- Prover-cost model constants (Steps 1-2 of the cost model). -------------
// Everything is expressed in base-field multiply-equivalents, so the FFT,
// Merkle-hash, sumcheck and batching terms can be summed into one number. The
// constants are rough order-of-magnitude values: only the *relative* shape of
// the modelled cost matters here. Calibrating them against a handful of
// measured points (model "Step 3") would make the absolute scale meaningful.

/// Cost of one extension-field multiply, in base-field multiplies (~d² with
/// schoolbook for a degree-d extension; ~9 with Karatsuba for d = 4).
const EXT_MUL_COST: f64 = 16.0;
/// Cost of one Poseidon permutation, in base-field multiplies (rough).
const PERM_COST: f64 = 200.0;
/// Degree of WHIR's eq-weighted sumcheck constraint (a constant multiplier on
/// the per-cell sumcheck work).
const SUMCHECK_DEG: f64 = 2.0;

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
    /// Number of evaluations carried by each query opening. This is the often
    /// overlooked term in argument size: high folding factors make every queried
    /// leaf much wider, even if they reduce the number of rounds.
    query_widths: Vec<usize>,
    /// Modelled prover cost (base-field multiply-equivalents): encode (FFT) +
    /// Merkle hash, per committed codeword, plus the open-phase sumcheck and
    /// claim-batching terms the raw oracle length omits.
    model_cost: f64,
}

/// Build a WHIR rig for `(m, log_width)` with an arbitrary folding strategy and
/// explicit per-round inverse-rate schedule.
///
/// Returns `None` if the resulting `WhirConfig` is invalid (panics during
/// construction, e.g. a rate step that would grow the RS domain or a two-adicity
/// overflow) or cannot reach the soundness target within the PoW budget.
fn whir_build<MT, Ch, const DIGEST_ELEMS: usize>(
    num_variables: usize,
    log_width: usize,
    folding_factor: FoldingFactor,
    round_log_inv_rates: Vec<usize>,
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

    let rate_seed = round_log_inv_rates
        .iter()
        .fold(0u64, |acc, r| acc.wrapping_mul(31) ^ (*r as u64));

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

    // Total committed oracle length in base-field elements = initial base
    // codeword + every per-round extension-field codeword. The trailing
    // direct-send phase commits no codeword, so it is excluded.
    let mut oracle_len: u128 = 1u128 << (num_variables + starting);
    let mut queries = Vec::with_capacity(config.round_parameters.len() + 1);
    let mut query_widths = Vec::with_capacity(config.round_parameters.len() + 1);
    for r in &config.round_parameters {
        oracle_len += EXT_DEGREE * (1u128 << (r.num_variables + r.log_inv_rate));
        queries.push(r.num_queries);
        query_widths.push(1usize << r.folding_factor);
    }
    queries.push(config.final_queries);
    query_widths.push(
        1usize
            << config
                .folding_factor
                .at_round(config.round_parameters.len()),
    );

    // ---- Prover-cost model (Steps 1-2): encode + hash + sumcheck + batch. ----
    let mut model_cost = 0.0f64;
    // Commit, initial base-field codeword: one FFT (encode) + Merkle hash.
    {
        let n = (1u128 << (num_variables + starting)) as f64;
        model_cost += n * n.log2(); // T_encode (base field, w_mul = 1)
        // Leaves = codeword / coset; clamp so an oversized coset gives one leaf.
        let leaves =
            (1u128 << (num_variables + starting).saturating_sub(config.folding_factor.at_round(0))) as f64;
        model_cost += (n + leaves) * PERM_COST; // T_hash
    }
    // Commit, each per-round extension-field codeword. Unlike FRI, WHIR
    // re-encodes (a fresh FFT) every round.
    for r in &config.round_parameters {
        let n = (1u128 << (r.num_variables + r.log_inv_rate)) as f64;
        model_cost += n * n.log2() * EXT_MUL_COST; // T_encode (extension field)
        let n_base = n * EXT_DEGREE as f64;
        let leaves =
            (1u128 << (r.num_variables + r.log_inv_rate).saturating_sub(r.folding_factor)) as f64;
        model_cost += (n_base + leaves) * PERM_COST; // T_hash
    }
    // Open: multilinear sumcheck over the 2^m hypercube. The cube halves each
    // round, so total prover work ≈ 2·2^m extension-field multiply-adds. FRI
    // has no analogue of this term.
    model_cost += SUMCHECK_DEG * 2.0 * (1u128 << num_variables) as f64 * EXT_MUL_COST;
    // Open: batching the `2^log_width` evaluation claims into one constraint.
    // The opening points share the trailing coordinates, so the eq-combination
    // factorises to ~one pass over the cube; generic (non-shared) points would
    // scale this with the claim count.
    model_cost += (1u128 << num_variables) as f64 * EXT_MUL_COST;

    let seed = BENCH_SEED
        ^ ((num_variables as u64) << 16)
        ^ ((log_width as u64) << 8)
        ^ ((starting as u64) << 4)
        ^ rate_seed;
    let mut rng = SmallRng::seed_from_u64(seed);

    let columns = (0..width)
        .map(|_| Poly::<F>::rand(&mut rng, log_height))
        .collect();
    let table = Table::new(columns);
    let initial_folding = config.folding_factor.at_round(0);
    let witness = WhirLayout::new_witness(vec![table], initial_folding);

    let protocol = OpeningProtocol::new(vec![TableSpec::new(
        TableShape::new(log_height, width),
        vec![(0..width).collect()],
    )])
    .pad_to_min_num_variables(initial_folding);

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
        query_widths,
        model_cost,
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

/// Total committed FRI oracle length in base-field elements: the input-matrix
/// LDE (the big Merkle commit, over `F`) plus the FFT-free commit-phase
/// codewords (over `EF`, hence multiplied by `EXT_DEGREE`).
fn fri_oracle_len(
    num_variables: usize,
    log_width: usize,
    log_blowup: usize,
    max_log_arity: usize,
) -> u128 {
    // Input matrix LDE: 2^log_width columns of 2^(log_height + log_blowup) rows.
    let mut total: u128 = 1u128 << (num_variables + log_blowup);
    // Commit phase folds a single reduced codeword starting at the per-column LDE
    // height, shrinking by 2^arity per round until it reaches the final-poly floor.
    let floor = log_blowup + FRI_LOG_FINAL_POLY_LEN;
    let mut h = num_variables - log_width + log_blowup;
    while h > floor {
        total += EXT_DEGREE * (1u128 << h);
        h -= max_log_arity.min(h - floor);
    }
    total
}

/// Modelled FRI prover cost (Steps 1-2), in base-field multiply-equivalents.
///
/// Unlike WHIR, FRI does a single FFT (the input-matrix LDE) and then folds for
/// free, so only the input oracle carries an encode term; the commit-phase
/// codewords are hashed but not re-encoded, and there is no sumcheck.
fn fri_cost(num_variables: usize, log_width: usize, log_blowup: usize, max_log_arity: usize) -> f64 {
    let per_col_log = (num_variables - log_width + log_blowup) as f64;
    let n_input = (1u128 << (num_variables + log_blowup)) as f64;

    // Commit, input matrix: one (per-column) FFT over the base field + hash.
    let mut model = n_input * per_col_log; // T_encode (base field)
    let input_leaves = (1u128 << (num_variables - log_width + log_blowup)) as f64;
    model += (n_input + input_leaves) * PERM_COST; // T_hash

    // Commit phase: hash only (FFT-free folding), extension field.
    let floor = log_blowup + FRI_LOG_FINAL_POLY_LEN;
    let mut h = num_variables - log_width + log_blowup;
    while h > floor {
        let arity = max_log_arity.min(h - floor);
        let n_base = EXT_DEGREE as f64 * (1u128 << h) as f64;
        let leaves = (1u128 << (h - arity)) as f64;
        model += (n_base + leaves) * PERM_COST; // T_hash, no T_encode
        h -= arity;
    }

    // Batching: forming g = Σ αⁱ·fᵢ — one extension-field pass over the matrix.
    model += n_input * EXT_MUL_COST; // T_batch
    model
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
struct WhirCandidate {
    name: String,
    group: usize,
    folding_factor: FoldingFactor,
    starting: usize,
    round_log_inv_rates: Vec<usize>,
}

#[allow(dead_code)]
fn whir_folding_candidates(num_variables: usize) -> Vec<(String, usize, FoldingFactor)> {
    let max_folding = std::env::var("PARETO_WHIR_MAX_FOLDING")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);
    let max_intermediate_rounds = std::env::var("PARETO_WHIR_MAX_ROUNDS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4);
    let samples = std::env::var("PARETO_WHIR_FOLDING_SAMPLES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(96);

    fn try_push(
        out: &mut Vec<(String, usize, FoldingFactor)>,
        seen: &mut HashSet<Vec<usize>>,
        factors: Vec<usize>,
        max_intermediate_rounds: usize,
        num_variables: usize,
    ) {
        if !seen.insert(factors.clone()) {
            return;
        }
        let folding = FoldingFactor::PerRound(factors.clone());
        if folding.check_validity(num_variables).is_err() {
            return;
        }
        let Some((rounds, _)) = panic::catch_unwind(AssertUnwindSafe(|| {
            folding.compute_number_of_rounds(num_variables)
        }))
        .ok() else {
            return;
        };
        // `PerRound` configs must contain exactly one folding phase per
        // intermediate round plus the final direct-send fold.
        if rounds <= max_intermediate_rounds && factors.len() == rounds + 1 {
            let name = factors
                .iter()
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
                .join("-");
            out.push((name, factors[0], folding));
        }
    }

    let mut out = Vec::new();
    let mut seen = HashSet::new();

    // Always include the PR-style constant-k baselines when they fit.
    for k in 1..=max_folding {
        for len in 1..=max_intermediate_rounds + 1 {
            try_push(
                &mut out,
                &mut seen,
                vec![k; len],
                max_intermediate_rounds,
                num_variables,
            );
        }
    }

    // Randomized grid search over [max_folding]^len for len <= max_rounds + 1.
    let mut rng =
        SmallRng::seed_from_u64(BENCH_SEED ^ ((num_variables as u64) << 32) ^ 0x5748_4952);
    let mut attempts = 0;
    while out.len() < samples && attempts < samples * 100 {
        attempts += 1;
        let len = rng.random_range(1..=max_intermediate_rounds + 1);
        let factors = (0..len)
            .map(|_| rng.random_range(1..=max_folding))
            .collect();
        try_push(
            &mut out,
            &mut seen,
            factors,
            max_intermediate_rounds,
            num_variables,
        );
    }

    out
}

fn parse_usize_list(var: &str, default: &[usize]) -> Vec<usize> {
    std::env::var(var)
        .ok()
        .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect())
        .filter(|v: &Vec<_>| !v.is_empty())
        .unwrap_or_else(|| default.to_vec())
}

fn whir_random_candidates(num_variables: usize, starts: &[usize]) -> Vec<WhirCandidate> {
    let max_folding = std::env::var("PARETO_WHIR_MAX_FOLDING")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);
    let max_intermediate_rounds = std::env::var("PARETO_WHIR_MAX_ROUNDS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4);
    let samples = std::env::var("PARETO_WHIR_SAMPLES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(512usize)
        .min(10_000);
    let forced_first_folding = std::env::var("PARETO_WHIR_FIRST_FOLDING")
        .ok()
        .and_then(|s| s.parse().ok());

    fn try_make(
        num_variables: usize,
        max_intermediate_rounds: usize,
        factors: Vec<usize>,
        starting: usize,
        rng: &mut SmallRng,
    ) -> Option<WhirCandidate> {
        let folding_factor = FoldingFactor::PerRound(factors.clone());
        folding_factor.check_validity(num_variables).ok()?;
        let (rounds, _) = panic::catch_unwind(AssertUnwindSafe(|| {
            folding_factor.compute_number_of_rounds(num_variables)
        }))
        .ok()?;
        if rounds > max_intermediate_rounds || factors.len() != rounds + 1 {
            return None;
        }

        let mut prev = starting;
        let mut round_log_inv_rates = Vec::with_capacity(rounds);
        for round in 0..rounds {
            // Legal grid point: 1 <= next_rate <= prev_rate + folding_i.
            let next = rng.random_range(1..=prev + folding_factor.at_round(round));
            round_log_inv_rates.push(next);
            prev = next;
        }

        let fold_name = factors
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join("-");
        Some(WhirCandidate {
            name: format!("f={fold_name} s={starting} r={round_log_inv_rates:?}"),
            group: factors[0],
            folding_factor,
            starting,
            round_log_inv_rates,
        })
    }

    let mut rng = SmallRng::seed_from_u64(BENCH_SEED ^ ((num_variables as u64) << 32) ^ 0xC0FFEE);
    let mut seen = HashSet::new();
    let mut out = Vec::new();

    // Keep a small set of constant-folding baselines, but do not grid-search
    // their rate schedules.
    for k in 3..=max_folding {
        if forced_first_folding.is_some_and(|first| first != k) {
            continue;
        }
        for &starting in starts {
            for len in 1..=max_intermediate_rounds + 1 {
                let factors = vec![k; len];
                if let Some(c) = try_make(
                    num_variables,
                    max_intermediate_rounds,
                    factors,
                    starting,
                    &mut rng,
                ) {
                    if seen.insert(c.name.clone()) {
                        out.push(c);
                    }
                }
            }
        }
    }

    let mut attempts = 0usize;
    while out.len() < samples && attempts < samples.saturating_mul(100) {
        attempts += 1;
        let len = rng.random_range(1..=max_intermediate_rounds + 1);
        let mut factors: Vec<usize> = (0..len)
            .map(|_| rng.random_range(1..=max_folding))
            .collect();
        if let Some(first) = forced_first_folding {
            factors[0] = first;
        }
        let starting = starts[rng.random_range(0..starts.len())];
        let Some(c) = try_make(
            num_variables,
            max_intermediate_rounds,
            factors,
            starting,
            &mut rng,
        ) else {
            continue;
        };
        if seen.insert(c.name.clone()) {
            out.push(c);
        }
    }

    out
}

#[allow(dead_code)]
fn whir_rate_schedules(
    folding: &FoldingFactor,
    num_variables: usize,
    starting: usize,
) -> Vec<Vec<usize>> {
    let (num_rounds, _) = folding.compute_number_of_rounds(num_variables);
    if num_rounds == 0 {
        return vec![vec![]];
    }

    let exhaustive = std::env::var("PARETO_EXHAUSTIVE_RATES").is_ok_and(|v| v != "0");
    if !exhaustive {
        let max_delta = (0..num_rounds)
            .map(|r| folding.at_round(r))
            .min()
            .unwrap_or(1);
        let mut out: Vec<Vec<usize>> = Vec::new();

        // Affine non-decreasing schedules from the starting rate. These are the
        // old/default schedules and include the PR-style decay-by-2 line.
        out.extend((0..=max_delta).map(|delta| {
            (0..num_rounds)
                .map(|round| starting + (round + 1) * delta)
                .collect()
        }));

        // Also explicitly test high-rate inner rounds. In particular, log_inv=1
        // is rate 1/2; these schedules are legal even when the starting codeword
        // has lower rate (e.g. start=2 or 3), and they can trade more queries for
        // fewer committed elements and smaller Merkle paths.
        for inner in 1..=starting.min(3) {
            out.push(vec![inner; num_rounds]);
        }
        if num_rounds >= 2 {
            out.push((0..num_rounds).map(|round| 1 + round.min(2)).collect());
            out.push((0..num_rounds).map(|round| 1 + 2 * round.min(2)).collect());
        }

        // Random legal inner-rate schedules. For round i, WHIR only requires
        // next_rate <= prev_rate + folding_i (otherwise the RS domain would
        // grow). Sampling this directly explores high-rate inner rounds such as
        // log_inv_rate=1 even when the initial codeword starts at rate 1/4 or
        // 1/8.
        let rate_samples = std::env::var("PARETO_WHIR_RATE_SAMPLES")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(8);
        let fold_seed = (0..num_rounds).fold(0u64, |acc, r| {
            acc.wrapping_mul(17) ^ (folding.at_round(r) as u64)
        });
        let mut rng = SmallRng::seed_from_u64(
            BENCH_SEED ^ 0x5241_5445 ^ ((starting as u64) << 24) ^ fold_seed,
        );
        for _ in 0..rate_samples {
            let mut prev = starting;
            let mut sched = Vec::with_capacity(num_rounds);
            for round in 0..num_rounds {
                let next = rng.random_range(1..=prev + folding.at_round(round));
                sched.push(next);
                prev = next;
            }
            out.push(sched);
        }

        out.sort_unstable();
        out.dedup();
        return out;
    }

    fn rec(
        out: &mut Vec<Vec<usize>>,
        cur: &mut Vec<usize>,
        folding: &FoldingFactor,
        round: usize,
        num_rounds: usize,
        prev: usize,
    ) {
        if round == num_rounds {
            out.push(cur.clone());
            return;
        }
        // Full legal range for this inner inverse rate: any rate that does not
        // require growing the RS domain after this fold.
        for next in 1..=prev + folding.at_round(round) {
            cur.push(next);
            rec(out, cur, folding, round + 1, num_rounds, next);
            cur.pop();
        }
    }
    let mut out = Vec::new();
    rec(&mut out, &mut Vec::new(), folding, 0, num_rounds, starting);
    out
}

#[derive(Clone)]
struct Record {
    protocol: &'static str,
    label: String,
    /// This config's knob values; currently emitted only for possible future CSV/plot labels.
    #[allow(dead_code)]
    knobs: Vec<(&'static str, usize)>,
    /// Total committed oracle length in base-field elements (the original,
    /// commit-only proxy). Kept in the CSV for reference.
    oracle_len: f64,
    /// Modelled prover cost (base-field multiply-equivalents): commit (encode +
    /// hash) plus the open-phase sumcheck + batching terms. This is the x-axis
    /// of the left panel.
    model_cost: f64,
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
         {SECURITY_LEVEL}-bit capacity-regime soundness, pow_bits = {POW_BITS}, Poseidon1.\n\
         WHIR env: PARETO_WHIR_SAMPLES, PARETO_WHIR_FIRST_FOLDING, PARETO_WHIR_MAX_FOLDING, PARETO_WHIR_MAX_ROUNDS, PARETO_WHIR_STARTS.\n",
        1 << log_width,
        m - log_width,
    );

    // Silence panic backtraces while probing invalid WHIR configs.
    panic::set_hook(Box::new(|_| {}));

    let mut records: Vec<Record> = Vec::new();

    // ---- WHIR sweep: folding strategy, explicit inner rates, starting rate ----
    // Prover time is the min of `PROVE_REPS` runs to suppress scheduler noise.
    const PROVE_REPS: usize = 3;
    let starts = parse_usize_list("PARETO_WHIR_STARTS", &[1, 2, 3]);
    let whir_candidates = whir_random_candidates(m, &starts);
    println!(
        "WHIR randomized sweep (folding, starting, inner rates): proving {} candidates...",
        whir_candidates.len()
    );
    for cand in whir_candidates {
        let (challenger, val_mmcs, _) = poseidon1::build_kit();
        let Some(built) = whir_build::<_, _, { poseidon1::DIGEST_ELEMS }>(
            m,
            log_width,
            cand.folding_factor.clone(),
            cand.round_log_inv_rates.clone(),
            cand.starting,
            val_mmcs,
            challenger,
        ) else {
            println!("  {}: skipped (invalid)", cand.name);
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

        let default_rates: Vec<usize> = (0..cand.round_log_inv_rates.len())
            .map(|round| 1 + (round + 1) * 3)
            .collect();
        let is_default = cand.starting == 1
            && matches!(cand.folding_factor, FoldingFactor::Constant(4))
            && cand.round_log_inv_rates == default_rates;
        let q: Vec<String> = built.queries.iter().map(|q| q.to_string()).collect();
        let qw: Vec<String> = built.query_widths.iter().map(|w| w.to_string()).collect();
        let queried_values: usize = built
            .queries
            .iter()
            .zip(&built.query_widths)
            .map(|(q, w)| q * w)
            .sum();
        println!(
            "  {}: oracle=2^{:.2} bytes={proof_bytes} prove={prove_ms}ms queries=[{}] widths=[{}] qvals={}{}",
            cand.name,
            (built.oracle_len as f64).log2(),
            q.join(","),
            qw.join(","),
            queried_values,
            if is_default { "  <- PR default" } else { "" },
        );

        records.push(Record {
            protocol: "WHIR",
            label: cand.name,
            knobs: vec![("k", cand.group), ("start", cand.starting)],
            oracle_len: built.oracle_len as f64,
            model_cost: built.model_cost,
            proof_bytes: proof_bytes as f64,
            prove_ms: prove_ms as f64,
            verify_us: verify_us as f64,
            is_default,
        });
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
                m,
                log_width,
                log_blowup,
                max_log_arity,
                num_queries,
                val_mmcs,
                challenge_mmcs,
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
            // FRI's PCS API returns the opened values separately from the proof,
            // whereas WHIR's `PcsProof` stores them in `evals`. Count both, so
            // the plotted argument size is comparable across protocols.
            let proof_bytes = postcard::to_allocvec(&(proof.clone(), values.clone()))
                .expect("postcard FRI proof + openings")
                .len();
            let oracle_len = fri_oracle_len(m, log_width, log_blowup, max_log_arity);
            let model_cost = fri_cost(m, log_width, log_blowup, max_log_arity);

            let arities: Vec<usize> = proof
                .query_proofs
                .first()
                .map(|qp| {
                    qp.commit_phase_openings
                        .iter()
                        .map(|step| step.log_arity as usize)
                        .collect()
                })
                .unwrap_or_default();
            let sibling_values_per_query: usize = proof
                .query_proofs
                .first()
                .map(|qp| {
                    qp.commit_phase_openings
                        .iter()
                        .map(|step| step.sibling_values.len())
                        .sum()
                })
                .unwrap_or(0);

            let is_default = log_blowup == 1 && max_log_arity == 1;
            println!(
                "  blowup={log_blowup} arity=2^{max_log_arity} queries={num_queries}: \
                 oracle=2^{:.2} bytes={proof_bytes} prove={prove_ms}ms arities={arities:?} sibvals/q={sibling_values_per_query}{}",
                (oracle_len as f64).log2(),
                if is_default { "  <- PR default" } else { "" },
            );

            records.push(Record {
                protocol: "FRI",
                label: format!("ρ⁻¹=2^{log_blowup} a=2^{max_log_arity} q={num_queries}"),
                knobs: vec![("blowup", log_blowup), ("arity", max_log_arity)],
                oracle_len: oracle_len as f64,
                model_cost,
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

fn csv_escape(s: &str) -> String {
    format!("\"{}\"", s.replace('"', "\"\""))
}

fn write_csv(records: &[Record]) {
    let mut s = String::new();
    s.push_str(
        "protocol,label,oracle_len_elems,model_cost,proof_bytes,prove_ms,verify_us,is_default\n",
    );
    for r in records {
        let _ = writeln!(
            s,
            "{},{},{:.0},{:.0},{:.0},{:.0},{:.0},{}",
            csv_escape(r.protocol),
            csv_escape(&r.label),
            r.oracle_len,
            r.model_cost,
            r.proof_bytes,
            r.prove_ms,
            r.verify_us,
            r.is_default,
        );
    }
    std::fs::write(concat!(env!("CARGO_MANIFEST_DIR"), "/pareto_data.csv"), s).expect("write csv");
}

// ---------------------------------------------------------------------------
// Self-contained SVG plotting: two log-log panels (oracle-length proxy and
// measured prover time). Each panel draws dotted iso-knob curves, shaded by
// value — WHIR grouped by k (blues), FRI grouped by log_blowup (reds). No deps.
// ---------------------------------------------------------------------------

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
    let _ = writeln!(
        out,
        r##"<rect x="{gx:.1}" y="{gy:.1}" width="{gw:.1}" height="{gh:.1}" fill="#fafafa" stroke="#bbb"/>"##,
    );
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

    // Draw all sampled points, then one Pareto frontier per protocol. This is
    // more meaningful for randomized search than iso-knob curves: each protocol
    // gets exactly one lower-left envelope in this panel.
    for (proto, color) in [("FRI", "#de2d26"), ("WHIR", "#08519c")] {
        let mut pts: Vec<(f64, f64, bool)> = records
            .iter()
            .filter(|r| r.protocol == proto)
            .map(|r| (xs(r), r.proof_bytes, r.is_default))
            .collect();
        pts.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        for &(xv, yv, def) in &pts {
            let (cx, cy) = (px(xv), py(yv));
            let _ = writeln!(
                out,
                r##"<circle cx="{cx:.1}" cy="{cy:.1}" r="3.2" fill="{color}" opacity="0.28"/>"##,
            );
            if def {
                let _ = writeln!(
                    out,
                    r##"<circle cx="{cx:.1}" cy="{cy:.1}" r="6.5" fill="none" stroke="#222" stroke-width="1.6"/>"##,
                );
            }
        }

        let mut frontier = Vec::new();
        let mut best_y = f64::INFINITY;
        for &(xv, yv, def) in &pts {
            if yv < best_y {
                frontier.push((xv, yv, def));
                best_y = yv;
            }
        }
        if frontier.len() >= 2 {
            let poly: String = frontier
                .iter()
                .map(|p| format!("{:.1},{:.1}", px(p.0), py(p.1)))
                .collect::<Vec<_>>()
                .join(" ");
            let _ = writeln!(
                out,
                r##"<polyline points="{poly}" fill="none" stroke="{color}" stroke-width="3.0" opacity="0.95"/>"##,
            );
        }
        for &(xv, yv, _) in &frontier {
            let (cx, cy) = (px(xv), py(yv));
            let _ = writeln!(
                out,
                r##"<circle cx="{cx:.1}" cy="{cy:.1}" r="4.8" fill="white" stroke="{color}" stroke-width="2.2"/>"##,
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
        r##"<text x="{:.1}" y="26" font-size="20" font-weight="bold" text-anchor="middle">FRI vs WHIR: prover cost vs argument size</text>"##,
        w / 2.0,
    );
    let _ = writeln!(
        out,
        r##"<text x="{:.1}" y="46" font-size="12" text-anchor="middle" fill="#666">m=2^{m}, 256 polys of 2^{}, {SECURITY_LEVEL}-bit capacity-regime, Poseidon1 — left x = modelled prover cost (encode+hash+sumcheck+batch), right x = measured wall-clock · solid lines = per-protocol Pareto frontiers · ◯ = PR default · lower-left is better</text>"##,
        w / 2.0,
        m - log_width,
    );

    // Legend: one frontier per protocol.
    let lx = w / 2.0 - 170.0;
    let _ = writeln!(
        out,
        r##"<line x1="{lx:.1}" y1="60" x2="{:.1}" y2="60" stroke="#08519c" stroke-width="3"/>"##,
        lx + 34.0
    );
    let _ = writeln!(
        out,
        r##"<text x="{:.1}" y="64" font-size="13" fill="#333">WHIR frontier</text>"##,
        lx + 42.0
    );
    let rx = w / 2.0 + 40.0;
    let _ = writeln!(
        out,
        r##"<line x1="{rx:.1}" y1="60" x2="{:.1}" y2="60" stroke="#de2d26" stroke-width="3"/>"##,
        rx + 34.0
    );
    let _ = writeln!(
        out,
        r##"<text x="{:.1}" y="64" font-size="13" fill="#333">FRI frontier</text>"##,
        rx + 42.0
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
        "Modelled prover cost",
        "encode + hash + sumcheck + batch (base-mult-equiv, log scale)",
        &|r: &Record| r.model_cost,
        &fmt_count,
        records,
    );
    draw_panel(
        &mut out,
        panel_w,
        panel_top,
        panel_w,
        panel_h,
        "Measured (all cores)",
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
