//! General-purpose state keeper metrics.

use std::{
    sync::{Mutex, Weak},
    time::{Duration, Instant},
};

use vise::{
    Buckets, Counter, EncodeLabelSet, EncodeLabelValue, Family, Gauge, Histogram, LabeledFamily,
    LatencyObserver, Metrics, Unit,
};
use zksync_mempool::MempoolStore;
use zksync_multivm::interface::{DeduplicatedWritesMetrics, FinishedL1Batch, VmRevertReason};
use zksync_types::{helpers::unix_timestamp_ms, ProtocolVersionId};

use super::seal_criteria::SealResolution;
use crate::updates::UpdatesManager;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, EncodeLabelValue, EncodeLabelSet)]
#[metrics(label = "stage", rename_all = "snake_case")]
pub enum TxExecutionStage {
    Execution,
    TxRollback,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, EncodeLabelValue, EncodeLabelSet)]
#[metrics(label = "tx_execution_type", rename_all = "snake_case")]
pub enum TxExecutionType {
    L1,
    L2,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, EncodeLabelValue)]
#[metrics(rename_all = "snake_case")]
pub enum TxExecutionStatus {
    Success,
    Rejected,
    Reverted,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, EncodeLabelSet)]
pub struct TxExecutionResult {
    status: TxExecutionStatus,
    reason: Option<&'static str>,
}

impl TxExecutionType {
    pub fn from_is_l1(is_l1: bool) -> TxExecutionType {
        match is_l1 {
            true => TxExecutionType::L1,
            false => TxExecutionType::L2,
        }
    }
}

const INCLUSION_DELAY_BUCKETS: Buckets = Buckets::values(&[
    0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9,
    2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 20.0, 30.0, 60.0, 120.0, 240.0,
]);

/// General-purpose state keeper metrics.
#[derive(Debug, Metrics)]
#[metrics(prefix = "server_state_keeper")]
pub struct StateKeeperMetrics {
    /// Latency to synchronize the mempool with Postgres.
    #[metrics(buckets = Buckets::LATENCIES)]
    pub mempool_sync: Histogram<Duration>,
    /// Number of stashed accounts in mempool
    pub mempool_stashed_accounts: Gauge<usize>,
    /// Number of purged accounts in mempool
    pub mempool_purged_accounts: Gauge<usize>,
    /// Latency of the state keeper waiting for a transaction.
    #[metrics(buckets = Buckets::LATENCIES)]
    pub waiting_for_tx: Histogram<Duration>,
    /// Latency of the state keeper getting a transaction from the mempool.
    #[metrics(buckets = Buckets::LATENCIES)]
    pub get_tx_from_mempool: Histogram<Duration>,
    /// Number of transactions completed with a specific result.
    pub tx_execution_result: Family<TxExecutionResult, Counter>,
    /// Time spent waiting for the hash of a previous L1 batch.
    #[metrics(buckets = Buckets::LATENCIES)]
    pub wait_for_prev_hash_time: Histogram<Duration>,
    /// Time spent waiting for the header of a previous L2 block.
    #[metrics(buckets = Buckets::LATENCIES)]
    pub load_previous_miniblock_header: Histogram<Duration>,
    /// The time it takes for transactions to be included in a block. Representative of the time user must wait before their transaction is confirmed.
    #[metrics(buckets = INCLUSION_DELAY_BUCKETS)]
    pub transaction_inclusion_delay: Family<TxExecutionType, Histogram<Duration>>,
    /// The time it takes to match seal resolution for each tx.
    #[metrics(buckets = Buckets::LATENCIES)]
    pub match_seal_resolution: Histogram<Duration>,
    /// The time it takes to determine seal resolution for each tx.
    #[metrics(buckets = Buckets::LATENCIES)]
    pub determine_seal_resolution: Histogram<Duration>,
    /// The time it takes for state keeper to wait for tx execution result from batch executor.
    #[metrics(buckets = Buckets::LATENCIES)]
    pub execute_tx_outer_time: Histogram<Duration>,
    /// The time it takes for one iteration of the main loop in `process_block`.
    #[metrics(buckets = Buckets::LATENCIES)]
    pub process_block_loop_iteration: Histogram<Duration>,
    /// The time it takes to wait for new L2 block parameters
    #[metrics(buckets = Buckets::LATENCIES)]
    pub wait_for_l2_block_params: Histogram<Duration>,
}

fn vm_revert_reason_as_metric_label(reason: &VmRevertReason) -> &'static str {
    match reason {
        VmRevertReason::General { .. } => "General",
        VmRevertReason::InnerTxError => "InnerTxError",
        VmRevertReason::VmError => "VmError",
        _ => "Unknown",
    }
}

impl StateKeeperMetrics {
    pub fn inc_rejected_txs(&self, reason: &'static str) {
        let result = TxExecutionResult {
            status: TxExecutionStatus::Rejected,
            reason: Some(reason),
        };

        self.tx_execution_result[&result].inc();
    }

    pub fn inc_succeeded_txs(&self) {
        let result = TxExecutionResult {
            status: TxExecutionStatus::Success,
            reason: None,
        };

        self.tx_execution_result[&result].inc();
    }

    pub fn inc_reverted_txs(&self, reason: &VmRevertReason) {
        let result = TxExecutionResult {
            status: TxExecutionStatus::Reverted,
            reason: Some(vm_revert_reason_as_metric_label(reason)),
        };

        self.tx_execution_result[&result].inc();
    }
}

#[vise::register]
pub static KEEPER_METRICS: vise::Global<StateKeeperMetrics> = vise::Global::new();

/// State keeper-related gauges exposed via a collector.
#[derive(Debug, Metrics)]
#[metrics(prefix = "server_state_keeper")]
pub(super) struct StateKeeperGauges {
    /// Current number of L1 transactions in the mempool.
    mempool_l1_size: Gauge<usize>,
    /// Current number of L2 transactions in the mempool.
    mempool_l2_size: Gauge<u64>,
    /// Current size of the L2 priority queue.
    l2_priority_queue_size: Gauge<usize>,
}

impl StateKeeperGauges {
    pub(super) fn register(pool_ref: Weak<Mutex<MempoolStore>>) {
        #[vise::register]
        static COLLECTOR: vise::Collector<Option<StateKeeperGauges>> = vise::Collector::new();

        let res = COLLECTOR.before_scrape(move || {
            pool_ref.upgrade().map(|pool| {
                let stats = pool.lock().expect("failed to acquire mempool lock").stats();
                drop(pool); // Don't prevent the pool to be dropped

                let gauges = StateKeeperGauges::default();
                gauges.mempool_l1_size.set(stats.l1_transaction_count);
                gauges.mempool_l2_size.set(stats.l2_transaction_count);
                gauges
                    .l2_priority_queue_size
                    .set(stats.l2_priority_queue_size);
                gauges
            })
        });
        if res.is_err() {
            tracing::warn!(
                "Mempool registered for metrics multiple times; this is a logical error"
            );
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, EncodeLabelValue)]
#[metrics(rename_all = "snake_case")]
enum SealResolutionLabel {
    NoSeal,
    IncludeAndSeal,
    ExcludeAndSeal,
    Unexecutable,
}

impl From<&SealResolution> for SealResolutionLabel {
    fn from(resolution: &SealResolution) -> Self {
        match resolution {
            SealResolution::NoSeal => Self::NoSeal,
            SealResolution::IncludeAndSeal => Self::IncludeAndSeal,
            SealResolution::ExcludeAndSeal => Self::ExcludeAndSeal,
            SealResolution::Unexecutable(_) => Self::Unexecutable,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, EncodeLabelSet, EncodeLabelValue)]
#[metrics(label = "reason", rename_all = "snake_case")]
pub(super) enum L2BlockSealReason {
    Timeout,
    PayloadSize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, EncodeLabelSet)]
struct TxAggregationLabels {
    criterion: &'static str,
    seal_resolution: Option<SealResolutionLabel>,
}

#[derive(Debug, Metrics)]
#[metrics(prefix = "server_tx_aggregation")]
pub(super) struct TxAggregationMetrics {
    reason: Family<TxAggregationLabels, Counter>,
    l2_block_reason: Family<L2BlockSealReason, Counter>,
    #[metrics(labels = ["criterion"], buckets = Buckets::ZERO_TO_ONE, unit = Unit::Ratios)]
    criterion_capacity_filled: LabeledFamily<&'static str, Histogram>,
}

impl TxAggregationMetrics {
    pub fn l1_batch_reason_inc(&self, criterion: &'static str, resolution: &SealResolution) {
        let labels = TxAggregationLabels {
            criterion,
            seal_resolution: Some(resolution.into()),
        };
        self.reason[&labels].inc();
    }

    pub fn l1_batch_reason_inc_criterion(&self, criterion: &'static str) {
        let labels = TxAggregationLabels {
            criterion,
            seal_resolution: None,
        };
        self.reason[&labels].inc();
    }

    pub fn l2_block_reason_inc(&self, reason: &L2BlockSealReason) {
        self.l2_block_reason[reason].inc();
    }

    pub fn record_criterion_capacity(&self, criterion: &'static str, value: f64) {
        self.criterion_capacity_filled[&criterion].observe(value)
    }
}

#[vise::register]
pub(super) static AGGREGATION_METRICS: vise::Global<TxAggregationMetrics> = vise::Global::new();

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, EncodeLabelValue, EncodeLabelSet)]
#[metrics(label = "stage", rename_all = "snake_case")]
pub(super) enum L1BatchSealStage {
    #[metrics(name = "fictive_miniblock")]
    FictiveL2Block,
    LogDeduplication,
    InsertL1BatchHeader,
    #[metrics(name = "set_l1_batch_number_for_miniblocks")]
    SetL1BatchNumberForL2Blocks,
    MarkTxsAsExecutedInL1Batch,
    InsertProtectiveReads,
    FilterWrittenSlots,
    InsertInitialWrites,
    CommitL1Batch,
}

/// Buckets for positive integer, not-so-large values (e.g., initial writes count).
const COUNT_BUCKETS: Buckets = Buckets::values(&[
    10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1_000.0, 2_000.0, 5_000.0, 10_000.0, 20_000.0, 50_000.0,
]);
/// Buckets for sealing deltas for L1 batches (in seconds). The expected delta is approximately 1 minute.
const L1_BATCH_SEAL_DELTA_BUCKETS: Buckets = Buckets::values(&[
    0.1, 0.5, 1.0, 5.0, 10.0, 20.0, 30.0, 40.0, 60.0, 90.0, 120.0, 180.0, 240.0, 300.0,
]);

/// Metrics related to L1 batch sealing.
#[derive(Debug, Metrics)]
#[metrics(prefix = "server_state_keeper_l1_batch")]
pub(crate) struct L1BatchMetrics {
    /// Delta between sealing consecutive L1 batches.
    #[metrics(buckets = L1_BATCH_SEAL_DELTA_BUCKETS)]
    pub seal_delta: Histogram<Duration>,
    /// Number of initial writes in a single L1 batch.
    #[metrics(buckets = COUNT_BUCKETS)]
    pub initial_writes: Histogram<usize>,
    /// Number of repeated writes in a single L1 batch.
    #[metrics(buckets = COUNT_BUCKETS)]
    pub repeated_writes: Histogram<usize>,
    /// Number of transactions in a single L1 batch.
    #[metrics(buckets = COUNT_BUCKETS)]
    pub transactions_in_l1_batch: Histogram<usize>,
    /// Total latency of sealing an L1 batch.
    #[metrics(buckets = Buckets::LATENCIES)]
    pub sealed_time: Histogram<Duration>,
    /// Latency of sealing an L1 batch split by the stage.
    #[metrics(buckets = Buckets::LATENCIES)]
    sealed_time_stage: Family<L1BatchSealStage, Histogram<Duration>>,
    /// Number of entities stored in Postgres during a specific stage of sealing an L1 batch.
    #[metrics(buckets = COUNT_BUCKETS)]
    sealed_entity_count: Family<L1BatchSealStage, Histogram<usize>>,
    /// Latency of sealing an L1 batch split by the stage and divided by the number of entries
    /// stored in the stage.
    #[metrics(buckets = Buckets::LATENCIES)]
    sealed_entity_per_unit: Family<L1BatchSealStage, Histogram<Duration>>,
}

impl L1BatchMetrics {
    pub(super) fn start(&self, stage: L1BatchSealStage) -> SealProgress<'_> {
        SealProgress {
            target: "L1 batch",
            stage_name: format!("{stage:?}"),
            latency: self.sealed_time_stage[&stage].start(),
            entity_count: &self.sealed_entity_count[&stage],
            latency_per_unit: &self.sealed_entity_per_unit[&stage],
        }
    }
}

#[vise::register]
pub(crate) static L1_BATCH_METRICS: vise::Global<L1BatchMetrics> = vise::Global::new();

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, EncodeLabelValue, EncodeLabelSet)]
#[metrics(label = "stage", rename_all = "snake_case")]
pub(super) enum L2BlockQueueStage {
    Submit,
    WaitForAllCommands,
    NextCommand,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, EncodeLabelValue)]
#[metrics(rename_all = "snake_case")]
pub(super) enum L2BlockSealStage {
    #[metrics(name = "insert_miniblock_header")]
    InsertL2BlockHeader,
    #[metrics(name = "mark_transactions_in_miniblock")]
    MarkTransactionsInL2Block,
    InsertStorageLogs,
    InsertFactoryDeps,
    ExtractAddedTokens,
    InsertTokens,
    ExtractEvents,
    InsertEvents,
    ExtractL2ToL1Logs,
    InsertL2ToL1Logs,
    ReportTxMetrics,
    CalculateLogsBloom,
    MarkInteropRootsAsSealed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, EncodeLabelSet)]
struct L2BlockSealLabels {
    stage: L2BlockSealStage,
    is_fictive: &'static str,
}

#[derive(Debug, Metrics)]
#[metrics(prefix = "server_state_keeper_miniblock")]
pub(super) struct L2BlockMetrics {
    /// Delta between sealing consecutive L2 blocks.
    #[metrics(buckets = Buckets::LATENCIES)]
    pub seal_delta: Histogram<Duration>,
    /// Current capacity of the seal queue for L2 blocks.
    pub seal_queue_capacity: Gauge<usize>,
    /// Latency of a certain operation concerning the seal queue for L2 blocks.
    #[metrics(buckets = Buckets::LATENCIES)]
    pub seal_queue_latency: Family<L2BlockQueueStage, Histogram<Duration>>,
    /// Number of transactions in a single L2 block.
    #[metrics(buckets = Buckets::linear(0.0..=50.0, 5.0))]
    pub transactions_in_miniblock: Histogram<usize>,
    /// Total latency of sealing an L2 block.
    #[metrics(buckets = Buckets::LATENCIES)]
    pub sealed_time: Histogram<Duration>,
    /// Latency of sealing an L2 block split by the stage.
    #[metrics(buckets = Buckets::LATENCIES)]
    sealed_time_stage: Family<L2BlockSealLabels, Histogram<Duration>>,
    /// Number of entities stored in Postgres during a specific stage of sealing an L2 block.
    #[metrics(buckets = COUNT_BUCKETS)]
    sealed_entity_count: Family<L2BlockSealLabels, Histogram<usize>>,
    /// Latency of sealing an L2 block split by the stage and divided by the number of entries
    /// stored in the stage.
    #[metrics(buckets = Buckets::LATENCIES)]
    sealed_entity_per_unit: Family<L2BlockSealLabels, Histogram<Duration>>,
}

impl L2BlockMetrics {
    pub(super) fn start(&self, stage: L2BlockSealStage, is_fictive: bool) -> SealProgress<'_> {
        let labels = L2BlockSealLabels {
            stage,
            is_fictive: if is_fictive { "true" } else { "false" },
        };
        SealProgress {
            target: "L2 block",
            stage_name: format!("{stage:?}"),
            latency: self.sealed_time_stage[&labels].start(),
            entity_count: &self.sealed_entity_count[&labels],
            latency_per_unit: &self.sealed_entity_per_unit[&labels],
        }
    }
}

#[vise::register]
pub(super) static L2_BLOCK_METRICS: vise::Global<L2BlockMetrics> = vise::Global::new();

/// Tracking progress of L1 batch or L2 block sealing.
#[must_use = "Progress must be `observe()`d"]
#[derive(Debug)]
pub(super) struct SealProgress<'a> {
    target: &'static str,
    stage_name: String,
    latency: LatencyObserver<'a>,
    entity_count: &'a Histogram<usize>,
    latency_per_unit: &'a Histogram<Duration>,
}

impl SealProgress<'_> {
    pub fn observe(self, count: impl Into<Option<usize>>) {
        const MIN_STAGE_DURATION_TO_REPORT: Duration = Duration::from_millis(10);

        let count = count.into();
        let elapsed = self.latency.observe();
        if elapsed > MIN_STAGE_DURATION_TO_REPORT {
            let target = self.target;
            let stage = &self.stage_name;
            tracing::debug!(
                "{target} execution stage {stage} took {elapsed:?} with count {count:?}"
            );
        }

        if let Some(count) = count {
            self.entity_count.observe(count);
            if count > 0 {
                self.latency_per_unit.observe(elapsed.div_f64(count as f64));
            }
        }
    }
}

#[derive(Debug, Metrics)]
#[metrics(prefix = "batch_tip")]
pub(crate) struct BatchTipMetrics {
    #[metrics(buckets = Buckets::exponential(1.0..=60000.0, 2.0))]
    block_writes_metrics_positive_size: Histogram<usize>,
    #[metrics(buckets = Buckets::exponential(1.0..=60000.0, 2.0))]
    block_writes_metrics_negative_size: Histogram<usize>,
}

impl BatchTipMetrics {
    pub fn observe_writes_metrics(
        &self,
        initial_writes_metrics: &DeduplicatedWritesMetrics,
        applied_writes_metrics: &DeduplicatedWritesMetrics,
        protocol_version_id: ProtocolVersionId,
    ) {
        let size_diff = applied_writes_metrics.size(protocol_version_id) as i128
            - initial_writes_metrics.size(protocol_version_id) as i128;

        if size_diff > 0 {
            self.block_writes_metrics_positive_size
                .observe(size_diff as usize);
        } else {
            self.block_writes_metrics_negative_size
                .observe(size_diff.unsigned_abs() as usize);
        }
    }
}

#[vise::register]
pub(crate) static BATCH_TIP_METRICS: vise::Global<BatchTipMetrics> = vise::Global::new();

#[derive(Debug, Metrics)]
#[metrics(prefix = "server_state_keeper_updates_manager")]
pub struct UpdatesManagerMetrics {
    #[metrics(buckets = Buckets::LATENCIES)]
    pub finish_batch: Histogram<Duration>,
    #[metrics(buckets = Buckets::LATENCIES)]
    pub extend_from_executed_transaction: Histogram<Duration>,
}

#[vise::register]
pub(crate) static UPDATES_MANAGER_METRICS: vise::Global<UpdatesManagerMetrics> =
    vise::Global::new();

/// Metrics and logs for batch-level sealing telemetry.
#[derive(Debug, Metrics)]
#[metrics(prefix = "server_state_keeper_batch")]
pub(crate) struct BatchTelemetryMetrics {
    /// Time from transaction receipt to L1 batch seal, split by tx type.
    #[metrics(buckets = Buckets::LATENCIES)]
    tx_finality_delay: Family<TxExecutionType, Histogram<Duration>>,
    /// Total time spent sealing the batch.
    #[metrics(buckets = Buckets::LATENCIES)]
    pub seal_time: Histogram<Duration>,
    /// Number of transactions in the batch.
    #[metrics(buckets = COUNT_BUCKETS)]
    pub tx_count: Histogram<usize>,
    /// Number of L1 transactions in the batch.
    #[metrics(buckets = COUNT_BUCKETS)]
    pub l1_tx_count: Histogram<usize>,
    /// Number of L2 transactions in the batch.
    #[metrics(buckets = COUNT_BUCKETS)]
    pub l2_tx_count: Histogram<usize>,
    /// Bootloader-encoded bytes accumulated in the batch.
    #[metrics(buckets = COUNT_BUCKETS)]
    pub bootloader_bytes: Histogram<usize>,
    /// Consensus payload bytes accumulated in the batch.
    #[metrics(buckets = COUNT_BUCKETS)]
    pub payload_bytes: Histogram<usize>,
    /// Actual pubdata bytes included in the batch commitment.
    #[metrics(buckets = COUNT_BUCKETS)]
    pub pubdata_input_bytes: Histogram<usize>,
    /// Absolute pubdata growth reconstructed from execution state.
    #[metrics(buckets = COUNT_BUCKETS)]
    pub pubdata_growth_bytes: Histogram<usize>,
    /// Ratio of pubdata bytes to full blob capacity.
    #[metrics(buckets = Buckets::ZERO_TO_ONE, unit = Unit::Ratios)]
    pub blob_utilization: Histogram<f64>,
    /// Total gas used by the batch's execution metrics.
    #[metrics(buckets = COUNT_BUCKETS)]
    pub gas_used: Histogram<usize>,
}

impl BatchTelemetryMetrics {
    fn record_tx_finality_delay(&self, manager: &UpdatesManager) {
        let seal_timestamp_ms = unix_timestamp_ms();

        for block in manager.pending_l2_blocks() {
            for tx in &block.executed_transactions {
                let delay_ms =
                    seal_timestamp_ms.saturating_sub(tx.transaction.received_timestamp_ms);
                let delay = Duration::from_millis(delay_ms);
                self.tx_finality_delay[&TxExecutionType::from_is_l1(tx.transaction.is_l1())]
                    .observe(delay);
            }
        }
    }

    pub(crate) fn record_batch_summary(
        &self,
        manager: &UpdatesManager,
        finished_batch: &FinishedL1Batch,
        seal_started_at: Instant,
    ) {
        let tx_count = manager.pending_executed_transactions_len();
        let l1_tx_count = manager.pending_l1_transactions_len();
        let l2_tx_count = tx_count.saturating_sub(l1_tx_count);
        let bootloader_bytes = manager.pending_txs_encoding_size();
        let payload_bytes = manager.pending_payload_encoding_size();
        let gas_used = manager.pending_execution_metrics().gas_used as usize;
        let pubdata_input_bytes = finished_batch
            .pubdata_input
            .as_ref()
            .map_or(0, |pubdata| pubdata.len());
        let pubdata_growth_bytes = finished_batch
            .final_execution_state
            .pubdata_costs
            .iter()
            .map(|value| value.unsigned_abs() as usize)
            .sum::<usize>();
        let blob_capacity = if pubdata_input_bytes == 0 {
            0
        } else {
            pubdata_input_bytes.div_ceil(kzg::ZK_SYNC_BYTES_PER_BLOB) * kzg::ZK_SYNC_BYTES_PER_BLOB
        };
        let blob_utilization = if blob_capacity == 0 {
            0.0
        } else {
            pubdata_input_bytes as f64 / blob_capacity as f64
        };

        self.record_tx_finality_delay(manager);
        self.seal_time.observe(seal_started_at.elapsed());
        self.tx_count.observe(tx_count);
        self.l1_tx_count.observe(l1_tx_count);
        self.l2_tx_count.observe(l2_tx_count);
        self.bootloader_bytes.observe(bootloader_bytes);
        self.payload_bytes.observe(payload_bytes);
        self.pubdata_input_bytes.observe(pubdata_input_bytes);
        self.pubdata_growth_bytes.observe(pubdata_growth_bytes);
        self.blob_utilization.observe(blob_utilization);
        self.gas_used.observe(gas_used);

        let batch_fee_input = manager.batch_fee_input();
        let base_fee_per_gas = manager.base_fee_per_gas();
        let estimated_cost_per_tx = if tx_count == 0 {
            0u128
        } else {
            let total_gas_cost =
                manager.pending_execution_metrics().gas_used as u128 * base_fee_per_gas as u128;
            let total_pubdata_cost =
                pubdata_input_bytes as u128 * batch_fee_input.fair_pubdata_price() as u128;
            (total_gas_cost + total_pubdata_cost) / tx_count as u128
        };

        tracing::info!(
            l1_batch = %manager.l1_batch_number(),
            seal_time = ?seal_started_at.elapsed(),
            tx_count,
            l1_tx_count,
            l2_tx_count,
            bootloader_bytes,
            payload_bytes,
            pubdata_input_bytes,
            pubdata_growth_bytes,
            blob_capacity,
            blob_utilization = %blob_utilization,
            gas_used,
            base_fee_per_gas,
            fair_l2_gas_price = batch_fee_input.fair_l2_gas_price(),
            fair_pubdata_price = batch_fee_input.fair_pubdata_price(),
            l1_gas_price = batch_fee_input.l1_gas_price(),
            interop_fee = %manager.interop_fee(),
            pubdata_limit = ?manager.pubdata_limit(),
            settlement_layer = ?manager.settlement_layer(),
            estimated_cost_per_tx = %estimated_cost_per_tx,
            "Batch telemetry snapshot"
        );
    }
}

#[vise::register]
pub(crate) static BATCH_TELEMETRY_METRICS: vise::Global<BatchTelemetryMetrics> =
    vise::Global::new();
