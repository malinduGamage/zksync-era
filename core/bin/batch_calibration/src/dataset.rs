use std::{
    collections::{BTreeMap, HashMap},
    fs::File,
    io::{BufRead, BufReader},
    path::Path,
};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use sqlx::types::chrono::{NaiveDateTime, Utc};
use sqlx::Row;
use zksync_basic_types::protocol_version::{
    ProtocolSemanticVersion, ProtocolVersionId, VersionPatch,
};
use zksync_dal::{ConnectionPool, Core, CoreDal};
use zksync_prover_dal::{ConnectionPool as ProverConnectionPool, Prover, ProverDal};

use crate::types::{BatchTargets, FeatureValues, ProtocolVersionKey};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelemetrySnapshotRow {
    pub l1_batch_number: u32,
    pub protocol_version: Option<ProtocolVersionKey>,
    pub sealed_at_ms: Option<i64>,
    pub seal_time_ms: Option<f64>,
    pub tx_count: Option<f64>,
    pub l1_tx_count: Option<f64>,
    pub l2_tx_count: Option<f64>,
    pub bootloader_bytes: Option<f64>,
    pub payload_bytes: Option<f64>,
    pub pubdata_input_bytes: Option<f64>,
    pub pubdata_growth_bytes: Option<f64>,
    pub gas_used: Option<f64>,
    pub blob_utilization: Option<f64>,
    pub circuit_usage: Option<f64>,
    pub keccak_rounds: Option<f64>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TelemetrySnapshot {
    pub rows: Vec<TelemetrySnapshotRow>,
}

impl TelemetrySnapshot {
    pub fn load(path: &Path) -> Result<Self> {
        let file =
            File::open(path).with_context(|| format!("opening telemetry snapshot {path:?}"))?;
        let reader = BufReader::new(file);
        let mut rows = Vec::new();
        for line in reader.lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            if let Ok(row) = serde_json::from_str::<TelemetrySnapshotRow>(&line) {
                rows.push(row);
                continue;
            }
            let parsed: TelemetrySnapshot = serde_json::from_str(&line)
                .with_context(|| format!("parsing telemetry snapshot row from {path:?}"))?;
            rows.extend(parsed.rows);
        }
        Ok(Self { rows })
    }

    pub fn by_batch(&self) -> HashMap<u32, TelemetrySnapshotRow> {
        self.rows
            .iter()
            .cloned()
            .map(|row| (row.l1_batch_number, row))
            .collect()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchDatasetRow {
    pub l1_batch_number: u32,
    pub protocol_version: ProtocolVersionKey,
    pub sealed_at_ms: i64,
    pub first_received_at_ms: Option<i64>,
    pub last_received_at_ms: Option<i64>,
    pub feature_values: FeatureValues,
    pub targets: BatchTargets,
}

#[derive(Debug, Clone, sqlx::FromRow)]
struct BatchBaseRow {
    number: i64,
    protocol_version: Option<i32>,
    timestamp: i64,
    tx_count: i64,
    l1_tx_count: i64,
    l2_tx_count: i64,
    l2_l1_logs: i64,
    pubdata_bytes: i64,
    storage_writes: i64,
    initial_storage_writes: i64,
    repeated_storage_writes: i64,
    priority_tx_count: i64,
    upgrade_tx_count: i64,
    first_received_at: Option<NaiveDateTime>,
    last_received_at: Option<NaiveDateTime>,
    commit_gas: Option<i64>,
    verify_gas: Option<i64>,
    finalize_gas: Option<i64>,
}

#[derive(Debug, Clone, sqlx::FromRow)]
struct ProvingWindowRow {
    l1_batch_number: i64,
    started_at_secs: Option<f64>,
    finished_at_secs: Option<f64>,
}

#[derive(Debug, Clone, sqlx::FromRow)]
struct CompressionRow {
    l1_batch_number: i64,
    time_taken_secs: Option<f64>,
}

pub struct BatchDatasetBuilder {
    telemetry: HashMap<u32, TelemetrySnapshotRow>,
}

impl BatchDatasetBuilder {
    pub fn new(telemetry: Option<TelemetrySnapshot>) -> Self {
        Self {
            telemetry: telemetry
                .map(|snapshot| snapshot.by_batch())
                .unwrap_or_default(),
        }
    }

    pub async fn build(
        &self,
        core_pool: &ConnectionPool<Core>,
        prover_pool: &ProverConnectionPool<Prover>,
        start_batch: Option<u32>,
        end_batch: Option<u32>,
        protocol_version: Option<&str>,
    ) -> Result<Vec<BatchDatasetRow>> {
        let mut core_conn = core_pool.connection().await?;
        let mut prover_conn = prover_pool.connection().await?;

        let base_rows = sqlx::query_as::<_, BatchBaseRow>(
            r#"
            WITH tx_agg AS (
                SELECT
                    l1_batch_number,
                    COUNT(*)::bigint AS tx_count,
                    COUNT(*) FILTER (WHERE is_priority)::bigint AS l1_tx_count,
                    COUNT(*) FILTER (WHERE NOT is_priority)::bigint AS l2_tx_count,
                    COUNT(*) FILTER (WHERE is_priority)::bigint AS priority_tx_count,
                    COUNT(*) FILTER (WHERE upgrade_id IS NOT NULL)::bigint AS upgrade_tx_count,
                    COALESCE(SUM(COALESCE(NULLIF(execution_info->>'storage_writes', '')::bigint, 0)), 0)::bigint AS storage_writes,
                    MIN(received_at) AS first_received_at,
                    MAX(received_at) AS last_received_at
                FROM transactions
                WHERE l1_batch_number IS NOT NULL
                GROUP BY l1_batch_number
            ),
            storage_agg AS (
                SELECT
                    miniblocks.l1_batch_number,
                    COUNT(*)::bigint AS storage_log_count,
                    COUNT(DISTINCT storage_logs.hashed_key)::bigint AS distinct_storage_keys
                FROM storage_logs
                JOIN miniblocks
                    ON miniblocks.number = storage_logs.miniblock_number
                GROUP BY miniblocks.l1_batch_number
            ),
            receipt_agg AS (
                SELECT
                    l1_batches.number,
                    MAX(CASE WHEN commit_history.tx_type = 'CommitBlocks' THEN commit_history.gas_used END) AS commit_gas,
                    MAX(CASE WHEN prove_history.tx_type = 'PublishProofBlocksOnchain' THEN prove_history.gas_used END) AS verify_gas,
                    MAX(CASE WHEN execute_history.tx_type = 'ExecuteBlocks' THEN execute_history.gas_used END) AS finalize_gas
                FROM l1_batches
                LEFT JOIN eth_txs_history AS commit_history
                    ON commit_history.eth_tx_id = l1_batches.eth_commit_tx_id
                LEFT JOIN eth_txs_history AS prove_history
                    ON prove_history.eth_tx_id = l1_batches.eth_prove_tx_id
                LEFT JOIN eth_txs_history AS execute_history
                    ON execute_history.eth_tx_id = l1_batches.eth_execute_tx_id
                GROUP BY l1_batches.number
            )
            SELECT
                l1_batches.number,
                l1_batches.protocol_version,
                l1_batches.timestamp,
                COALESCE(tx_agg.tx_count, (l1_batches.l1_tx_count + l1_batches.l2_tx_count)::bigint, 0) AS tx_count,
                COALESCE(tx_agg.l1_tx_count, l1_batches.l1_tx_count::bigint, 0) AS l1_tx_count,
                COALESCE(tx_agg.l2_tx_count, l1_batches.l2_tx_count::bigint, 0) AS l2_tx_count,
                COALESCE(array_length(l1_batches.l2_to_l1_logs, 1), 0)::bigint AS l2_l1_logs,
                COALESCE(octet_length(l1_batches.pubdata_input), 0)::bigint AS pubdata_bytes,
                COALESCE(tx_agg.storage_writes, 0) AS storage_writes,
                COALESCE(storage_agg.distinct_storage_keys, 0) AS initial_storage_writes,
                GREATEST(COALESCE(storage_agg.storage_log_count, 0) - COALESCE(storage_agg.distinct_storage_keys, 0), 0) AS repeated_storage_writes,
                COALESCE(tx_agg.priority_tx_count, 0) AS priority_tx_count,
                COALESCE(tx_agg.upgrade_tx_count, 0) AS upgrade_tx_count,
                tx_agg.first_received_at,
                tx_agg.last_received_at,
                receipt_agg.commit_gas,
                receipt_agg.verify_gas,
                receipt_agg.finalize_gas
            FROM l1_batches
            LEFT JOIN tx_agg ON tx_agg.l1_batch_number = l1_batches.number
            LEFT JOIN storage_agg ON storage_agg.l1_batch_number = l1_batches.number
            LEFT JOIN receipt_agg ON receipt_agg.number = l1_batches.number
            WHERE
                l1_batches.is_sealed = TRUE
                AND ($1::INT IS NULL OR l1_batches.number >= $1)
                AND ($2::INT IS NULL OR l1_batches.number <= $2)
                AND ($3::TEXT IS NULL OR l1_batches.protocol_version::TEXT = $3)
            ORDER BY l1_batches.number
            "#,
        )
        .bind(start_batch.map(|v| v as i32))
        .bind(end_batch.map(|v| v as i32))
        .bind(protocol_version)
        .fetch_all(core_conn.conn())
        .await
        .context("loading batch aggregates")?;

        let proving_rows = sqlx::query_as::<_, ProvingWindowRow>(
            r#"
            WITH jobs AS (
                SELECT
                    l1_batch_number,
                    EXTRACT(EPOCH FROM COALESCE(processing_started_at, created_at))::float8 AS started_at_secs,
                    EXTRACT(EPOCH FROM COALESCE(processing_started_at, created_at))::float8
                        + EXTRACT(EPOCH FROM COALESCE(time_taken, '00:00:00'::time))::float8 AS finished_at_secs
                FROM prover_jobs_fri
                WHERE status = 'successful'
                UNION ALL
                SELECT
                    l1_batch_number,
                    EXTRACT(EPOCH FROM COALESCE(processing_started_at, created_at))::float8 AS started_at_secs,
                    EXTRACT(EPOCH FROM COALESCE(processing_started_at, created_at))::float8
                        + EXTRACT(EPOCH FROM COALESCE(time_taken, '00:00:00'::time))::float8 AS finished_at_secs
                FROM witness_inputs_fri
                WHERE status = 'successful'
            )
            SELECT
                l1_batch_number,
                MIN(started_at_secs) AS started_at_secs,
                MAX(finished_at_secs) AS finished_at_secs
            FROM jobs
            GROUP BY l1_batch_number
            "#,
        )
        .fetch_all(prover_conn.conn())
        .await
        .context("loading proving windows")?;

        let compression_rows = sqlx::query_as::<_, CompressionRow>(
            r#"
            SELECT
                l1_batch_number,
                EXTRACT(EPOCH FROM time_taken)::float8 AS time_taken_secs
            FROM proof_compression_jobs_fri
            WHERE status = 'successful'
            "#,
        )
        .fetch_all(prover_conn.conn())
        .await
        .context("loading proof compression times")?;

        let proving_by_batch: HashMap<_, _> = proving_rows
            .into_iter()
            .map(|row| (row.l1_batch_number as u32, row))
            .collect();
        let compression_by_batch: HashMap<_, _> = compression_rows
            .into_iter()
            .map(|row| (row.l1_batch_number as u32, row))
            .collect();

        let mut dataset = Vec::with_capacity(base_rows.len());
        for row in base_rows {
            let batch_number = row.number as u32;
            let sealed_at_ms = self
                .telemetry
                .get(&batch_number)
                .and_then(|telemetry| telemetry.sealed_at_ms)
                .unwrap_or_else(|| row.timestamp.saturating_mul(1000));
            let telemetry = self.telemetry.get(&batch_number);

            let protocol_version = if let Some(protocol_version) = row.protocol_version {
                ProtocolVersionKey::new(protocol_version as u16, 0)
            } else if let Some(telemetry) = telemetry.and_then(|t| t.protocol_version) {
                telemetry
            } else {
                ProtocolVersionKey::new(0, 0)
            };

            let proving_time_ms = proving_by_batch.get(&batch_number).and_then(|value| {
                match (value.started_at_secs, value.finished_at_secs) {
                    (Some(started_at), Some(finished_at)) => {
                        Some((finished_at - started_at) * 1000.0)
                    }
                    _ => None,
                }
            });
            let compression_time_ms = compression_by_batch
                .get(&batch_number)
                .and_then(|row| row.time_taken_secs)
                .map(|secs| secs * 1000.0);

            let feature_values = FeatureValues {
                tx_count: row.tx_count as f64,
                pubdata_bytes: telemetry
                    .and_then(|t| t.pubdata_input_bytes.or(t.payload_bytes))
                    .unwrap_or(row.pubdata_bytes as f64),
                storage_writes: row.storage_writes as f64,
                initial_storage_writes: row.initial_storage_writes as f64,
                repeated_storage_writes: row.repeated_storage_writes as f64,
                l2_l1_logs: row.l2_l1_logs as f64,
                keccak_rounds: telemetry.and_then(|t| t.keccak_rounds),
                circuit_usage: telemetry.and_then(|t| t.circuit_usage),
                l1_tx_count: row.l1_tx_count as f64,
                l2_tx_count: row.l2_tx_count as f64,
                priority_tx_count: row.priority_tx_count as f64,
                upgrade_tx_count: row.upgrade_tx_count as f64,
            };

            let targets = BatchTargets {
                commit_gas: row.commit_gas.map(|v| v as f64),
                verify_gas: row.verify_gas.map(|v| v as f64),
                finalize_gas: row.finalize_gas.map(|v| v as f64),
                execution_time_ms: telemetry.and_then(|t| t.seal_time_ms),
                proving_time_ms,
                compression_time_ms,
                incremental_resource_bytes: telemetry.and_then(|t| t.pubdata_growth_bytes),
            };

            dataset.push(BatchDatasetRow {
                l1_batch_number: batch_number,
                protocol_version,
                sealed_at_ms,
                first_received_at_ms: row
                    .first_received_at
                    .map(|ts| ts.and_utc().timestamp_millis()),
                last_received_at_ms: row
                    .last_received_at
                    .map(|ts| ts.and_utc().timestamp_millis()),
                feature_values,
                targets,
            });
        }

        Ok(dataset)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn telemetry_snapshot_round_trips_jsonl() {
        let snapshot = TelemetrySnapshot {
            rows: vec![TelemetrySnapshotRow {
                l1_batch_number: 7,
                protocol_version: Some(ProtocolVersionKey::new(1, 2)),
                sealed_at_ms: Some(42),
                seal_time_ms: Some(3.5),
                tx_count: Some(10.0),
                l1_tx_count: Some(1.0),
                l2_tx_count: Some(9.0),
                bootloader_bytes: Some(11.0),
                payload_bytes: Some(12.0),
                pubdata_input_bytes: Some(13.0),
                pubdata_growth_bytes: Some(14.0),
                gas_used: Some(15.0),
                blob_utilization: Some(0.25),
                circuit_usage: Some(16.0),
                keccak_rounds: Some(17.0),
            }],
        };
        let encoded = serde_json::to_string(&snapshot.rows[0]).unwrap();
        let path = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(path.path(), format!("{encoded}\n")).unwrap();
        let loaded = TelemetrySnapshot::load(path.path()).unwrap();
        assert_eq!(loaded.rows.len(), 1);
        assert_eq!(loaded.rows[0].l1_batch_number, 7);
    }
}
