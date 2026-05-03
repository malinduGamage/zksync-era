use std::cmp::Ordering;

use serde::{Deserialize, Serialize};

use crate::{artifact::CalibrationArtifact, dataset::BatchDatasetRow, types::FeatureValues};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FeatureColumn {
    TxCount,
    PubdataBytes,
    StorageWrites,
    InitialStorageWrites,
    RepeatedStorageWrites,
    L2L1Logs,
    KeccakRounds,
    CircuitUsage,
    L1TxCount,
    L2TxCount,
    PriorityTxCount,
    UpgradeTxCount,
    TxCountXPubdataBytes,
    PubdataBytesXL2L1Logs,
    InitialXRepeatedStorageWrites,
}

impl FeatureColumn {
    pub fn all_base() -> Vec<Self> {
        vec![
            Self::TxCount,
            Self::PubdataBytes,
            Self::StorageWrites,
            Self::InitialStorageWrites,
            Self::RepeatedStorageWrites,
            Self::L2L1Logs,
            Self::KeccakRounds,
            Self::CircuitUsage,
            Self::L1TxCount,
            Self::L2TxCount,
            Self::PriorityTxCount,
            Self::UpgradeTxCount,
        ]
    }

    pub fn derived_for_incremental_resource() -> Vec<Self> {
        vec![
            Self::TxCountXPubdataBytes,
            Self::PubdataBytesXL2L1Logs,
            Self::InitialXRepeatedStorageWrites,
        ]
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::TxCount => "tx_count",
            Self::PubdataBytes => "pubdata_bytes",
            Self::StorageWrites => "storage_writes",
            Self::InitialStorageWrites => "initial_storage_writes",
            Self::RepeatedStorageWrites => "repeated_storage_writes",
            Self::L2L1Logs => "l2_l1_logs",
            Self::KeccakRounds => "keccak_rounds",
            Self::CircuitUsage => "circuit_usage",
            Self::L1TxCount => "l1_tx_count",
            Self::L2TxCount => "l2_tx_count",
            Self::PriorityTxCount => "priority_tx_count",
            Self::UpgradeTxCount => "upgrade_tx_count",
            Self::TxCountXPubdataBytes => "tx_count_x_pubdata_bytes",
            Self::PubdataBytesXL2L1Logs => "pubdata_bytes_x_l2_l1_logs",
            Self::InitialXRepeatedStorageWrites => "initial_x_repeated_storage_writes",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelTarget {
    CommitGas,
    VerifyGas,
    FinalizeGas,
    ExecutionTimeP50,
    ExecutionTimeP95,
    ProvingTimeP50,
    ProvingTimeP95,
    CompressionTimeP50,
    CompressionTimeP95,
    IncrementalResource,
}

impl ModelTarget {
    pub fn all() -> Vec<Self> {
        vec![
            Self::CommitGas,
            Self::VerifyGas,
            Self::FinalizeGas,
            Self::ExecutionTimeP50,
            Self::ExecutionTimeP95,
            Self::ProvingTimeP50,
            Self::ProvingTimeP95,
            Self::CompressionTimeP50,
            Self::CompressionTimeP95,
            Self::IncrementalResource,
        ]
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::CommitGas => "commit_gas",
            Self::VerifyGas => "verify_gas",
            Self::FinalizeGas => "finalize_gas",
            Self::ExecutionTimeP50 => "execution_time_p50",
            Self::ExecutionTimeP95 => "execution_time_p95",
            Self::ProvingTimeP50 => "proving_time_p50",
            Self::ProvingTimeP95 => "proving_time_p95",
            Self::CompressionTimeP50 => "compression_time_p50",
            Self::CompressionTimeP95 => "compression_time_p95",
            Self::IncrementalResource => "incremental_resource",
        }
    }

    pub fn is_quantile(self) -> bool {
        matches!(
            self,
            Self::ExecutionTimeP50
                | Self::ExecutionTimeP95
                | Self::ProvingTimeP50
                | Self::ProvingTimeP95
                | Self::CompressionTimeP50
                | Self::CompressionTimeP95
        )
    }

    pub fn quantile(self) -> Option<f64> {
        match self {
            Self::ExecutionTimeP50 | Self::ProvingTimeP50 | Self::CompressionTimeP50 => Some(0.5),
            Self::ExecutionTimeP95 | Self::ProvingTimeP95 | Self::CompressionTimeP95 => Some(0.95),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StandardizationStats {
    pub feature: FeatureColumn,
    pub mean: f64,
    pub std_dev: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FitStatistics {
    pub mae: f64,
    pub rmse: f64,
    pub mape: Option<f64>,
    pub smape: Option<f64>,
    pub r2: Option<f64>,
    pub coverage: Option<f64>,
    pub residual_p50: f64,
    pub residual_p95: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibratedModel {
    pub target: ModelTarget,
    pub transform: String,
    pub lambda: f64,
    pub intercept: f64,
    pub coefficients: Vec<(FeatureColumn, f64)>,
    pub standardization: Vec<StandardizationStats>,
    pub selected_features: Vec<FeatureColumn>,
    pub validation: FitStatistics,
    pub validation_residuals: Vec<f64>,
    pub bucket_inputs: Vec<(f64, f64)>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FitResult {
    pub models: Vec<CalibratedModel>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingSplit {
    pub train_indices: Vec<usize>,
    pub validation_indices: Vec<usize>,
}

fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    xs.iter().sum::<f64>() / xs.len() as f64
}

fn std_dev(xs: &[f64], mean: f64) -> f64 {
    if xs.len() < 2 {
        return 1.0;
    }
    let variance = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / xs.len() as f64;
    variance.sqrt().max(1e-9)
}

fn quantile(mut values: Vec<f64>, q: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let idx = ((values.len() - 1) as f64 * q).round() as usize;
    values[idx]
}

fn residual_stats(residuals: &[f64], actuals: &[f64], predictions: &[f64]) -> FitStatistics {
    let n = residuals.len().max(1) as f64;
    let mae = residuals.iter().map(|r| r.abs()).sum::<f64>() / n;
    let rmse = (residuals.iter().map(|r| r * r).sum::<f64>() / n).sqrt();
    let residual_p50 = quantile(residuals.to_vec(), 0.5);
    let residual_p95 = quantile(residuals.to_vec(), 0.95);
    let mape = if actuals.iter().all(|v| *v > 0.0) {
        Some(
            residuals
                .iter()
                .zip(actuals)
                .map(|(r, y)| r.abs() / y.abs())
                .sum::<f64>()
                / n,
        )
    } else {
        None
    };
    let smape = if actuals.iter().any(|v| *v != 0.0) {
        Some(
            residuals
                .iter()
                .zip(actuals)
                .zip(predictions)
                .map(|((_, y), p)| (p - y).abs() / ((p.abs() + y.abs()) / 2.0).max(1e-9))
                .sum::<f64>()
                / n,
        )
    } else {
        None
    };
    let mean_y = mean(actuals);
    let ss_tot = actuals.iter().map(|y| (y - mean_y).powi(2)).sum::<f64>();
    let ss_res = residuals.iter().map(|r| r * r).sum::<f64>();
    let r2 = if ss_tot > 0.0 {
        Some(1.0 - ss_res / ss_tot)
    } else {
        None
    };
    FitStatistics {
        mae,
        rmse,
        mape,
        smape,
        r2,
        coverage: None,
        residual_p50,
        residual_p95,
    }
}

fn select_time_folds(n: usize, folds: usize) -> Vec<TrainingSplit> {
    if n < 2 {
        return vec![];
    }
    let folds = folds.max(2).min(n);
    let chunk = (n / folds).max(1);
    let mut splits = Vec::new();
    for fold in 0..folds {
        let validation_start = fold * chunk;
        let mut validation_end = if fold == folds - 1 {
            n
        } else {
            ((fold + 1) * chunk).min(n)
        };
        if validation_start >= n {
            break;
        }
        validation_end = validation_end.max(validation_start + 1);
        let validation_indices = (validation_start..validation_end).collect::<Vec<_>>();
        let train_indices = (0..validation_start).collect::<Vec<_>>();
        if !train_indices.is_empty() && !validation_indices.is_empty() {
            splits.push(TrainingSplit {
                train_indices,
                validation_indices,
            });
        }
    }
    splits
}

fn feature_value(row: &BatchDatasetRow, feature: FeatureColumn) -> Option<f64> {
    let f = &row.feature_values;
    Some(match feature {
        FeatureColumn::TxCount => f.tx_count,
        FeatureColumn::PubdataBytes => f.pubdata_bytes,
        FeatureColumn::StorageWrites => f.storage_writes,
        FeatureColumn::InitialStorageWrites => f.initial_storage_writes,
        FeatureColumn::RepeatedStorageWrites => f.repeated_storage_writes,
        FeatureColumn::L2L1Logs => f.l2_l1_logs,
        FeatureColumn::KeccakRounds => f.keccak_rounds?,
        FeatureColumn::CircuitUsage => f.circuit_usage?,
        FeatureColumn::L1TxCount => f.l1_tx_count,
        FeatureColumn::L2TxCount => f.l2_tx_count,
        FeatureColumn::PriorityTxCount => f.priority_tx_count,
        FeatureColumn::UpgradeTxCount => f.upgrade_tx_count,
        FeatureColumn::TxCountXPubdataBytes => f.tx_count * f.pubdata_bytes,
        FeatureColumn::PubdataBytesXL2L1Logs => f.pubdata_bytes * f.l2_l1_logs,
        FeatureColumn::InitialXRepeatedStorageWrites => {
            f.initial_storage_writes * f.repeated_storage_writes
        }
    })
}

fn target_value(row: &BatchDatasetRow, target: ModelTarget) -> Option<f64> {
    match target {
        ModelTarget::CommitGas => row.targets.commit_gas,
        ModelTarget::VerifyGas => row.targets.verify_gas,
        ModelTarget::FinalizeGas => row.targets.finalize_gas,
        ModelTarget::ExecutionTimeP50 | ModelTarget::ExecutionTimeP95 => {
            row.targets.execution_time_ms
        }
        ModelTarget::ProvingTimeP50 | ModelTarget::ProvingTimeP95 => row.targets.proving_time_ms,
        ModelTarget::CompressionTimeP50 | ModelTarget::CompressionTimeP95 => {
            row.targets.compression_time_ms
        }
        ModelTarget::IncrementalResource => row.targets.incremental_resource_bytes,
    }
}

fn solve_linear_system(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Option<Vec<f64>> {
    let n = b.len();
    for i in 0..n {
        let mut pivot = i;
        for r in (i + 1)..n {
            if a[r][i].abs() > a[pivot][i].abs() {
                pivot = r;
            }
        }
        if a[pivot][i].abs() < 1e-12 {
            return None;
        }
        a.swap(i, pivot);
        b.swap(i, pivot);

        let inv = 1.0 / a[i][i];
        for j in i..n {
            a[i][j] *= inv;
        }
        b[i] *= inv;

        for r in 0..n {
            if r == i {
                continue;
            }
            let factor = a[r][i];
            for j in i..n {
                a[r][j] -= factor * a[i][j];
            }
            b[r] -= factor * b[i];
        }
    }
    Some(b)
}

fn fit_ridge(x: &[Vec<f64>], y: &[f64], lambda: f64) -> Option<Vec<f64>> {
    let rows = x.len();
    let cols = x.first()?.len();
    let mut xtx = vec![vec![0.0; cols]; cols];
    let mut xty = vec![0.0; cols];
    for (row, target) in x.iter().zip(y) {
        for i in 0..cols {
            xty[i] += row[i] * target;
            for j in 0..cols {
                xtx[i][j] += row[i] * row[j];
            }
        }
    }
    for i in 1..cols {
        xtx[i][i] += lambda * rows as f64;
    }
    solve_linear_system(xtx, xty)
}

fn fit_quantile(x: &[Vec<f64>], y: &[f64], tau: f64, lambda: f64) -> Vec<f64> {
    let cols = x.first().map(|row| row.len()).unwrap_or(0);
    let mut weights = vec![0.0; cols];
    let lr = 0.01;
    for _ in 0..2_000 {
        let mut grad = vec![0.0; cols];
        for (row, target) in x.iter().zip(y) {
            let pred = dot(row, &weights);
            let residual = target - pred;
            let subgrad = if residual > 0.0 { -tau } else { 1.0 - tau };
            for j in 0..cols {
                grad[j] += subgrad * row[j];
            }
        }
        for j in 1..cols {
            grad[j] += lambda * weights[j];
        }
        for j in 0..cols {
            weights[j] -= lr * grad[j] / x.len().max(1) as f64;
        }
    }
    weights
}

fn dot(row: &[f64], weights: &[f64]) -> f64 {
    row.iter().zip(weights).map(|(a, b)| a * b).sum()
}

fn build_matrix(
    rows: &[BatchDatasetRow],
    features: &[FeatureColumn],
    target: ModelTarget,
) -> (Vec<Vec<f64>>, Vec<f64>, Vec<usize>) {
    let mut x = Vec::new();
    let mut y = Vec::new();
    let mut kept_rows = Vec::new();
    for (idx, row) in rows.iter().enumerate() {
        let Some(target_value) = target_value(row, target) else {
            continue;
        };
        let mut feature_row = vec![1.0];
        let mut complete = true;
        for feature in features {
            if let Some(value) = feature_value(row, *feature) {
                feature_row.push(value);
            } else {
                complete = false;
                break;
            }
        }
        if complete {
            x.push(feature_row);
            y.push(target_value);
            kept_rows.push(idx);
        }
    }
    (x, y, kept_rows)
}

fn standardize_features(x: &mut [Vec<f64>]) -> Vec<(f64, f64)> {
    if x.is_empty() {
        return vec![];
    }
    let feature_count = x[0].len();
    let mut stats = Vec::with_capacity(feature_count.saturating_sub(1));
    for col in 1..feature_count {
        let values = x.iter().map(|row| row[col]).collect::<Vec<_>>();
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let std = (values
            .iter()
            .map(|value| (value - mean).powi(2))
            .sum::<f64>()
            / values.len() as f64)
            .sqrt()
            .max(1e-9);
        for row in x.iter_mut() {
            row[col] = (row[col] - mean) / std;
        }
        stats.push((mean, std));
    }
    stats
}

fn predict_linear(weights: &[f64], x: &[Vec<f64>]) -> Vec<f64> {
    x.iter().map(|row| dot(row, weights)).collect()
}

fn transform_target(target: f64, raw: bool) -> f64 {
    if raw {
        target
    } else {
        (target.max(0.0) + 1.0).ln()
    }
}

fn invert_target(target: f64, raw: bool) -> f64 {
    if raw {
        target
    } else {
        target.exp() - 1.0
    }
}

fn feature_set_for_target(target: ModelTarget) -> Vec<FeatureColumn> {
    let mut features = FeatureColumn::all_base();
    if target == ModelTarget::IncrementalResource {
        features.extend(FeatureColumn::derived_for_incremental_resource());
    }
    features
}

struct CandidateEvaluation {
    score: f64,
    residuals: Vec<f64>,
    actuals: Vec<f64>,
    preds: Vec<f64>,
    bucket_inputs: Vec<(f64, f64)>,
    transform: String,
    lambda: f64,
    selected_features: Vec<FeatureColumn>,
    standardization: Vec<StandardizationStats>,
}

fn evaluate_linear_candidate(
    x_raw: &[Vec<f64>],
    y_raw: &[f64],
    features: &[FeatureColumn],
    splits: &[TrainingSplit],
    raw_transform: bool,
    lambda: f64,
) -> Option<CandidateEvaluation> {
    let mut all_residuals = Vec::new();
    let mut all_actuals = Vec::new();
    let mut all_preds = Vec::new();
    let mut bucket_inputs = Vec::new();

    for split in splits {
        let mut train_x = split
            .train_indices
            .iter()
            .map(|&idx| x_raw[idx].clone())
            .collect::<Vec<_>>();
        let train_y = split
            .train_indices
            .iter()
            .map(|&idx| transform_target(y_raw[idx], raw_transform))
            .collect::<Vec<_>>();
        let stats = standardize_features(&mut train_x);
        let weights = fit_ridge(&train_x, &train_y, lambda).unwrap_or_else(|| {
            let mut fallback = vec![0.0; train_x[0].len()];
            fallback[0] = mean(&train_y);
            fallback
        });

        let mut val_x = split
            .validation_indices
            .iter()
            .map(|&idx| x_raw[idx].clone())
            .collect::<Vec<_>>();
        for row in &mut val_x {
            for (col_idx, (mean, std)) in stats.iter().enumerate() {
                row[col_idx + 1] = (row[col_idx + 1] - mean) / std;
            }
        }
        let val_y = split
            .validation_indices
            .iter()
            .map(|&idx| y_raw[idx])
            .collect::<Vec<_>>();
        bucket_inputs.extend(split.validation_indices.iter().map(|&idx| {
            let row = &x_raw[idx];
            (row[1], row[2])
        }));
        let preds = predict_linear(&weights, &val_x)
            .into_iter()
            .map(|pred| invert_target(pred, raw_transform).max(0.0))
            .collect::<Vec<_>>();
        all_residuals.extend(
            preds
                .iter()
                .zip(val_y.iter())
                .map(|(pred, actual)| pred - actual),
        );
        all_actuals.extend(val_y.iter().copied());
        all_preds.extend(preds);
    }

    if all_residuals.is_empty() {
        return None;
    }

    let score =
        (all_residuals.iter().map(|r| r * r).sum::<f64>() / all_residuals.len() as f64).sqrt();
    let mut full_x = x_raw.to_vec();
    let standardization_pairs = standardize_features(&mut full_x);
    Some(CandidateEvaluation {
        score,
        residuals: all_residuals,
        actuals: all_actuals,
        preds: all_preds,
        bucket_inputs,
        transform: if raw_transform { "raw" } else { "log1p" }.to_string(),
        lambda,
        selected_features: features.to_vec(),
        standardization: features
            .iter()
            .zip(standardization_pairs.into_iter())
            .map(|(feature, (mean, std_dev))| StandardizationStats {
                feature: *feature,
                mean,
                std_dev,
            })
            .collect(),
    })
}

fn evaluate_quantile_candidate(
    x_raw: &[Vec<f64>],
    y_raw: &[f64],
    features: &[FeatureColumn],
    splits: &[TrainingSplit],
    tau: f64,
    lambda: f64,
) -> Option<CandidateEvaluation> {
    let mut all_residuals = Vec::new();
    let mut all_actuals = Vec::new();
    let mut all_preds = Vec::new();
    let mut bucket_inputs = Vec::new();

    for split in splits {
        let mut train_x = split
            .train_indices
            .iter()
            .map(|&idx| x_raw[idx].clone())
            .collect::<Vec<_>>();
        let train_y = split
            .train_indices
            .iter()
            .map(|&idx| y_raw[idx])
            .collect::<Vec<_>>();
        let stats = standardize_features(&mut train_x);
        let weights = fit_quantile(&train_x, &train_y, tau, lambda);

        let mut val_x = split
            .validation_indices
            .iter()
            .map(|&idx| x_raw[idx].clone())
            .collect::<Vec<_>>();
        for row in &mut val_x {
            for (col_idx, (mean, std)) in stats.iter().enumerate() {
                row[col_idx + 1] = (row[col_idx + 1] - mean) / std;
            }
        }
        let val_y = split
            .validation_indices
            .iter()
            .map(|&idx| y_raw[idx])
            .collect::<Vec<_>>();
        bucket_inputs.extend(split.validation_indices.iter().map(|&idx| {
            let row = &x_raw[idx];
            (row[1], row[2])
        }));
        let preds = predict_linear(&weights, &val_x);
        all_residuals.extend(
            preds
                .iter()
                .zip(val_y.iter())
                .map(|(pred, actual)| pred - actual),
        );
        all_actuals.extend(val_y.iter().copied());
        all_preds.extend(preds);
    }

    if all_residuals.is_empty() {
        return None;
    }

    let score =
        (all_residuals.iter().map(|r| r * r).sum::<f64>() / all_residuals.len() as f64).sqrt();
    let mut full_x = x_raw.to_vec();
    let standardization_pairs = standardize_features(&mut full_x);
    Some(CandidateEvaluation {
        score,
        residuals: all_residuals,
        actuals: all_actuals,
        preds: all_preds,
        bucket_inputs,
        transform: "raw".to_string(),
        lambda,
        selected_features: features.to_vec(),
        standardization: features
            .iter()
            .zip(standardization_pairs.into_iter())
            .map(|(feature, (mean, std_dev))| StandardizationStats {
                feature: *feature,
                mean,
                std_dev,
            })
            .collect(),
    })
}

pub fn fit_target_models(rows: &[BatchDatasetRow], folds: usize) -> Vec<CalibratedModel> {
    let mut models = Vec::new();
    for target in ModelTarget::all() {
        let features = feature_set_for_target(target);
        let (x_raw, y_raw, _) = build_matrix(rows, &features, target);
        if x_raw.len() < 5 {
            continue;
        }
        let splits = select_time_folds(x_raw.len(), folds);
        if splits.is_empty() {
            continue;
        }
        if target.is_quantile() {
            if let Some(tau) = target.quantile() {
                let mut best: Option<CandidateEvaluation> = None;
                for &lambda in &[0.0, 0.1, 1.0] {
                    if let Some(candidate) =
                        evaluate_quantile_candidate(&x_raw, &y_raw, &features, &splits, tau, lambda)
                    {
                        if best
                            .as_ref()
                            .map(|b| candidate.score < b.score)
                            .unwrap_or(true)
                        {
                            best = Some(candidate);
                        }
                    }
                }
                let Some(best) = best else {
                    continue;
                };

                let mut x = x_raw.clone();
                standardize_features(&mut x);
                let weights = fit_quantile(&x, &y_raw, tau, best.lambda);
                let mut validation = residual_stats(&best.residuals, &best.actuals, &best.preds);
                validation.coverage = Some(
                    best.preds
                        .iter()
                        .zip(best.actuals.iter())
                        .filter(|(pred, actual)| *actual <= *pred)
                        .count() as f64
                        / best.actuals.len() as f64,
                );
                models.push(CalibratedModel {
                    target,
                    transform: "raw".to_string(),
                    lambda: best.lambda,
                    intercept: weights[0],
                    coefficients: features
                        .iter()
                        .enumerate()
                        .map(|(idx, feature)| (*feature, weights[idx + 1]))
                        .collect(),
                    standardization: best.standardization,
                    selected_features: best.selected_features,
                    validation,
                    validation_residuals: best.residuals,
                    bucket_inputs: best.bucket_inputs,
                });
            }
            continue;
        }

        let mut best: Option<CandidateEvaluation> = None;
        for &raw_transform in &[true, false] {
            for &lambda in &[0.0, 0.1, 1.0, 10.0] {
                if let Some(candidate) = evaluate_linear_candidate(
                    &x_raw,
                    &y_raw,
                    &features,
                    &splits,
                    raw_transform,
                    lambda,
                ) {
                    if best
                        .as_ref()
                        .map(|b| candidate.score < b.score)
                        .unwrap_or(true)
                    {
                        best = Some(candidate);
                    }
                }
            }
        }
        let Some(best) = best else {
            continue;
        };

        let mut x = x_raw.clone();
        standardize_features(&mut x);
        let y_fit = y_raw
            .iter()
            .map(|value| transform_target(*value, best.transform == "raw"))
            .collect::<Vec<_>>();
        let weights = fit_ridge(&x, &y_fit, best.lambda).unwrap_or_else(|| {
            let mut fallback = vec![0.0; x[0].len()];
            fallback[0] = mean(&y_fit);
            fallback
        });
        let validation = residual_stats(&best.residuals, &best.actuals, &best.preds);

        models.push(CalibratedModel {
            target,
            transform: best.transform,
            lambda: best.lambda,
            intercept: weights[0],
            coefficients: features
                .iter()
                .enumerate()
                .map(|(idx, feature)| (*feature, weights[idx + 1]))
                .collect(),
            standardization: best.standardization,
            selected_features: best.selected_features,
            validation,
            validation_residuals: best.residuals,
            bucket_inputs: best.bucket_inputs,
        });
    }

    models
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{BatchTargets, FeatureValues, ProtocolVersionKey};

    fn sample_row(value: f64) -> BatchDatasetRow {
        BatchDatasetRow {
            l1_batch_number: value as u32,
            protocol_version: ProtocolVersionKey::new(1, 0),
            sealed_at_ms: value as i64,
            first_received_at_ms: None,
            last_received_at_ms: None,
            feature_values: FeatureValues {
                tx_count: value,
                pubdata_bytes: value * 2.0,
                storage_writes: value * 3.0,
                initial_storage_writes: value,
                repeated_storage_writes: value * 2.0,
                l2_l1_logs: value,
                keccak_rounds: Some(value),
                circuit_usage: Some(value),
                l1_tx_count: value,
                l2_tx_count: value,
                priority_tx_count: value,
                upgrade_tx_count: 0.0,
            },
            targets: BatchTargets {
                commit_gas: Some(10.0 * value),
                verify_gas: Some(11.0 * value),
                finalize_gas: Some(12.0 * value),
                execution_time_ms: Some(13.0 * value),
                proving_time_ms: Some(14.0 * value),
                compression_time_ms: Some(15.0 * value),
                incremental_resource_bytes: Some(16.0 * value),
            },
        }
    }

    #[test]
    fn fits_non_empty_dataset() {
        let rows = (1..=20).map(|i| sample_row(i as f64)).collect::<Vec<_>>();
        let models = fit_target_models(&rows, 4);
        assert!(!models.is_empty());
        assert!(models
            .iter()
            .any(|model| model.target == ModelTarget::CommitGas));
    }
}
