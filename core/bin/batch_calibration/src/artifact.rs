use std::{
    cmp::Ordering,
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::model::{CalibratedModel, FeatureColumn, FitStatistics, ModelTarget};
use crate::types::ProtocolVersionKey;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorSummary {
    pub mean_absolute_error: f64,
    pub median_absolute_error: f64,
    pub p50_residual: f64,
    pub p95_residual: f64,
    pub worst_bucket_errors: Vec<BucketError>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BucketError {
    pub bucket: String,
    pub mae: f64,
    pub count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationArtifact {
    pub artifact_schema_version: u32,
    pub model_family: String,
    pub target: ModelTarget,
    pub protocol_version: ProtocolVersionKey,
    pub training_window: TrainingWindow,
    pub feature_list: Vec<FeatureColumn>,
    pub normalization: BTreeMap<FeatureColumn, Normalization>,
    pub coefficients: Vec<(FeatureColumn, f64)>,
    pub intercept: f64,
    pub transform: String,
    pub fit_timestamp_ms: i64,
    pub git_commit: String,
    pub sample_size: usize,
    pub validation: FitStatistics,
    pub error_summary: ErrorSummary,
    pub selected_lambda: f64,
    pub status: ArtifactStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Normalization {
    pub mean: f64,
    pub std_dev: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingWindow {
    pub first_batch: u32,
    pub last_batch: u32,
    pub first_sealed_at_ms: i64,
    pub last_sealed_at_ms: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactStatus {
    Ready,
    InsufficientData,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TargetArtifactEntry {
    pub status: ArtifactStatus,
    pub path: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProtocolVersionArtifacts {
    pub status: ArtifactStatus,
    pub targets: BTreeMap<String, TargetArtifactEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationManifest {
    pub schema_version: u32,
    pub generated_at_ms: i64,
    pub git_commit: String,
    pub versions: BTreeMap<String, ProtocolVersionArtifacts>,
}

impl CalibrationArtifact {
    pub fn from_model(
        model: &CalibratedModel,
        protocol_version: ProtocolVersionKey,
        training_window: TrainingWindow,
        sample_size: usize,
        git_commit: String,
        fit_timestamp_ms: i64,
    ) -> Self {
        let normalization = model
            .standardization
            .iter()
            .map(|stats| {
                (
                    stats.feature,
                    Normalization {
                        mean: stats.mean,
                        std_dev: stats.std_dev,
                    },
                )
            })
            .collect::<BTreeMap<_, _>>();
        Self {
            artifact_schema_version: 1,
            model_family: if model.target.is_quantile() {
                "quantile_regression".to_string()
            } else {
                "ridge_regression".to_string()
            },
            target: model.target,
            protocol_version,
            training_window,
            feature_list: model.selected_features.clone(),
            normalization,
            coefficients: model.coefficients.clone(),
            intercept: model.intercept,
            transform: model.transform.clone(),
            fit_timestamp_ms,
            git_commit,
            sample_size,
            validation: model.validation.clone(),
            error_summary: error_summary(
                &model.validation,
                &model.validation_residuals,
                &model.bucket_inputs,
            ),
            selected_lambda: model.lambda,
            status: ArtifactStatus::Ready,
        }
    }

    pub fn write_json(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).with_context(|| format!("creating {parent:?}"))?;
        }
        let bytes = serde_json::to_vec_pretty(self)?;
        fs::write(path, bytes).with_context(|| format!("writing {path:?}"))?;
        Ok(())
    }
}

fn error_summary(
    validation: &FitStatistics,
    residuals: &[f64],
    bucket_inputs: &[(f64, f64)],
) -> ErrorSummary {
    let mut absolute = residuals.iter().map(|r| r.abs()).collect::<Vec<_>>();
    absolute.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let median_absolute_error = if absolute.is_empty() {
        0.0
    } else {
        absolute[absolute.len() / 2]
    };

    let mut by_bucket: BTreeMap<String, (f64, usize)> = BTreeMap::new();
    let tx_thresholds = quantile_thresholds(bucket_inputs.iter().map(|(tx, _)| *tx).collect());
    let pubdata_thresholds =
        quantile_thresholds(bucket_inputs.iter().map(|(_, pubdata)| *pubdata).collect());
    for ((tx_count, pubdata_bytes), residual) in bucket_inputs.iter().zip(residuals.iter()) {
        let bucket = format!(
            "tx:{}|pub:{}",
            bucket_label(*tx_count, &tx_thresholds),
            bucket_label(*pubdata_bytes, &pubdata_thresholds)
        );
        let entry = by_bucket.entry(bucket).or_insert((0.0, 0));
        entry.0 += residual.abs();
        entry.1 += 1;
    }
    let mut worst_bucket_errors = by_bucket
        .into_iter()
        .map(|(bucket, (sum_abs, count))| BucketError {
            bucket,
            mae: if count == 0 {
                0.0
            } else {
                sum_abs / count as f64
            },
            count,
        })
        .collect::<Vec<_>>();
    worst_bucket_errors.sort_by(|a, b| b.mae.partial_cmp(&a.mae).unwrap_or(Ordering::Equal));
    worst_bucket_errors.truncate(3);

    ErrorSummary {
        mean_absolute_error: validation.mae,
        median_absolute_error,
        p50_residual: validation.residual_p50,
        p95_residual: validation.residual_p95,
        worst_bucket_errors,
    }
}

fn quantile_thresholds(mut values: Vec<f64>) -> (f64, f64) {
    if values.is_empty() {
        return (0.0, 0.0);
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let lower = values[((values.len() as f64 * 0.33).floor() as usize).min(values.len() - 1)];
    let upper = values[((values.len() as f64 * 0.66).floor() as usize).min(values.len() - 1)];
    (lower, upper)
}

fn bucket_label(value: f64, thresholds: &(f64, f64)) -> &'static str {
    if value <= thresholds.0 {
        "small"
    } else if value <= thresholds.1 {
        "medium"
    } else {
        "large"
    }
}

impl CalibrationManifest {
    pub fn write_json(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).with_context(|| format!("creating {parent:?}"))?;
        }
        fs::write(path, serde_json::to_vec_pretty(self)?)
            .with_context(|| format!("writing {path:?}"))?;
        Ok(())
    }
}

pub fn artifact_path(
    output_dir: &Path,
    protocol: ProtocolVersionKey,
    target: ModelTarget,
) -> PathBuf {
    output_dir
        .join("models")
        .join(format!("{}-{}", protocol.minor, protocol.patch))
        .join(format!("{}.json", target.name()))
}
