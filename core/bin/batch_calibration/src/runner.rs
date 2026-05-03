use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use anyhow::{Context, Result};
use chrono::Utc;
use zksync_basic_types::protocol_version::{
    ProtocolSemanticVersion, ProtocolVersionId, VersionPatch,
};
use zksync_dal::{ConnectionPool, Core};
use zksync_prover_dal::{ConnectionPool as ProverConnectionPool, Prover};

use crate::{
    artifact::{
        artifact_path, ArtifactStatus, CalibrationArtifact, CalibrationManifest,
        ProtocolVersionArtifacts, TargetArtifactEntry, TrainingWindow,
    },
    dataset::{BatchDatasetBuilder, TelemetrySnapshot},
    model::{fit_target_models, ModelTarget},
    types::ProtocolVersionKey,
};

pub async fn run(
    core_db_url: &str,
    prover_db_url: &str,
    telemetry_jsonl: Option<&Path>,
    output_dir: &Path,
    start_batch: Option<u32>,
    end_batch: Option<u32>,
    protocol_version_filter: Option<&str>,
    minimum_samples: usize,
    folds: usize,
) -> Result<()> {
    fs::create_dir_all(output_dir).with_context(|| format!("creating {output_dir:?}"))?;

    let telemetry = match telemetry_jsonl {
        Some(path) => Some(TelemetrySnapshot::load(path)?),
        None => None,
    };

    let core_pool = ConnectionPool::<Core>::builder(core_db_url.parse()?, 1)
        .build()
        .await
        .context("connecting to core DB")?;
    let prover_pool = ProverConnectionPool::<Prover>::builder(prover_db_url.parse()?, 1)
        .build()
        .await
        .context("connecting to prover DB")?;

    let builder = BatchDatasetBuilder::new(telemetry);
    let dataset = builder
        .build(
            &core_pool,
            &prover_pool,
            start_batch,
            end_batch,
            protocol_version_filter,
        )
        .await?;

    let dataset_path = output_dir.join("dataset.jsonl");
    write_dataset(&dataset_path, &dataset)?;

    let git_commit = git_commit().unwrap_or_else(|_| "unknown".to_string());
    let generated_at_ms = Utc::now().timestamp_millis();
    let mut manifest = CalibrationManifest {
        schema_version: 1,
        generated_at_ms,
        git_commit: git_commit.clone(),
        versions: BTreeMap::new(),
    };

    let mut grouped: BTreeMap<ProtocolVersionKey, Vec<_>> = BTreeMap::new();
    for row in dataset {
        grouped.entry(row.protocol_version).or_default().push(row);
    }

    for (protocol, rows) in grouped {
        let first = rows
            .first()
            .map(|row| row.l1_batch_number)
            .unwrap_or_default();
        let last = rows
            .last()
            .map(|row| row.l1_batch_number)
            .unwrap_or_default();
        let first_sealed_at_ms = rows.first().map(|row| row.sealed_at_ms).unwrap_or_default();
        let last_sealed_at_ms = rows.last().map(|row| row.sealed_at_ms).unwrap_or_default();
        let training_window = TrainingWindow {
            first_batch: first,
            last_batch: last,
            first_sealed_at_ms,
            last_sealed_at_ms,
        };

        if rows.len() < minimum_samples {
            let mut version_artifacts = BTreeMap::new();
            for target in ModelTarget::all() {
                version_artifacts.insert(
                    target.name().to_string(),
                    TargetArtifactEntry {
                        status: ArtifactStatus::InsufficientData,
                        path: None,
                    },
                );
            }
            manifest.versions.insert(
                format!("{}.{}", protocol.minor, protocol.patch),
                ProtocolVersionArtifacts {
                    status: ArtifactStatus::InsufficientData,
                    targets: version_artifacts,
                },
            );
            continue;
        }

        let models = fit_target_models(&rows, folds);
        let mut version_artifacts = BTreeMap::new();
        let fitted_targets = models
            .iter()
            .map(|model| model.target.name().to_string())
            .collect::<std::collections::BTreeSet<_>>();
        for target in ModelTarget::all() {
            if !fitted_targets.contains(target.name()) {
                version_artifacts.insert(
                    target.name().to_string(),
                    TargetArtifactEntry {
                        status: ArtifactStatus::InsufficientData,
                        path: None,
                    },
                );
            }
        }
        for model in models {
            let artifact = CalibrationArtifact::from_model(
                &model,
                protocol,
                training_window.clone(),
                rows.len(),
                git_commit.clone(),
                generated_at_ms,
            );
            let path = artifact_path(output_dir, protocol, model.target);
            artifact.write_json(&path)?;
            version_artifacts.insert(
                model.target.name().to_string(),
                TargetArtifactEntry {
                    status: ArtifactStatus::Ready,
                    path: Some(
                        path.strip_prefix(output_dir)
                            .unwrap_or(&path)
                            .to_string_lossy()
                            .to_string(),
                    ),
                },
            );
        }

        manifest.versions.insert(
            format!("{}.{}", protocol.minor, protocol.patch),
            ProtocolVersionArtifacts {
                status: ArtifactStatus::Ready,
                targets: version_artifacts,
            },
        );
    }

    manifest.write_json(&output_dir.join("manifest.json"))?;
    Ok(())
}

fn write_dataset(path: &Path, dataset: &[crate::dataset::BatchDatasetRow]) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut buffer = String::new();
    for row in dataset {
        buffer.push_str(&serde_json::to_string(row)?);
        buffer.push('\n');
    }
    fs::write(path, buffer)?;
    Ok(())
}

fn git_commit() -> Result<String> {
    let output = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .context("running git rev-parse")?;
    if !output.status.success() {
        anyhow::bail!("git rev-parse failed");
    }
    Ok(String::from_utf8(output.stdout)?.trim().to_string())
}
