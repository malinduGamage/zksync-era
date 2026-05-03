use serde::{Deserialize, Serialize};
use zksync_basic_types::protocol_version::{
    ProtocolSemanticVersion, ProtocolVersionId, VersionPatch,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd, Serialize, Deserialize)]
pub struct ProtocolVersionKey {
    pub minor: u16,
    pub patch: u32,
}

impl ProtocolVersionKey {
    pub fn new(minor: u16, patch: u32) -> Self {
        Self { minor, patch }
    }

    pub fn as_semantic_version(self) -> ProtocolSemanticVersion {
        ProtocolSemanticVersion::new(
            ProtocolVersionId::try_from(self.minor).unwrap(),
            VersionPatch(self.patch),
        )
    }
}

impl From<ProtocolSemanticVersion> for ProtocolVersionKey {
    fn from(value: ProtocolSemanticVersion) -> Self {
        Self {
            minor: value.minor as u16,
            patch: value.patch.0,
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct FeatureValues {
    pub tx_count: f64,
    pub pubdata_bytes: f64,
    pub storage_writes: f64,
    pub initial_storage_writes: f64,
    pub repeated_storage_writes: f64,
    pub l2_l1_logs: f64,
    pub keccak_rounds: Option<f64>,
    pub circuit_usage: Option<f64>,
    pub l1_tx_count: f64,
    pub l2_tx_count: f64,
    pub priority_tx_count: f64,
    pub upgrade_tx_count: f64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct BatchTargets {
    pub commit_gas: Option<f64>,
    pub verify_gas: Option<f64>,
    pub finalize_gas: Option<f64>,
    pub execution_time_ms: Option<f64>,
    pub proving_time_ms: Option<f64>,
    pub compression_time_ms: Option<f64>,
    pub incremental_resource_bytes: Option<f64>,
}
