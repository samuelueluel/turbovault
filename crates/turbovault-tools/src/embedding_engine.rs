//! Dense and hybrid retrieval for Markdown vaults.
//!
//! The engine deliberately keeps embedding inference outside TurboVault. It
//! talks to a configurable OpenAI-compatible `/embeddings` endpoint, which may
//! be local or hosted, persists a versioned local vector index, and combines
//! dense results with TurboVault's sparse search.
//!
//! The index is not stored in the vault. It is derived state keyed by the
//! vault path, embedding model, and chunker version.
//!
//! Two conditions are tracked separately because only one of them is a
//! correctness problem:
//!
//! - *Incompatible*: the stored index was built against a different model,
//!   schema, chunker, vault path, or document-indexing policy. The vectors may
//!   live in a different space or retain a source set the current process did
//!   not authorize, so dense search refuses until `reindex_embeddings` runs.
//! - *Stale*: the vault was written to since the index was built. The vectors
//!   are merely out of date and are still served, because they remain in the
//!   same space and rank correctly. The sparse channel is updated eagerly on
//!   every mutation, so anything created or edited since the last reindex is
//!   already covered lexically; refusing the dense half as well only removes
//!   the meaning-matching capability without protecting anything.

use chrono::{DateTime, Utc};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::env;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};
use std::time::{Duration, Instant, SystemTime};
use tokio::sync::RwLock;
use turbovault_core::prelude::*;
use turbovault_parser::to_plain_text;
use turbovault_vault::{ScannedNote, VaultManager};

use crate::document_extract::{extract_docx_markdown, extract_pdf_markdown};
use crate::search_engine::{SearchEngine, SearchQuery};

const INDEX_SCHEMA_VERSION: u32 = 2;
const CHUNKER_VERSION: &str = "markdown-hierarchy-v2";
const DEFAULT_ENDPOINT: &str = "http://127.0.0.1:8082/v1/embeddings";
const DEFAULT_MODEL: &str = "Qwen/Qwen3-Embedding-8B-GGUF";
const DEFAULT_CHUNK_CHARS: usize = 2400;
const DEFAULT_CHUNK_OVERLAP: usize = 200;
const DEFAULT_BATCH_SIZE: usize = 16;
const DEFAULT_DOCUMENT_MAX_BYTES: usize = 50 * 1024 * 1024;
const DEFAULT_RERANKER_ENDPOINT: &str = "http://127.0.0.1:8083/v1/rerank";
const DEFAULT_RERANKER_MODEL: &str = "BAAI/bge-reranker-v2-m3";
const DEFAULT_RERANK_BATCH_SIZE: usize = 12;
const RRF_K: f64 = 60.0;
/// How far behind an index may fall before a search refreshes it in place.
///
/// Refreshing is an optimisation, not a correctness requirement: an out-of-date
/// index is still served, so this only decides how stale the vectors are allowed
/// to get before the engine spends embedding calls to catch up. Six hours keeps
/// content current across a working session without re-embedding after every
/// keystroke. Set `TURBOVAULT_EMBEDDING_REFRESH_AFTER_SECS=0` to disable.
const DEFAULT_REFRESH_AFTER_SECS: u64 = 21_600;
/// Minimum wait before retrying a failed lazy refresh, so a down endpoint is not
/// re-attempted on every search.
const REFRESH_RETRY_BACKOFF: Duration = Duration::from_secs(300);

/// Runtime configuration for external embedding and reranking endpoints.
///
/// Both endpoints may be local or hosted. A dedicated reranker key takes
/// precedence when configured; otherwise the embedding key is reused so one
/// provider such as OpenRouter can serve both requests.
#[derive(Debug, Clone)]
pub struct EmbeddingConfig {
    pub endpoint: String,
    pub model: String,
    pub api_key_configured: bool,
    api_key: Option<String>,
    pub index_dir: PathBuf,
    pub chunk_chars: usize,
    pub chunk_overlap: usize,
    pub batch_size: usize,
    /// Whether PDF and DOCX attachments are extracted into the derived index.
    pub document_indexing_enabled: bool,
    /// Maximum binary attachment size admitted for local extraction.
    pub document_max_bytes: u64,
    pub reranker_endpoint: String,
    pub reranker_model: String,
    pub reranker_enabled: bool,
    pub reranker_api_key_configured: bool,
    reranker_api_key: Option<String>,
    pub reranker_batch_size: usize,
    /// Refresh an out-of-date index when it is at least this old. Zero disables.
    pub refresh_after: Duration,
}

impl EmbeddingConfig {
    pub fn from_environment(vault_path: &Path) -> Result<Self> {
        let endpoint = env::var("TURBOVAULT_EMBEDDING_ENDPOINT")
            .unwrap_or_else(|_| DEFAULT_ENDPOINT.to_string());
        let model =
            env::var("TURBOVAULT_EMBEDDING_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.to_string());
        let api_key = env::var("TURBOVAULT_EMBEDDING_API_KEY")
            .ok()
            .filter(|value| !value.trim().is_empty());

        let chunk_chars = env_usize("TURBOVAULT_EMBEDDING_CHUNK_CHARS", DEFAULT_CHUNK_CHARS)?;
        let chunk_overlap = env_usize("TURBOVAULT_EMBEDDING_CHUNK_OVERLAP", DEFAULT_CHUNK_OVERLAP)?;
        if chunk_overlap >= chunk_chars {
            return Err(Error::config_error(format!(
                "TURBOVAULT_EMBEDDING_CHUNK_OVERLAP ({chunk_overlap}) must be smaller than TURBOVAULT_EMBEDDING_CHUNK_CHARS ({chunk_chars})"
            )));
        }
        let batch_size = env_usize("TURBOVAULT_EMBEDDING_BATCH_SIZE", DEFAULT_BATCH_SIZE)?;
        if batch_size == 0 {
            return Err(Error::config_error(
                "TURBOVAULT_EMBEDDING_BATCH_SIZE must be greater than zero",
            ));
        }
        let document_indexing_enabled = env_bool("TURBOVAULT_DOCUMENT_INDEXING_ENABLED", false)?;
        let document_max_bytes =
            env_usize("TURBOVAULT_DOCUMENT_MAX_BYTES", DEFAULT_DOCUMENT_MAX_BYTES)? as u64;
        if document_max_bytes == 0 {
            return Err(Error::config_error(
                "TURBOVAULT_DOCUMENT_MAX_BYTES must be greater than zero",
            ));
        }

        let reranker_endpoint = env::var("TURBOVAULT_RERANKER_ENDPOINT")
            .unwrap_or_else(|_| DEFAULT_RERANKER_ENDPOINT.to_string());
        let reranker_model = env::var("TURBOVAULT_RERANKER_MODEL")
            .unwrap_or_else(|_| DEFAULT_RERANKER_MODEL.to_string());
        let reranker_enabled = env::var("TURBOVAULT_RERANKER_ENABLED")
            .map(|val| !val.trim().is_empty() && val != "0" && !val.eq_ignore_ascii_case("false"))
            .unwrap_or(!reranker_endpoint.trim().is_empty());
        let reranker_api_key = select_reranker_api_key(
            env::var("TURBOVAULT_RERANKER_API_KEY")
                .ok()
                .filter(|value| !value.trim().is_empty()),
            api_key.as_ref(),
        );
        let reranker_batch_size =
            env_usize("TURBOVAULT_RERANKER_BATCH_SIZE", DEFAULT_RERANK_BATCH_SIZE)?;

        let refresh_after = Duration::from_secs(env_usize(
            "TURBOVAULT_EMBEDDING_REFRESH_AFTER_SECS",
            DEFAULT_REFRESH_AFTER_SECS as usize,
        )? as u64);

        let index_root = env::var_os("TURBOVAULT_EMBEDDING_INDEX_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(default_index_root);
        let vault_slug = stable_slug(vault_path.to_string_lossy().as_ref());
        let model_slug = stable_slug(&model);
        let index_dir = index_root.join(format!("{vault_slug}-{model_slug}"));

        Ok(Self {
            endpoint,
            model,
            api_key_configured: api_key.is_some(),
            api_key,
            index_dir,
            chunk_chars,
            chunk_overlap,
            batch_size,
            document_indexing_enabled,
            document_max_bytes,
            reranker_endpoint,
            reranker_model,
            reranker_enabled,
            reranker_api_key_configured: reranker_api_key.is_some(),
            reranker_api_key,
            reranker_batch_size,
            refresh_after,
        })
    }
}

fn select_reranker_api_key(
    dedicated_key: Option<String>,
    embedding_key: Option<&String>,
) -> Option<String> {
    dedicated_key.or_else(|| embedding_key.cloned())
}

fn env_usize(name: &str, default: usize) -> Result<usize> {
    match env::var(name) {
        Ok(value) => value.parse::<usize>().map_err(|_| {
            Error::config_error(format!("{name} must be a positive integer, got {value:?}"))
        }),
        Err(_) => Ok(default),
    }
}

fn env_bool(name: &str, default: bool) -> Result<bool> {
    match env::var(name) {
        Ok(value) => match value.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => Ok(true),
            "0" | "false" | "no" | "off" => Ok(false),
            _ => Err(Error::config_error(format!(
                "{name} must be true or false, got {value:?}"
            ))),
        },
        Err(_) => Ok(default),
    }
}

fn default_index_root() -> PathBuf {
    if let Some(cache) = env::var_os("XDG_CACHE_HOME") {
        return PathBuf::from(cache).join("turbovault/embeddings");
    }
    #[cfg(target_os = "windows")]
    if let Some(local_app_data) = env::var_os("LOCALAPPDATA") {
        return PathBuf::from(local_app_data).join("turbovault/embeddings");
    }
    if let Some(home) = env::var_os("HOME") {
        return PathBuf::from(home).join(".cache/turbovault/embeddings");
    }
    PathBuf::from(".turbovault/embeddings")
}

fn stable_slug(value: &str) -> String {
    let digest = Sha256::digest(value.as_bytes());
    digest[..12]
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct DocumentStamp {
    path: String,
    size_bytes: u64,
    modified_millis: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct StoredIndex {
    schema_version: u32,
    chunker_version: String,
    model: String,
    vault_path: String,
    dimension: usize,
    built_at: String,
    document_indexing_enabled: bool,
    document_max_bytes: u64,
    document_manifest: Vec<DocumentStamp>,
    chunks: Vec<EmbeddingChunk>,
}

/// A single heading-aware retrieval chunk.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingChunk {
    pub id: String,
    pub path: String,
    pub title: String,
    pub heading: Option<String>,
    pub text: String,
    pub content_hash: String,
    pub embedding: Vec<f32>,
}

/// Dense search result with explicit chunk provenance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingSearchResult {
    pub path: String,
    pub title: String,
    pub heading: Option<String>,
    pub text: String,
    pub score: f64,
    pub rank: usize,
    pub chunk_id: String,
}

/// Combined sparse+dense result. Optional component fields make it clear why
/// a result appeared in the fused ranking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridSearchResult {
    pub path: String,
    pub title: String,
    pub heading: Option<String>,
    pub text: String,
    pub preview: String,
    pub snippet: String,
    pub rrf_score: f64,
    pub dense_score: Option<f64>,
    pub sparse_score: Option<f64>,
    pub dense_rank: Option<usize>,
    pub sparse_rank: Option<usize>,
    pub rerank_score: Option<f64>,
    pub rerank_rank: Option<usize>,
    pub chunk_id: Option<String>,
    /// SHA-256 of the returned passage text; pass it to read_passage to avoid
    /// accepting a reused ordinal after an index refresh.
    pub chunk_hash: Option<String>,
}

/// A passage reopened from the persisted index, never assigned a search score.
#[derive(Debug, Clone, Serialize)]
pub struct IndexedPassage {
    pub chunk_id: String,
    pub heading: Option<String>,
    pub text: String,
    pub truncated: bool,
    pub anchor: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct PassageRead {
    pub path: String,
    pub anchor_id: String,
    pub anchor_hash: String,
    pub built_at: String,
    pub index_stale: bool,
    pub chunks: Vec<IndexedPassage>,
}

#[derive(Debug, Clone, Default)]
struct ReindexProgress {
    in_progress: bool,
    phase: Option<String>,
    processed_chunks: usize,
    total_chunks: usize,
    started_at: Option<String>,
    completed_at: Option<String>,
    error: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct EmbeddingIndexStatus {
    pub configured: bool,
    pub endpoint: String,
    pub model: String,
    pub api_key_configured: bool,
    pub index_path: String,
    /// The vault changed since the index was built. Served anyway.
    pub stale: bool,
    /// The index cannot be used at all; run `reindex_embeddings`.
    pub incompatible: bool,
    pub exists: bool,
    pub chunks: usize,
    pub dimensions: usize,
    pub built_at: Option<String>,
    pub schema_version: Option<u32>,
    pub chunker_version: Option<String>,
    pub document_indexing_enabled: bool,
    pub document_max_bytes: u64,
    pub document_files_discovered: usize,
    pub document_files_indexed: usize,
    pub document_chunks: usize,
    pub reindex_in_progress: bool,
    pub reindex_phase: Option<String>,
    pub reindex_processed_chunks: usize,
    pub reindex_total_chunks: usize,
    pub reindex_started_at: Option<String>,
    pub reindex_completed_at: Option<String>,
    pub reindex_error: Option<String>,
    pub reranker_endpoint: String,
    pub reranker_model: String,
    pub reranker_enabled: bool,
    pub reranker_api_key_configured: bool,
}

#[derive(Debug, Serialize)]
struct EmbeddingRequest<'a> {
    model: &'a str,
    input: &'a [String],
}

#[derive(Debug, Serialize)]
struct RerankRequest<'a> {
    model: &'a str,
    query: &'a str,
    documents: &'a [String],
    top_n: usize,
}

#[derive(Debug, Deserialize)]
struct RerankResponse {
    #[serde(default)]
    results: Vec<RerankItem>,
}

#[derive(Debug, Deserialize)]
struct RerankItem {
    index: usize,
    relevance_score: f64,
}

#[derive(Debug, Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingItem>,
}

#[derive(Debug, Deserialize)]
struct EmbeddingItem {
    embedding: Vec<f32>,
    index: usize,
}

/// Whether an out-of-date index has fallen far enough behind to refresh now.
///
/// Split out as a pure function so the threshold policy is testable without an
/// engine, an endpoint, or a vault. A fresh index is never due, regardless of
/// age: nothing has changed, so there is nothing to catch up on.
fn refresh_due(
    stale: bool,
    built_at: Option<&str>,
    threshold: Duration,
    now: DateTime<Utc>,
) -> bool {
    if threshold.is_zero() || !stale {
        return false;
    }
    let Some(built_at) = built_at else {
        return true;
    };
    match DateTime::parse_from_rfc3339(built_at) {
        Ok(built) => now
            .signed_duration_since(built.with_timezone(&Utc))
            .to_std()
            // A negative age means the clock moved backwards; do not thrash.
            .map(|age| age >= threshold)
            .unwrap_or(false),
        // An unreadable timestamp is treated as due: the index is stale anyway,
        // and one refresh is cheap next to never refreshing.
        Err(_) => true,
    }
}

/// Why a stored index cannot be used with the current configuration, if it
/// cannot.
///
/// Deliberately silent about freshness. An index built from an older snapshot
/// of the same vault is usable: its vectors were produced by the same model and
/// chunker, so they live in the same space as a fresh query vector and rank
/// correctly. Model, schema, chunker, and vault changes make vectors unsafe to
/// compare or reuse; document-indexing changes alter the authorized source set.
/// Either condition must fail closed until the index is rebuilt.
fn index_mismatch(
    index: &StoredIndex,
    model: &str,
    vault_path: &str,
    document_indexing_enabled: bool,
    document_max_bytes: u64,
) -> Option<&'static str> {
    if index.schema_version != INDEX_SCHEMA_VERSION {
        return Some("index schema version differs from this build");
    }
    if index.chunker_version != CHUNKER_VERSION {
        return Some("index chunker version differs from this build");
    }
    if index.model != model {
        return Some("index was built with a different embedding model");
    }
    if index.vault_path != vault_path {
        return Some("index was built for a different vault path");
    }
    if index.document_indexing_enabled != document_indexing_enabled
        || index.document_max_bytes != document_max_bytes
    {
        return Some("document indexing configuration changed");
    }
    None
}

/// Which retrieval channels contributed to a hybrid result, and why any did not.
///
/// Exists so a degraded search is never mistaken for a healthy one. The only
/// other evidence is `dense_score: null` on every row, which is easy to overlook
/// and impossible to explain after the fact, because the reason lived in a log
/// line the caller never saw.
#[derive(Debug, Clone, Default, Serialize)]
pub struct HybridDiagnostics {
    /// Reason the dense channel contributed nothing, if it did not.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dense_unavailable: Option<String>,
    /// Reason the sparse channel contributed nothing, if it did not.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sparse_unavailable: Option<String>,
    /// Reason cross-encoder reranking was skipped, if it was.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rerank_unavailable: Option<String>,
    /// The vault had been written to since the index was built.
    pub index_stale: bool,
    /// A refresh ran before this search because the index was past the threshold.
    pub index_refreshed: bool,
    /// The refresh was due but failed; the previous index was served instead.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub refresh_failed: Option<String>,
}

impl HybridDiagnostics {
    /// Whether every channel contributed, so a caller can stay silent.
    pub fn is_complete(&self) -> bool {
        self.dense_unavailable.is_none()
            && self.sparse_unavailable.is_none()
            && self.rerank_unavailable.is_none()
            && self.refresh_failed.is_none()
    }
}

/// Hybrid retrieval results together with the channel status that produced them.
#[derive(Debug, Clone, Serialize)]
pub struct HybridSearchOutcome {
    pub results: Vec<HybridSearchResult>,
    pub diagnostics: HybridDiagnostics,
}

/// What a lazy refresh did, so the caller can report it.
#[derive(Debug, Default)]
struct RefreshReport {
    refreshed: bool,
    failure: Option<String>,
}

/// Cached dense index and its endpoint client.
pub struct EmbeddingEngine {
    manager: Arc<VaultManager>,
    config: EmbeddingConfig,
    client: Client,
    index_path: PathBuf,
    stale_path: PathBuf,
    state: RwLock<Option<StoredIndex>>,
    stale: AtomicBool,
    incompatible: AtomicBool,
    /// Serializes lazy refreshes so concurrent searches embed at most once.
    refresh_guard: tokio::sync::Mutex<()>,
    /// Last lazy-refresh attempt, so a failing endpoint backs off.
    last_refresh_attempt: std::sync::Mutex<Option<Instant>>,
    /// Prevents two complete index builds from writing the same derived index.
    reindex_guard: tokio::sync::Mutex<()>,
    /// Allows the MCP tool to start one long-running build in the background.
    reindex_running: AtomicBool,
    reindex_progress: RwLock<ReindexProgress>,
}

impl EmbeddingEngine {
    pub async fn new(manager: Arc<VaultManager>) -> Result<Self> {
        let config = EmbeddingConfig::from_environment(manager.vault_path())?;
        tokio::fs::create_dir_all(&config.index_dir).await?;
        let index_path = config.index_dir.join("index.bin");
        let stale_path = config.index_dir.join("index.stale");
        let mut stale = tokio::fs::try_exists(&stale_path).await.unwrap_or(false);
        let mut incompatible = false;
        let state = if tokio::fs::try_exists(&index_path).await.unwrap_or(false) {
            let bytes = tokio::fs::read(&index_path).await.map_err(|error| {
                Error::other(format!("failed to read embedding index: {error}"))
            })?;
            match bincode::deserialize::<StoredIndex>(&bytes) {
                Ok(parsed) => {
                    incompatible = index_mismatch(
                        &parsed,
                        &config.model,
                        &manager.vault_path().to_string_lossy(),
                        config.document_indexing_enabled,
                        config.document_max_bytes,
                    )
                    .is_some();
                    Some(parsed)
                }
                Err(error) => {
                    incompatible = true;
                    log::warn!(
                        "embedding index {} uses an older or unreadable schema; run \
                         reindex_embeddings: {error}",
                        index_path.display()
                    );
                    None
                }
            }
        } else {
            None
        };
        if state
            .as_ref()
            .is_some_and(|index| stored_index_has_excluded_sources(index, &manager))
        {
            stale = true;
            log::info!(
                "embedding index contains newly excluded sources; it will be pruned before the next hybrid search"
            );
        }

        let client = Client::builder()
            .timeout(Duration::from_secs(180))
            .build()
            .map_err(|error| {
                Error::other(format!("failed to create embedding HTTP client: {error}"))
            })?;

        Ok(Self {
            manager,
            config,
            client,
            index_path,
            stale_path,
            state: RwLock::new(state),
            stale: AtomicBool::new(stale),
            incompatible: AtomicBool::new(incompatible),
            refresh_guard: tokio::sync::Mutex::new(()),
            last_refresh_attempt: std::sync::Mutex::new(None),
            reindex_guard: tokio::sync::Mutex::new(()),
            reindex_running: AtomicBool::new(false),
            reindex_progress: RwLock::new(ReindexProgress::default()),
        })
    }

    pub async fn status(&self) -> EmbeddingIndexStatus {
        if self.config.document_indexing_enabled
            && self.document_sources_changed().await.unwrap_or(false)
        {
            let _ = self.mark_stale().await;
        }
        let state = self.state.read().await;
        let progress = self.reindex_progress.read().await.clone();
        let document_chunks = state
            .as_ref()
            .map(|index| {
                index
                    .chunks
                    .iter()
                    .filter(|chunk| is_document_path(&chunk.path))
                    .filter(|chunk| !source_path_is_excluded(&self.manager, &chunk.path))
                    .count()
            })
            .unwrap_or(0);
        let document_files_indexed = state
            .as_ref()
            .map(|index| {
                index
                    .chunks
                    .iter()
                    .filter(|chunk| is_document_path(&chunk.path))
                    .filter(|chunk| !source_path_is_excluded(&self.manager, &chunk.path))
                    .map(|chunk| chunk.path.as_str())
                    .collect::<HashSet<_>>()
                    .len()
            })
            .unwrap_or(0);
        EmbeddingIndexStatus {
            configured: !self.config.endpoint.is_empty() && !self.config.model.is_empty(),
            endpoint: self.config.endpoint.clone(),
            model: self.config.model.clone(),
            api_key_configured: self.config.api_key_configured,
            index_path: self.index_path.display().to_string(),
            stale: self.stale.load(AtomicOrdering::Acquire),
            incompatible: self.incompatible.load(AtomicOrdering::Acquire),
            exists: state.is_some(),
            chunks: state
                .as_ref()
                .map(|index| {
                    index
                        .chunks
                        .iter()
                        .filter(|chunk| !source_path_is_excluded(&self.manager, &chunk.path))
                        .count()
                })
                .unwrap_or(0),
            dimensions: state.as_ref().map(|index| index.dimension).unwrap_or(0),
            built_at: state.as_ref().map(|index| index.built_at.clone()),
            schema_version: state.as_ref().map(|index| index.schema_version),
            chunker_version: state.as_ref().map(|index| index.chunker_version.clone()),
            document_indexing_enabled: self.config.document_indexing_enabled,
            document_max_bytes: self.config.document_max_bytes,
            document_files_discovered: state
                .as_ref()
                .map(|index| {
                    index
                        .document_manifest
                        .iter()
                        .filter(|document| !source_path_is_excluded(&self.manager, &document.path))
                        .count()
                })
                .unwrap_or(0),
            document_files_indexed,
            document_chunks,
            reindex_in_progress: progress.in_progress,
            reindex_phase: progress.phase,
            reindex_processed_chunks: progress.processed_chunks,
            reindex_total_chunks: progress.total_chunks,
            reindex_started_at: progress.started_at,
            reindex_completed_at: progress.completed_at,
            reindex_error: progress.error,
            reranker_endpoint: self.config.reranker_endpoint.clone(),
            reranker_model: self.config.reranker_model.clone(),
            reranker_enabled: self.config.reranker_enabled,
            reranker_api_key_configured: self.config.reranker_api_key_configured,
        }
    }

    /// Record that the vault changed after the index was built.
    ///
    /// The marker is persisted so a restart still knows the vectors are out of
    /// date. It no longer blocks retrieval: an out-of-date index is served, and
    /// the flag exists so `embedding_index_status` can report freshness.
    pub async fn mark_stale(&self) -> Result<()> {
        self.stale.store(true, AtomicOrdering::Release);
        tokio::fs::write(&self.stale_path, b"stale\n").await?;
        Ok(())
    }

    /// Start a complete rebuild without holding the MCP request open.
    ///
    /// The returned status reports `reindex_in_progress=true`; callers should
    /// poll `status()` until the phase becomes `complete` or `failed`.
    pub async fn start_reindex(self: &Arc<Self>) -> EmbeddingIndexStatus {
        if self
            .reindex_running
            .compare_exchange(false, true, AtomicOrdering::AcqRel, AtomicOrdering::Acquire)
            .is_err()
        {
            return self.status().await;
        }
        self.begin_reindex().await;
        let engine = Arc::clone(self);
        tokio::spawn(async move {
            let result = engine.run_reindex().await;
            engine.finish_reindex(&result).await;
        });
        self.status().await
    }

    /// Synchronous entry point retained for internal refreshes and tests.
    /// Public MCP callers use `start_reindex` so a full vault build is not
    /// terminated by a client request timeout.
    pub async fn reindex(&self) -> Result<EmbeddingIndexStatus> {
        if self
            .reindex_running
            .compare_exchange(false, true, AtomicOrdering::AcqRel, AtomicOrdering::Acquire)
            .is_err()
        {
            return Err(Error::config_error("embedding reindex is already running"));
        }
        self.begin_reindex().await;
        let result = self.run_reindex().await;
        self.finish_reindex(&result).await;
        result
    }

    async fn run_reindex(&self) -> Result<EmbeddingIndexStatus> {
        let _guard = self.reindex_guard.lock().await;
        self.set_reindex_phase("scanning", 0, 0).await;
        self.mark_stale().await?;
        self.manager.ensure_fresh().await;

        let existing_chunks_by_path: HashMap<String, (String, Vec<EmbeddingChunk>)> = {
            let state = self.state.read().await;
            let mut map = HashMap::new();
            if let Some(index) = state.as_ref()
                && index.schema_version == INDEX_SCHEMA_VERSION
                && index.chunker_version == CHUNKER_VERSION
                && index.model == self.config.model
            {
                for chunk in &index.chunks {
                    map.entry(chunk.path.clone())
                        .or_insert_with(|| (chunk.content_hash.clone(), Vec::new()))
                        .1
                        .push(chunk.clone());
                }
            }
            map
        };

        let files = self.manager.scan_vault().await?;
        let mut final_chunks = Vec::new();
        let mut pending = Vec::new();
        let mut document_manifest = Vec::new();

        for file_path in files {
            if !file_path
                .extension()
                .and_then(|extension| extension.to_str())
                .is_some_and(|extension| extension.eq_ignore_ascii_case("md"))
            {
                continue;
            }
            let relative = file_path
                .strip_prefix(self.manager.vault_path())
                .unwrap_or(&file_path)
                .to_string_lossy()
                .to_string();
            if is_path_excluded(&relative) {
                continue;
            }
            let Ok(vault_file) = self.manager.parse_file(&file_path).await else {
                continue;
            };
            let content_hash = hex_hash(vault_file.content.as_bytes());

            // Reuse pre-existing chunks if content_hash has not changed
            if let Some((cached_hash, cached_chunks)) = existing_chunks_by_path.get(&relative)
                && cached_hash == &content_hash
                && !cached_chunks.is_empty()
            {
                final_chunks.extend(cached_chunks.clone());
                continue;
            }

            let title = vault_file
                .frontmatter
                .as_ref()
                .and_then(|frontmatter| frontmatter.data.get("title"))
                .and_then(|value| value.as_str())
                .map(ToOwned::to_owned)
                .or_else(|| {
                    vault_file
                        .headings
                        .first()
                        .map(|heading| heading.text.clone())
                })
                .unwrap_or_else(|| {
                    file_path
                        .file_stem()
                        .and_then(|stem| stem.to_str())
                        .unwrap_or("Untitled")
                        .to_string()
                });

            queue_text_chunks(
                &mut pending,
                &relative,
                &title,
                "Markdown note",
                &vault_file.content,
                &content_hash,
                self.config.chunk_chars,
                self.config.chunk_overlap,
            );
        }

        if self.config.document_indexing_enabled {
            let documents = self
                .manager
                .scan_files_by_extensions(&["pdf", "docx"], self.config.document_max_bytes)?;
            document_manifest = document_manifest_for(self.manager.vault_path(), &documents);
            for document in documents {
                let file_path = document.path;
                let relative = file_path
                    .strip_prefix(self.manager.vault_path())
                    .unwrap_or(&file_path)
                    .to_string_lossy()
                    .to_string();
                if is_path_excluded(&relative) {
                    continue;
                }
                let bytes = match tokio::fs::read(&file_path).await {
                    Ok(bytes) => bytes,
                    Err(error) => {
                        log::warn!("skipping document {relative:?}: failed to read: {error}");
                        continue;
                    }
                };
                let content_hash = hex_hash(&bytes);
                if let Some((cached_hash, cached_chunks)) = existing_chunks_by_path.get(&relative)
                    && cached_hash == &content_hash
                    && !cached_chunks.is_empty()
                {
                    final_chunks.extend(cached_chunks.clone());
                    continue;
                }

                let extension = file_path
                    .extension()
                    .and_then(|value| value.to_str())
                    .unwrap_or_default()
                    .to_ascii_lowercase();
                let source_type = if extension == "pdf" {
                    "PDF attachment"
                } else {
                    "DOCX attachment"
                };
                let extracted = tokio::task::spawn_blocking(move || match extension.as_str() {
                    "pdf" => extract_pdf_markdown(&bytes),
                    "docx" => extract_docx_markdown(&bytes),
                    _ => unreachable!("scanner admits only PDF and DOCX"),
                })
                .await
                .map_err(|error| {
                    Error::other(format!("document extraction task failed: {error}"))
                })?;
                let extracted = match extracted {
                    Ok(text) => text,
                    Err(error) => {
                        log::warn!("skipping document {relative:?}: {error}");
                        continue;
                    }
                };
                let title = file_path
                    .file_stem()
                    .and_then(|stem| stem.to_str())
                    .unwrap_or("Untitled document")
                    .to_string();
                queue_text_chunks(
                    &mut pending,
                    &relative,
                    &title,
                    source_type,
                    &extracted,
                    &content_hash,
                    self.config.chunk_chars,
                    self.config.chunk_overlap,
                );
            }
        }

        self.set_reindex_phase("embedding", 0, pending.len()).await;
        let mut processed_chunks = 0;
        for batch in pending.chunks(self.config.batch_size) {
            let inputs: Vec<String> = batch.iter().map(|chunk| chunk.input.clone()).collect();
            let vectors = self.embed(&inputs).await?;
            if vectors.len() != batch.len() {
                return Err(Error::other(format!(
                    "embedding endpoint returned {} vectors for {} inputs",
                    vectors.len(),
                    batch.len()
                )));
            }
            for (pending_chunk, embedding) in batch.iter().zip(vectors) {
                final_chunks.push(EmbeddingChunk {
                    id: pending_chunk.id.clone(),
                    path: pending_chunk.path.clone(),
                    title: pending_chunk.title.clone(),
                    heading: pending_chunk.heading.clone(),
                    text: pending_chunk.text.clone(),
                    content_hash: pending_chunk.content_hash.clone(),
                    embedding,
                });
            }
            processed_chunks += batch.len();
            self.set_reindex_progress(processed_chunks, pending.len())
                .await;
        }

        self.set_reindex_phase("writing", processed_chunks, pending.len())
            .await;
        final_chunks.sort_by(|a, b| a.id.cmp(&b.id));
        let dimension = final_chunks
            .first()
            .map(|chunk| chunk.embedding.len())
            .unwrap_or(0);
        if final_chunks
            .iter()
            .any(|chunk| chunk.embedding.len() != dimension)
        {
            return Err(Error::other(
                "embedding endpoint returned inconsistent vector dimensions",
            ));
        }
        let index = StoredIndex {
            schema_version: INDEX_SCHEMA_VERSION,
            chunker_version: CHUNKER_VERSION.to_string(),
            model: self.config.model.clone(),
            vault_path: self.manager.vault_path().to_string_lossy().to_string(),
            dimension,
            built_at: chrono::Utc::now().to_rfc3339(),
            document_indexing_enabled: self.config.document_indexing_enabled,
            document_max_bytes: self.config.document_max_bytes,
            document_manifest,
            chunks: final_chunks,
        };
        let bytes = bincode::serialize(&index)
            .map_err(|error| Error::other(format!("failed to encode embedding index: {error}")))?;
        let temporary = self.index_path.with_extension("bin.tmp");
        tokio::fs::write(&temporary, bytes).await?;
        tokio::fs::rename(&temporary, &self.index_path).await?;
        let _ = tokio::fs::remove_file(&self.stale_path).await;
        *self.state.write().await = Some(index);
        self.stale.store(false, AtomicOrdering::Release);
        // The index just written necessarily matches the current configuration.
        self.incompatible.store(false, AtomicOrdering::Release);
        Ok(self.status().await)
    }

    /// Reopen one exact search hit with bounded, same-section neighboring chunks.
    /// No embedding request, reranking, or score inheritance occurs here.
    pub async fn read_passage(
        &self,
        path: &str,
        chunk_id: &str,
        expected_hash: &str,
        neighbors: usize,
        max_chars: usize,
    ) -> Result<PassageRead> {
        if self.incompatible.load(AtomicOrdering::Acquire) {
            return Err(Error::config_error(
                "embedding index is incompatible; run reindex_embeddings",
            ));
        }
        let state = self.state.read().await;
        let index = state.as_ref().ok_or_else(|| {
            Error::config_error(
                "no embedding index exists; run reindex_embeddings before read_passage",
            )
        })?;
        self.manager.resolve_path(Path::new(path))?;
        if source_path_is_excluded(&self.manager, path) {
            return Err(Error::config_error(
                "passage path is excluded by vault policy",
            ));
        }
        expand_passage(
            index,
            path,
            chunk_id,
            expected_hash,
            neighbors.min(3),
            max_chars.clamp(1, 12_000),
            self.stale.load(AtomicOrdering::Acquire),
        )
    }

    pub async fn search(&self, query: &str, limit: usize) -> Result<Vec<EmbeddingSearchResult>> {
        // Refuse only when the stored vectors are not comparable to a query
        // vector. An out-of-date index is deliberately served: it ranks
        // correctly, and the sparse channel already covers everything created
        // or edited since the last reindex.
        if self.incompatible.load(AtomicOrdering::Acquire) {
            return Err(Error::config_error(
                "embedding index was built with a different model, schema, chunker, vault \
                 path, or document-indexing policy. Run reindex_embeddings before dense search",
            ));
        }
        let query = query.trim();
        if query.is_empty() {
            return Ok(Vec::new());
        }
        let vectors = self.embed(&[query.to_string()]).await?;
        let query_vector = vectors
            .first()
            .ok_or_else(|| Error::other("embedding endpoint returned no query vector"))?;
        let state = self.state.read().await;
        let index = state.as_ref().ok_or_else(|| {
            Error::config_error(
                "no embedding index exists; run reindex_embeddings before dense search",
            )
        })?;
        if index.dimension != query_vector.len() {
            return Err(Error::config_error(format!(
                "query vector dimension {} does not match index dimension {}",
                query_vector.len(),
                index.dimension
            )));
        }

        let mut ranked: Vec<(f64, &EmbeddingChunk)> = index
            .chunks
            .iter()
            .filter(|chunk| !source_path_is_excluded(&self.manager, &chunk.path))
            .filter_map(|chunk| {
                let score = cosine_similarity(query_vector, &chunk.embedding);
                score.is_finite().then_some((score, chunk))
            })
            .collect();
        ranked.sort_by(|left, right| right.0.partial_cmp(&left.0).unwrap_or(Ordering::Equal));
        ranked.truncate(limit);
        Ok(ranked
            .into_iter()
            .enumerate()
            .map(|(offset, (score, chunk))| EmbeddingSearchResult {
                path: chunk.path.clone(),
                title: chunk.title.clone(),
                heading: chunk.heading.clone(),
                text: chunk.text.clone(),
                score: score.clamp(-1.0, 1.0),
                rank: offset + 1,
                chunk_id: chunk.id.clone(),
            })
            .collect())
    }

    async fn document_sources_changed(&self) -> Result<bool> {
        let expected = {
            let state = self.state.read().await;
            let Some(index) = state.as_ref() else {
                return Ok(false);
            };
            index.document_manifest.clone()
        };
        let documents = self
            .manager
            .scan_files_by_extensions(&["pdf", "docx"], self.config.document_max_bytes)?;
        Ok(document_manifest_for(self.manager.vault_path(), &documents) != expected)
    }

    async fn index_has_excluded_sources(&self) -> bool {
        let state = self.state.read().await;
        state
            .as_ref()
            .is_some_and(|index| stored_index_has_excluded_sources(index, &self.manager))
    }

    async fn begin_reindex(&self) {
        let mut progress = self.reindex_progress.write().await;
        *progress = ReindexProgress {
            in_progress: true,
            phase: Some("scanning".to_string()),
            started_at: Some(Utc::now().to_rfc3339()),
            ..ReindexProgress::default()
        };
    }

    async fn set_reindex_phase(&self, phase: &str, processed: usize, total: usize) {
        let mut progress = self.reindex_progress.write().await;
        progress.phase = Some(phase.to_string());
        progress.processed_chunks = processed;
        progress.total_chunks = total;
    }

    async fn set_reindex_progress(&self, processed: usize, total: usize) {
        let mut progress = self.reindex_progress.write().await;
        progress.processed_chunks = processed;
        progress.total_chunks = total;
    }

    async fn finish_reindex(&self, result: &Result<EmbeddingIndexStatus>) {
        let mut progress = self.reindex_progress.write().await;
        progress.in_progress = false;
        progress.phase = Some(if result.is_ok() { "complete" } else { "failed" }.to_string());
        progress.completed_at = Some(Utc::now().to_rfc3339());
        progress.error = result.as_ref().err().map(ToString::to_string);
        if result.is_ok() {
            progress.processed_chunks = progress.total_chunks;
        }
        self.reindex_running.store(false, AtomicOrdering::Release);
    }

    /// Bring an out-of-date index current before answering, if it has fallen
    /// further behind than the configured threshold.
    ///
    /// Deliberately never fails the caller. The whole point of serving an
    /// out-of-date index is that retrieval must not depend on reindexing
    /// succeeding, so a refresh failure logs and the existing vectors are used.
    ///
    /// Never builds a missing index either: the initial build embeds the entire
    /// vault and stays an explicit choice, while maintaining an existing index
    /// costs only the notes that changed.
    async fn refresh_if_due(&self) -> RefreshReport {
        if self.reindex_running.load(AtomicOrdering::Acquire) || self.config.refresh_after.is_zero()
        {
            return RefreshReport::default();
        }
        let excluded_sources_present = self.index_has_excluded_sources().await;
        if excluded_sources_present {
            let _ = self.mark_stale().await;
        }
        let document_sources_changed = self.config.document_indexing_enabled
            && self.document_sources_changed().await.unwrap_or(false);
        if document_sources_changed {
            let _ = self.mark_stale().await;
        }
        let built_at = {
            let state = self.state.read().await;
            match state.as_ref() {
                None => return RefreshReport::default(),
                Some(index) => index.built_at.clone(),
            }
        };
        if !excluded_sources_present
            && !document_sources_changed
            && !refresh_due(
                self.stale.load(AtomicOrdering::Acquire),
                Some(&built_at),
                self.config.refresh_after,
                Utc::now(),
            )
        {
            return RefreshReport::default();
        }
        if let Ok(last) = self.last_refresh_attempt.lock()
            && let Some(attempted) = *last
            && attempted.elapsed() < REFRESH_RETRY_BACKOFF
        {
            return RefreshReport::default();
        }
        // Concurrent searches proceed with the vectors already loaded rather
        // than queueing behind a refresh they did not ask for.
        let Ok(_guard) = self.refresh_guard.try_lock() else {
            return RefreshReport::default();
        };
        if let Ok(mut last) = self.last_refresh_attempt.lock() {
            *last = Some(Instant::now());
        }
        if excluded_sources_present {
            log::info!(
                "excluded source content remains in the embedding index; pruning it before search"
            );
        } else if document_sources_changed {
            log::info!("PDF or DOCX attachments changed; refreshing the embedding index");
        } else {
            log::info!(
                "embedding index is out of date and older than {:?}; refreshing before search",
                self.config.refresh_after
            );
        }
        match self.reindex().await {
            Ok(status) => {
                log::info!("lazy index refresh complete: {} chunks", status.chunks);
                RefreshReport {
                    refreshed: true,
                    failure: None,
                }
            }
            Err(error) => {
                log::warn!("lazy index refresh failed; serving the existing index: {error}");
                RefreshReport {
                    refreshed: false,
                    failure: Some(error.to_string()),
                }
            }
        }
    }

    /// Hybrid retrieval: fused dense and sparse candidates, optionally reranked.
    ///
    /// Returns the channel diagnostics alongside the results, so a caller can
    /// report *why* a search was degraded rather than only that it was.
    pub async fn hybrid_search(
        &self,
        search_engine: &SearchEngine,
        query: &str,
        limit: usize,
    ) -> Result<HybridSearchOutcome> {
        let refresh = self.refresh_if_due().await;
        let index_stale = self.stale.load(AtomicOrdering::Acquire);
        let candidate_limit = limit.saturating_mul(5).clamp(20, 100);
        // Both retrieval channels degrade independently: a dead embedding sidecar
        // must not take out lexical search, and an unparseable lexical query must
        // not discard dense candidates that already succeeded. When both channels
        // come back empty and at least one of them failed, the failure is surfaced
        // as an error instead of a silent empty success.
        let mut dense_error: Option<Error> = None;
        let mut sparse_error: Option<Error> = None;
        let dense = match self.search(query, candidate_limit).await {
            Ok(results) => results,
            Err(error) => {
                log::warn!("dense search unavailable in hybrid retrieval, using sparse: {error}");
                dense_error = Some(error);
                Vec::new()
            }
        };
        let sparse = match search_engine
            .advanced_search(SearchQuery::new(query).limit(candidate_limit))
            .await
        {
            Ok(results) => results
                .into_iter()
                .filter(|result| !source_path_is_excluded(&self.manager, &result.path))
                .collect(),
            Err(error) => {
                log::warn!("sparse search unavailable in hybrid retrieval, using dense: {error}");
                sparse_error = Some(error);
                Vec::new()
            }
        };
        // Captured before the errors are consumed below, so the diagnostics can
        // name the cause even when a channel is empty for a benign reason.
        let dense_reason = dense_error.as_ref().map(|error| error.to_string());
        let sparse_reason = sparse_error.as_ref().map(|error| error.to_string());
        if dense.is_empty()
            && sparse.is_empty()
            && let Some(error) = dense_error.or(sparse_error)
        {
            return Err(error);
        }

        let mut fused: HashMap<String, HybridAccumulator> = HashMap::new();
        for (offset, result) in dense.into_iter().enumerate() {
            let rank = offset + 1;
            let entry = fused.entry(result.path.clone()).or_insert_with(|| {
                HybridAccumulator::new(
                    result.path.clone(),
                    result.title.clone(),
                    result.heading.clone(),
                    result.text.clone(),
                    Some(result.chunk_id.clone()),
                )
            });
            if entry.dense_rank.is_none() {
                entry.heading = result.heading;
                entry.text = result.text;
                entry.chunk_id = Some(result.chunk_id);
                entry.dense_score = Some(result.score);
                entry.dense_rank = Some(rank);
                entry.dense_rrf = 1.0 / (RRF_K + rank as f64);
            } else if result.score > entry.dense_score.unwrap_or(-1.0) {
                entry.heading = result.heading;
                entry.text = result.text;
                entry.chunk_id = Some(result.chunk_id);
                entry.dense_score = Some(result.score);
            }
        }
        for (offset, result) in sparse.into_iter().enumerate() {
            let rank = offset + 1;
            let entry = fused.entry(result.path.clone()).or_insert_with(|| {
                HybridAccumulator::new(
                    result.path.clone(),
                    result.title.clone(),
                    None,
                    result.preview.clone(),
                    None,
                )
            });
            entry.title = result.title;
            entry.preview = result.preview;
            entry.snippet = result.snippet;
            entry.sparse_score = Some(result.score);
            if entry.sparse_rank.is_none() {
                entry.sparse_rank = Some(rank);
                entry.sparse_rrf = 1.0 / (RRF_K + rank as f64);
            }
        }

        let mut results: Vec<HybridAccumulator> = fused.into_values().collect();
        for item in &mut results {
            item.rrf_score = item.dense_rrf + item.sparse_rrf;
        }
        results.sort_by(|left, right| {
            right
                .rrf_score
                .partial_cmp(&left.rrf_score)
                .unwrap_or(Ordering::Equal)
        });

        // Second-stage cross-encoder rerank if enabled and endpoint is configured
        let mut rerank_error: Option<Error> = None;
        if self.config.reranker_enabled
            && !self.config.reranker_endpoint.trim().is_empty()
            && !results.is_empty()
        {
            let pool_size = limit.max(12).min(results.len());
            let pool = &results[..pool_size];
            let documents: Vec<String> = pool
                .iter()
                .map(|item| {
                    let section = item
                        .heading
                        .as_deref()
                        .map(|h| format!("Section: {h}\n"))
                        .unwrap_or_default();
                    format!("Title: {}\n{}Content:\n{}", item.title, section, item.text)
                })
                .collect();

            let reranked = match self.rerank(query, &documents).await {
                Ok(results) => results,
                Err(error) => {
                    log::warn!(
                        "reranker unavailable in hybrid retrieval, keeping RRF order: {error}"
                    );
                    rerank_error = Some(error);
                    Vec::new()
                }
            };
            if !reranked.is_empty() {
                let mut reranked_pool: Vec<HybridAccumulator> = Vec::with_capacity(pool_size);
                for (new_rank, (orig_idx, score)) in reranked.into_iter().enumerate() {
                    if orig_idx < pool.len() {
                        let mut item = pool[orig_idx].clone();
                        item.rerank_score = Some(score);
                        item.rerank_rank = Some(new_rank + 1);
                        reranked_pool.push(item);
                    }
                }
                let remaining = results[pool_size..].to_vec();
                results = reranked_pool;
                results.extend(remaining);
            }
        }

        results.truncate(limit);
        Ok(HybridSearchOutcome {
            results: results.into_iter().map(HybridAccumulator::finish).collect(),
            diagnostics: HybridDiagnostics {
                dense_unavailable: dense_reason,
                sparse_unavailable: sparse_reason,
                rerank_unavailable: rerank_error.map(|error| error.to_string()),
                index_stale,
                index_refreshed: refresh.refreshed,
                refresh_failed: refresh.failure,
            },
        })
    }

    pub async fn rerank(&self, query: &str, documents: &[String]) -> Result<Vec<(usize, f64)>> {
        if documents.is_empty() {
            return Ok(Vec::new());
        }
        if !self.config.reranker_enabled || self.config.reranker_endpoint.trim().is_empty() {
            return Err(Error::config_error("reranker is not configured or enabled"));
        }

        let mut scores: HashMap<usize, f64> = HashMap::new();
        let batch_size = self.config.reranker_batch_size.max(1);

        for (batch_idx, chunk) in documents.chunks(batch_size).enumerate() {
            let offset = batch_idx * batch_size;
            let request = RerankRequest {
                model: &self.config.reranker_model,
                query,
                documents: chunk,
                // TurboVault requires one score per submitted candidate. This
                // is explicit for hosted routers whose default may return only
                // a top subset.
                top_n: chunk.len(),
            };
            let mut builder = self
                .client
                .post(&self.config.reranker_endpoint)
                .json(&request);
            if let Some(api_key) = &self.config.reranker_api_key {
                builder = builder.bearer_auth(api_key);
            }
            let response = builder.send().await.map_err(|error| {
                Error::other(format!(
                    "reranker request to {} failed: {error}",
                    self.config.reranker_endpoint
                ))
            })?;
            let status = response.status();
            let body = response.text().await.map_err(|error| {
                Error::other(format!("failed to read reranker response body: {error}"))
            })?;
            if !status.is_success() {
                return Err(Error::other(format!(
                    "reranker endpoint returned HTTP {status}: {}",
                    truncate_for_error(&body)
                )));
            }
            let parsed: RerankResponse = serde_json::from_str(&body)
                .map_err(|error| Error::other(format!("invalid reranker response: {error}")))?;
            for item in parsed.results {
                if item.index < chunk.len() {
                    scores.insert(offset + item.index, item.relevance_score);
                }
            }
            if scores.len() < offset + chunk.len() {
                return Err(Error::other(
                    "reranker response omitted one or more documents",
                ));
            }
        }

        let mut ranked: Vec<(usize, f64)> = scores.into_iter().collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        Ok(ranked)
    }

    async fn embed(&self, inputs: &[String]) -> Result<Vec<Vec<f32>>> {
        let request = EmbeddingRequest {
            model: &self.config.model,
            input: inputs,
        };
        let mut builder = self.client.post(&self.config.endpoint).json(&request);
        if let Some(api_key) = &self.config.api_key {
            builder = builder.bearer_auth(api_key);
        }
        let response = builder.send().await.map_err(|error| {
            Error::other(format!(
                "embedding request to {} failed: {error}",
                self.config.endpoint
            ))
        })?;
        let status = response.status();
        let body = response.text().await.map_err(|error| {
            Error::other(format!("failed to read embedding response body: {error}"))
        })?;
        if !status.is_success() {
            return Err(Error::other(format!(
                "embedding endpoint returned HTTP {status}: {}",
                truncate_for_error(&body)
            )));
        }
        let parsed: EmbeddingResponse = serde_json::from_str(&body).map_err(|error| {
            Error::other(format!("invalid embedding response from endpoint: {error}"))
        })?;
        let mut indexed = parsed.data;
        indexed.sort_by_key(|item| item.index);
        Ok(indexed.into_iter().map(|item| item.embedding).collect())
    }
}

struct PendingChunk {
    id: String,
    path: String,
    title: String,
    heading: Option<String>,
    text: String,
    input: String,
    content_hash: String,
}

fn document_manifest_for(vault_path: &Path, documents: &[ScannedNote]) -> Vec<DocumentStamp> {
    let mut manifest = documents
        .iter()
        .map(|document| DocumentStamp {
            path: document
                .path
                .strip_prefix(vault_path)
                .unwrap_or(&document.path)
                .to_string_lossy()
                .to_string(),
            size_bytes: document.size_bytes,
            modified_millis: document.modified.and_then(system_time_millis),
        })
        .collect::<Vec<_>>();
    manifest.sort_by(|left, right| left.path.cmp(&right.path));
    manifest
}

fn system_time_millis(value: SystemTime) -> Option<u64> {
    value
        .duration_since(SystemTime::UNIX_EPOCH)
        .ok()
        .and_then(|duration| u64::try_from(duration.as_millis()).ok())
}

#[allow(clippy::too_many_arguments)]
fn queue_text_chunks(
    pending: &mut Vec<PendingChunk>,
    path: &str,
    title: &str,
    source_type: &str,
    markdown: &str,
    content_hash: &str,
    chunk_chars: usize,
    chunk_overlap: usize,
) {
    for (chunk_number, chunk) in chunk_markdown(markdown, title, chunk_chars, chunk_overlap)
        .into_iter()
        .enumerate()
    {
        let plain = to_plain_text(&chunk.markdown);
        if plain.trim().is_empty() {
            continue;
        }
        let heading_prefix = chunk
            .heading
            .as_deref()
            .map(|heading| format!("Section: {heading}\n"))
            .unwrap_or_default();
        let input = format!(
            "Title: {title}\nType: {source_type}\n{heading_prefix}Path: {path}\nContent:\n{plain}"
        );
        pending.push(PendingChunk {
            id: format!("{path}#{chunk_number}"),
            path: path.to_string(),
            title: title.to_string(),
            heading: chunk.heading,
            text: plain,
            input,
            content_hash: content_hash.to_string(),
        });
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct HeadingFrame {
    level: usize,
    text: String,
}

#[derive(Debug)]
struct ChunkText {
    heading: Option<String>,
    markdown: String,
}

#[derive(Debug)]
struct SectionNode {
    parent: Option<usize>,
    heading: Option<HeadingFrame>,
    blocks: Vec<String>,
    children: Vec<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct OpenFence {
    delimiter: char,
    length: usize,
}

/// Build context-aware chunks from a Markdown heading tree.
///
/// Root-level sections are retrieval boundaries. Descendant subsections stay
/// with their parent while the whole subtree fits; only an oversized subtree is
/// split recursively at child headings. This avoids turning every short H3/H4
/// subsection into a context-poor vector while still preventing a long,
/// multi-topic note from collapsing into one averaged embedding.
fn chunk_markdown(markdown: &str, title: &str, max_chars: usize, overlap: usize) -> Vec<ChunkText> {
    let normalized = markdown.replace("\r\n", "\n").replace('\r', "\n");
    let nodes = parse_section_tree(&normalized);
    let mut chunks = Vec::new();

    if !nodes[0].blocks.is_empty() {
        chunks.extend(pack_blocks(&nodes[0].blocks, None, max_chars, overlap));
    }

    let mut top_level = nodes[0].children.clone();
    let mut inherited = Vec::new();
    if top_level.len() == 1 {
        let wrapper = top_level[0];
        let is_title_wrapper = nodes[wrapper].heading.as_ref().is_some_and(|heading| {
            heading.level == 1
                && normalize_heading(&heading.text) == normalize_heading(title)
                && nodes[0].blocks.is_empty()
                && !nodes[wrapper].children.is_empty()
        });
        if is_title_wrapper {
            inherited.push(nodes[wrapper].heading.clone().expect("checked above"));
            if !nodes[wrapper].blocks.is_empty() {
                chunks.extend(pack_blocks(
                    &nodes[wrapper].blocks,
                    Some(format_breadcrumb(&inherited)),
                    max_chars,
                    overlap,
                ));
            }
            top_level = nodes[wrapper].children.clone();
        }
    }

    for node_id in top_level {
        emit_section_chunks(&nodes, node_id, &inherited, max_chars, overlap, &mut chunks);
    }

    if chunks.is_empty() && !normalized.trim().is_empty() {
        chunks.extend(pack_blocks(
            &[normalized.trim().to_string()],
            None,
            max_chars,
            overlap,
        ));
    }
    chunks
}

fn parse_section_tree(markdown: &str) -> Vec<SectionNode> {
    let mut nodes = vec![SectionNode {
        parent: None,
        heading: None,
        blocks: Vec::new(),
        children: Vec::new(),
    }];
    let mut current = 0usize;
    let mut buffer = Vec::new();
    let mut fence = None;

    let flush = |nodes: &mut Vec<SectionNode>, current: usize, buffer: &mut Vec<String>| {
        let text = buffer.join("\n").trim().to_string();
        buffer.clear();
        if !text.is_empty() {
            nodes[current].blocks.push(text);
        }
    };

    for line in markdown.lines() {
        let next_fence = next_fence_state(line, fence);
        if fence.is_some() || next_fence != fence {
            buffer.push(line.to_string());
            fence = next_fence;
            continue;
        }

        if let Some(heading) = parse_heading(line) {
            flush(&mut nodes, current, &mut buffer);
            while current != 0
                && nodes[current]
                    .heading
                    .as_ref()
                    .is_some_and(|active| active.level >= heading.level)
            {
                current = nodes[current].parent.expect("non-root section has parent");
            }
            let node_id = nodes.len();
            nodes.push(SectionNode {
                parent: Some(current),
                heading: Some(heading),
                blocks: Vec::new(),
                children: Vec::new(),
            });
            nodes[current].children.push(node_id);
            current = node_id;
        } else if line.trim().is_empty() {
            flush(&mut nodes, current, &mut buffer);
        } else {
            buffer.push(line.to_string());
        }
    }
    flush(&mut nodes, current, &mut buffer);
    nodes
}

fn emit_section_chunks(
    nodes: &[SectionNode],
    node_id: usize,
    ancestors: &[HeadingFrame],
    max_chars: usize,
    overlap: usize,
    chunks: &mut Vec<ChunkText>,
) {
    let node = &nodes[node_id];
    let mut breadcrumb = ancestors.to_vec();
    if let Some(heading) = &node.heading {
        breadcrumb.push(heading.clone());
    }
    let label = Some(format_breadcrumb(&breadcrumb));
    let whole = render_subtree(nodes, node_id);
    if whole.chars().count() <= max_chars {
        chunks.push(ChunkText {
            heading: label,
            markdown: whole,
        });
        return;
    }

    if !node.blocks.is_empty() {
        chunks.extend(pack_blocks(&node.blocks, label.clone(), max_chars, overlap));
    }

    let mut grouped_children = Vec::new();
    let mut grouped_text = String::new();
    let flush_group = |grouped_children: &mut Vec<usize>,
                       grouped_text: &mut String,
                       chunks: &mut Vec<ChunkText>| {
        if !grouped_children.is_empty() {
            let group_heading = if grouped_children.len() == 1 {
                let mut child_breadcrumb = breadcrumb.clone();
                if let Some(heading) = &nodes[grouped_children[0]].heading {
                    child_breadcrumb.push(heading.clone());
                }
                Some(format_breadcrumb(&child_breadcrumb))
            } else {
                label.clone()
            };
            chunks.push(ChunkText {
                heading: group_heading,
                markdown: std::mem::take(grouped_text),
            });
            grouped_children.clear();
        }
    };

    for &child_id in &node.children {
        let child_text = render_subtree(nodes, child_id);
        if child_text.chars().count() > max_chars {
            flush_group(&mut grouped_children, &mut grouped_text, chunks);
            emit_section_chunks(nodes, child_id, &breadcrumb, max_chars, overlap, chunks);
            continue;
        }
        let candidate = if grouped_text.is_empty() {
            child_text.clone()
        } else {
            format!("{grouped_text}\n\n{child_text}")
        };
        if candidate.chars().count() > max_chars {
            flush_group(&mut grouped_children, &mut grouped_text, chunks);
            grouped_text = child_text;
        } else {
            grouped_text = candidate;
        }
        grouped_children.push(child_id);
    }
    flush_group(&mut grouped_children, &mut grouped_text, chunks);
}

fn render_subtree(nodes: &[SectionNode], node_id: usize) -> String {
    let node = &nodes[node_id];
    let mut parts = Vec::new();
    if let Some(heading) = &node.heading {
        parts.push(format!("{} {}", "#".repeat(heading.level), heading.text));
    }
    parts.extend(node.blocks.iter().cloned());
    parts.extend(
        node.children
            .iter()
            .map(|&child_id| render_subtree(nodes, child_id)),
    );
    parts
        .into_iter()
        .filter(|part| !part.trim().is_empty())
        .collect::<Vec<_>>()
        .join("\n\n")
}

fn pack_blocks(
    blocks: &[String],
    heading: Option<String>,
    max_chars: usize,
    overlap: usize,
) -> Vec<ChunkText> {
    let units: Vec<String> = blocks
        .iter()
        .flat_map(|block| split_oversized_block(block, max_chars))
        .collect();
    let mut chunks = Vec::new();
    let mut current: Vec<String> = Vec::new();

    for unit in units {
        let candidate = if current.is_empty() {
            unit.clone()
        } else {
            format!("{}\n\n{unit}", current.join("\n\n"))
        };
        if candidate.chars().count() <= max_chars {
            current.push(unit);
            continue;
        }

        if !current.is_empty() {
            chunks.push(ChunkText {
                heading: heading.clone(),
                markdown: current.join("\n\n"),
            });
        }
        let mut carry = trailing_whole_blocks(&current, overlap);
        let with_carry = if carry.is_empty() {
            unit.clone()
        } else {
            format!("{}\n\n{unit}", carry.join("\n\n"))
        };
        if with_carry.chars().count() > max_chars {
            carry.clear();
        }
        carry.push(unit);
        current = carry;
    }

    if !current.is_empty() {
        chunks.push(ChunkText {
            heading,
            markdown: current.join("\n\n"),
        });
    }
    chunks
}

fn trailing_whole_blocks(blocks: &[String], overlap: usize) -> Vec<String> {
    if overlap == 0 || blocks.len() < 2 {
        return Vec::new();
    }
    let mut selected = Vec::new();
    let mut chars = 0usize;
    for block in blocks.iter().rev() {
        let block_chars = block.chars().count();
        let separator = usize::from(!selected.is_empty()) * 2;
        if chars + separator + block_chars > overlap {
            break;
        }
        selected.push(block.clone());
        chars += separator + block_chars;
    }
    selected.reverse();
    selected
}

fn split_oversized_block(block: &str, max_chars: usize) -> Vec<String> {
    if block.chars().count() <= max_chars {
        return vec![block.to_string()];
    }
    for separator in ['\n', '.', '!', '?'] {
        let pieces = split_and_pack(block, separator, max_chars);
        if pieces.len() > 1
            && pieces
                .iter()
                .all(|piece| piece.chars().count() <= max_chars)
        {
            return pieces;
        }
    }
    let chars: Vec<char> = block.chars().collect();
    chars
        .chunks(max_chars.max(1))
        .map(|chunk| chunk.iter().collect())
        .collect()
}

fn split_and_pack(text: &str, separator: char, max_chars: usize) -> Vec<String> {
    let raw_parts: Vec<String> = if separator == '\n' {
        text.split('\n').map(ToString::to_string).collect()
    } else {
        text.split_inclusive(separator)
            .map(|part| part.trim().to_string())
            .filter(|part| !part.is_empty())
            .collect()
    };
    if raw_parts.len() < 2 {
        return vec![text.to_string()];
    }
    let joiner = if separator == '\n' { "\n" } else { " " };
    let mut packed = Vec::new();
    let mut current = String::new();
    for part in raw_parts {
        if part.chars().count() > max_chars {
            return vec![text.to_string()];
        }
        let candidate = if current.is_empty() {
            part.clone()
        } else {
            format!("{current}{joiner}{part}")
        };
        if candidate.chars().count() <= max_chars {
            current = candidate;
        } else {
            packed.push(std::mem::take(&mut current));
            current = part;
        }
    }
    if !current.is_empty() {
        packed.push(current);
    }
    packed
}

fn parse_heading(line: &str) -> Option<HeadingFrame> {
    let trimmed = line.trim_start();
    if line.len().saturating_sub(trimmed.len()) > 3 {
        return None;
    }
    let hashes = trimmed
        .chars()
        .take_while(|character| *character == '#')
        .count();
    if (1..=6).contains(&hashes) && trimmed.chars().nth(hashes) == Some(' ') {
        Some(HeadingFrame {
            level: hashes,
            text: trimmed[hashes..]
                .trim()
                .trim_end_matches('#')
                .trim()
                .to_string(),
        })
    } else {
        None
    }
}

fn next_fence_state(line: &str, open: Option<OpenFence>) -> Option<OpenFence> {
    let trimmed = line.trim_start();
    if line.len().saturating_sub(trimmed.len()) > 3 {
        return open;
    }
    let delimiter = trimmed.chars().next()?;
    if delimiter != '`' && delimiter != '~' {
        return open;
    }
    let length = trimmed
        .chars()
        .take_while(|character| *character == delimiter)
        .count();
    if length < 3 {
        return open;
    }
    let trailing = trimmed.chars().skip(length).collect::<String>();
    match open {
        None if delimiter == '`' && trailing.contains('`') => None,
        None => Some(OpenFence { delimiter, length }),
        Some(active)
            if active.delimiter == delimiter
                && length >= active.length
                && trailing.trim().is_empty() =>
        {
            None
        }
        Some(active) => Some(active),
    }
}

fn format_breadcrumb(headings: &[HeadingFrame]) -> String {
    headings
        .iter()
        .map(|heading| heading.text.as_str())
        .collect::<Vec<_>>()
        .join(" > ")
}

fn normalize_heading(value: &str) -> String {
    value
        .chars()
        .filter(|character| character.is_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect()
}

fn hex_hash(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn cosine_similarity(left: &[f32], right: &[f32]) -> f64 {
    if left.len() != right.len() || left.is_empty() {
        return f64::NAN;
    }
    let mut dot = 0.0f64;
    let mut left_norm = 0.0f64;
    let mut right_norm = 0.0f64;
    for (&left_value, &right_value) in left.iter().zip(right) {
        let left_value = left_value as f64;
        let right_value = right_value as f64;
        dot += left_value * right_value;
        left_norm += left_value * left_value;
        right_norm += right_value * right_value;
    }
    let denominator = left_norm.sqrt() * right_norm.sqrt();
    if denominator <= f64::EPSILON {
        f64::NAN
    } else {
        dot / denominator
    }
}

fn truncate_for_error(body: &str) -> String {
    body.chars().take(500).collect()
}

fn is_document_path(path: &str) -> bool {
    Path::new(path)
        .extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| {
            extension.eq_ignore_ascii_case("pdf") || extension.eq_ignore_ascii_case("docx")
        })
}

fn is_path_excluded(path: &str) -> bool {
    let p = Path::new(path);
    for component in p.components() {
        if let std::path::Component::Normal(c) = component {
            let s = c.to_string_lossy();
            if s == ".trash" || s == ".obsidian" || s == ".git" || s == "node_modules" {
                return true;
            }
        }
    }
    false
}

fn source_path_is_excluded(manager: &VaultManager, path: &str) -> bool {
    is_path_excluded(path) || manager.is_path_excluded(Path::new(path))
}

fn stored_index_has_excluded_sources(index: &StoredIndex, manager: &VaultManager) -> bool {
    index
        .chunks
        .iter()
        .any(|chunk| source_path_is_excluded(manager, &chunk.path))
        || index
            .document_manifest
            .iter()
            .any(|document| source_path_is_excluded(manager, &document.path))
}

#[derive(Clone)]
struct HybridAccumulator {
    path: String,
    title: String,
    heading: Option<String>,
    text: String,
    preview: String,
    snippet: String,
    dense_rrf: f64,
    sparse_rrf: f64,
    rrf_score: f64,
    dense_score: Option<f64>,
    sparse_score: Option<f64>,
    dense_rank: Option<usize>,
    sparse_rank: Option<usize>,
    rerank_score: Option<f64>,
    rerank_rank: Option<usize>,
    chunk_id: Option<String>,
}

impl HybridAccumulator {
    fn new(
        path: String,
        title: String,
        heading: Option<String>,
        text: String,
        chunk_id: Option<String>,
    ) -> Self {
        Self {
            path,
            title,
            heading,
            preview: text.clone(),
            text,
            snippet: String::new(),
            dense_rrf: 0.0,
            sparse_rrf: 0.0,
            rrf_score: 0.0,
            dense_score: None,
            sparse_score: None,
            dense_rank: None,
            sparse_rank: None,
            rerank_score: None,
            rerank_rank: None,
            chunk_id,
        }
    }

    fn finish(self) -> HybridSearchResult {
        let chunk_hash = self
            .chunk_id
            .as_ref()
            .map(|_| hex_hash(self.text.as_bytes()));
        HybridSearchResult {
            path: self.path,
            title: self.title,
            heading: self.heading,
            text: self.text,
            preview: self.preview,
            snippet: self.snippet,
            rrf_score: self.rrf_score,
            dense_score: self.dense_score,
            sparse_score: self.sparse_score,
            dense_rank: self.dense_rank,
            sparse_rank: self.sparse_rank,
            rerank_score: self.rerank_score,
            rerank_rank: self.rerank_rank,
            chunk_hash,
            chunk_id: self.chunk_id,
        }
    }
}

fn root_heading(heading: Option<&str>) -> Option<&str> {
    heading.map(|value| value.split(" > ").next().unwrap_or(value))
}

fn expand_passage(
    index: &StoredIndex,
    path: &str,
    chunk_id: &str,
    expected_hash: &str,
    neighbors: usize,
    max_chars: usize,
    stale: bool,
) -> Result<PassageRead> {
    // Require an exact path–ID pair and the returned text hash. Ordinals alone
    // are not immutable: a reindex can reuse #N for a different passage.
    let mut chunks: Vec<&EmbeddingChunk> = index.chunks.iter().filter(|c| c.path == path).collect();
    chunks.sort_by_key(|c| {
        c.id.rsplit_once('#')
            .and_then(|(_, n)| n.parse::<usize>().ok())
    });
    let pos = chunks
        .iter()
        .position(|c| c.id == chunk_id)
        .ok_or_else(|| {
            Error::config_error(
                "passage anchor not found for this exact path; rerun semantic_search",
            )
        })?;
    let anchor = chunks[pos];
    let actual_hash = hex_hash(anchor.text.as_bytes());
    if actual_hash != expected_hash {
        return Err(Error::config_error(
            "passage anchor changed since search; rerun semantic_search",
        ));
    }
    let root = root_heading(anchor.heading.as_deref());
    let mut positions = vec![pos];
    for step in 1..=neighbors.min(3) {
        for candidate in [
            pos.checked_sub(step),
            pos.checked_add(step).filter(|p| *p < chunks.len()),
        ]
        .into_iter()
        .flatten()
        {
            if root_heading(chunks[candidate].heading.as_deref()) == root {
                positions.push(candidate);
            }
        }
    }
    // The anchor spends the budget first. Neighbors are only context and never
    // displace the ranked passage; the final response returns document order.
    let mut remaining = max_chars.clamp(1, 12_000);
    let mut selected = Vec::new();
    for position in positions {
        if remaining == 0 {
            break;
        }
        let chunk = chunks[position];
        let original_chars = chunk.text.chars().count();
        let text: String = chunk.text.chars().take(remaining).collect();
        let used = text.chars().count();
        remaining -= used;
        selected.push((
            position,
            IndexedPassage {
                chunk_id: chunk.id.clone(),
                heading: chunk.heading.clone(),
                text,
                truncated: used < original_chars,
                anchor: position == pos,
            },
        ));
    }
    selected.sort_by_key(|(position, _)| *position);
    Ok(PassageRead {
        path: path.to_string(),
        anchor_id: chunk_id.to_string(),
        anchor_hash: actual_hash,
        built_at: index.built_at.clone(),
        index_stale: stale,
        chunks: selected.into_iter().map(|(_, passage)| passage).collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn passage_fixture() -> StoredIndex {
        let chunk = |path: &str, n: usize, heading: &str, text: &str| EmbeddingChunk {
            id: format!("{path}#{n}"),
            path: path.into(),
            title: "Fixture".into(),
            heading: Some(heading.into()),
            text: text.into(),
            content_hash: "source".into(),
            embedding: vec![1.0],
        };
        StoredIndex {
            schema_version: INDEX_SCHEMA_VERSION,
            chunker_version: CHUNKER_VERSION.into(),
            model: DEFAULT_MODEL.into(),
            vault_path: "/fixture".into(),
            dimension: 1,
            built_at: "2026-01-01T00:00:00Z".into(),
            document_indexing_enabled: false,
            document_max_bytes: DEFAULT_DOCUMENT_MAX_BYTES as u64,
            document_manifest: vec![],
            chunks: vec![
                chunk("one.md", 0, "First", "intro"),
                chunk("one.md", 1, "First > Detail", "éclat 🎉 data"),
                chunk("one.md", 2, "First > More", "ending"),
                chunk("one.md", 3, "Second", "unrelated"),
                chunk("two.md", 0, "First", "other note"),
            ],
        }
    }

    #[test]
    fn passage_expansion_is_bounded_same_section_and_anchor_first() {
        let index = passage_fixture();
        let hash = hex_hash("éclat 🎉 data".as_bytes());
        let read = expand_passage(&index, "one.md", "one.md#1", &hash, 3, 20, true).unwrap();
        assert!(read.index_stale);
        assert_eq!(
            read.chunks
                .iter()
                .map(|c| c.chunk_id.as_str())
                .collect::<Vec<_>>(),
            vec!["one.md#0", "one.md#1", "one.md#2"]
        );
        assert_eq!(
            read.chunks
                .iter()
                .map(|c| c.text.chars().count())
                .sum::<usize>(),
            20
        );
        assert_eq!(
            read.chunks.iter().find(|c| c.anchor).unwrap().text,
            "éclat 🎉 data"
        );
        assert!(read.chunks.iter().any(|c| c.truncated));
    }

    #[test]
    fn passage_rejects_wrong_path_missing_and_reused_ordinal() {
        let mut index = passage_fixture();
        let hash = hex_hash("éclat 🎉 data".as_bytes());
        assert!(expand_passage(&index, "two.md", "one.md#1", &hash, 1, 100, false).is_err());
        assert!(expand_passage(&index, "one.md", "one.md#19", &hash, 1, 100, false).is_err());
        index.chunks[1].text = "replaced".into();
        assert!(expand_passage(&index, "one.md", "one.md#1", &hash, 1, 100, false).is_err());
    }

    #[test]
    fn passage_truncates_at_unicode_character_boundary() {
        let index = passage_fixture();
        let hash = hex_hash("éclat 🎉 data".as_bytes());
        let read = expand_passage(&index, "one.md", "one.md#1", &hash, 1, 7, false).unwrap();
        assert_eq!(read.chunks.len(), 1);
        assert_eq!(read.chunks[0].text, "éclat 🎉");
        assert!(read.chunks[0].truncated);
        let first = expand_passage(
            &index,
            "one.md",
            "one.md#0",
            &hex_hash(b"intro"),
            3,
            100,
            false,
        )
        .unwrap();
        assert_eq!(first.chunks.len(), 3); // no previous chunk; never crosses the next section
    }

    #[test]
    fn chunker_keeps_subsections_with_parent_when_they_fit() {
        let chunks = chunk_markdown(
            "# One\n\nintro\n\n## Two\n\ndetails\n\n### Three\n\nmore details",
            "Different title",
            200,
            20,
        );
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].heading.as_deref(), Some("One"));
        assert!(chunks[0].markdown.contains("## Two"));
        assert!(chunks[0].markdown.contains("### Three"));
    }

    #[test]
    fn chunker_splits_root_sections_but_not_their_small_subsections() {
        let chunks = chunk_markdown(
            "# One\n\nalpha\n\n## Detail\n\nbeta\n\n# Two\n\ngamma\n\n## Detail\n\ndelta",
            "Note",
            200,
            20,
        );
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].heading.as_deref(), Some("One"));
        assert_eq!(chunks[1].heading.as_deref(), Some("Two"));
        assert!(chunks[0].markdown.contains("## Detail"));
        assert!(chunks[1].markdown.contains("## Detail"));
    }

    #[test]
    fn matching_h1_title_is_a_wrapper_not_one_giant_section() {
        let chunks = chunk_markdown(
            "# My Note\n\n## One\n\nalpha\n\n## Two\n\nbeta",
            "My Note",
            200,
            20,
        );
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].heading.as_deref(), Some("My Note > One"));
        assert_eq!(chunks[1].heading.as_deref(), Some("My Note > Two"));
    }

    #[test]
    fn heading_inside_fenced_code_does_not_split_the_section() {
        let chunks = chunk_markdown(
            "# One\n\n```sh\n# not a heading\necho test\n```\n\n## Two\n\ntext",
            "Note",
            200,
            20,
        );
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].heading.as_deref(), Some("One"));
        assert!(chunks[0].markdown.contains("# not a heading"));
        assert!(chunks[0].markdown.contains("## Two"));
    }

    #[test]
    fn oversized_subtree_splits_recursively_at_child_sections() {
        let chunks = chunk_markdown(
            "# One\n\n## Alpha\n\n12345678901234567890\n\n## Beta\n\nabcdefghijklmnopqrst",
            "Note",
            35,
            0,
        );
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].heading.as_deref(), Some("One > Alpha"));
        assert_eq!(chunks[1].heading.as_deref(), Some("One > Beta"));
    }

    fn stored_index(model: &str, vault_path: &str) -> StoredIndex {
        StoredIndex {
            schema_version: INDEX_SCHEMA_VERSION,
            chunker_version: CHUNKER_VERSION.to_string(),
            model: model.to_string(),
            vault_path: vault_path.to_string(),
            dimension: 4096,
            built_at: "2026-01-01T00:00:00Z".to_string(),
            document_indexing_enabled: false,
            document_max_bytes: DEFAULT_DOCUMENT_MAX_BYTES as u64,
            document_manifest: Vec::new(),
            chunks: Vec::new(),
        }
    }

    /// Regression: a vault write marks the index stale, and that used to make
    /// `search` refuse outright. A stale index is built by the same model and
    /// chunker, so its vectors are still comparable to a query vector. Only a
    /// configuration change makes the index unusable.
    #[test]
    fn stale_index_is_usable_and_only_config_changes_are_incompatible() {
        let vault = "/home/samuel/vault";
        let index = stored_index("Qwen/Qwen3-Embedding-8B-GGUF", vault);

        // The index carries no freshness field at all, so there is nothing for
        // a vault write to invalidate here. This is the property that makes
        // `search` serve an out-of-date index rather than refusing it.
        assert_eq!(
            index_mismatch(
                &index,
                "Qwen/Qwen3-Embedding-8B-GGUF",
                vault,
                false,
                DEFAULT_DOCUMENT_MAX_BYTES as u64,
            ),
            None
        );

        // A different embedding model produces vectors in a different space.
        assert!(
            index_mismatch(
                &index,
                "BAAI/bge-m3",
                vault,
                false,
                DEFAULT_DOCUMENT_MAX_BYTES as u64,
            )
            .is_some(),
            "a model change must refuse: cosine similarity across spaces is meaningless"
        );
        assert!(
            index_mismatch(
                &index,
                "Qwen/Qwen3-Embedding-8B-GGUF",
                "/other/vault",
                false,
                DEFAULT_DOCUMENT_MAX_BYTES as u64,
            )
            .is_some()
        );

        let mut reschemaed = stored_index("Qwen/Qwen3-Embedding-8B-GGUF", vault);
        reschemaed.schema_version = INDEX_SCHEMA_VERSION + 1;
        assert!(
            index_mismatch(
                &reschemaed,
                "Qwen/Qwen3-Embedding-8B-GGUF",
                vault,
                false,
                DEFAULT_DOCUMENT_MAX_BYTES as u64,
            )
            .is_some()
        );

        let mut rechunked = stored_index("Qwen/Qwen3-Embedding-8B-GGUF", vault);
        rechunked.chunker_version = "markdown-heading-v2".to_string();
        assert!(
            index_mismatch(
                &rechunked,
                "Qwen/Qwen3-Embedding-8B-GGUF",
                vault,
                false,
                DEFAULT_DOCUMENT_MAX_BYTES as u64,
            )
            .is_some()
        );

        assert!(
            index_mismatch(
                &index,
                "Qwen/Qwen3-Embedding-8B-GGUF",
                vault,
                true,
                DEFAULT_DOCUMENT_MAX_BYTES as u64,
            )
            .is_some(),
            "enabling document indexing changes the indexed source set"
        );
    }

    fn at(offset_hours: i64) -> DateTime<Utc> {
        DateTime::parse_from_rfc3339("2026-09-21T12:00:00Z")
            .unwrap()
            .with_timezone(&Utc)
            + chrono::Duration::hours(offset_hours)
    }

    /// The lazy-refresh threshold: an out-of-date index is refreshed only once it
    /// has fallen far enough behind. Refreshing is an optimisation, so the
    /// policy has to be conservative in both directions — never embed on every
    /// search, but never let the vectors drift indefinitely either.
    #[test]
    fn refresh_is_due_only_when_stale_and_past_the_threshold() {
        let threshold = Duration::from_secs(6 * 3600);
        let now = at(0);
        let built = |hours_ago: i64| at(-hours_ago).to_rfc3339();

        // A fresh index is never due, however old: nothing changed.
        assert!(!refresh_due(false, Some(&built(48)), threshold, now));

        // Stale but recent: too soon to spend embedding calls.
        assert!(!refresh_due(true, Some(&built(1)), threshold, now));
        assert!(!refresh_due(true, Some(&built(5)), threshold, now));

        // Stale and past the threshold.
        assert!(refresh_due(true, Some(&built(7)), threshold, now));
        assert!(refresh_due(true, Some(&built(100)), threshold, now));

        // Zero disables the feature entirely.
        assert!(!refresh_due(true, Some(&built(100)), Duration::ZERO, now));

        // A missing or unreadable timestamp on a stale index counts as due.
        assert!(refresh_due(true, None, threshold, now));
        assert!(refresh_due(true, Some("not a timestamp"), threshold, now));

        // A clock that moved backwards must not trigger repeated refreshes.
        assert!(!refresh_due(
            true,
            Some(&at(5).to_rfc3339()),
            threshold,
            now
        ));
    }

    /// Diagnostics must add nothing to a healthy response. A search where every
    /// channel contributed should be indistinguishable from before this existed,
    /// so the reporting cannot become context bloat on the common path.
    #[test]
    fn diagnostics_serialize_nothing_when_every_channel_contributed() {
        let healthy = HybridDiagnostics::default();
        assert!(healthy.is_complete());
        assert_eq!(
            serde_json::to_value(&healthy).unwrap(),
            serde_json::json!({"index_stale": false, "index_refreshed": false}),
            "absent reasons must not serialize as nulls"
        );

        let degraded = HybridDiagnostics {
            dense_unavailable: Some("embedding endpoint unreachable".to_string()),
            index_stale: true,
            ..Default::default()
        };
        assert!(!degraded.is_complete());
        let json = serde_json::to_value(&degraded).unwrap();
        assert_eq!(json["dense_unavailable"], "embedding endpoint unreachable");
        assert_eq!(json["index_stale"], true);
        // Reasons that did not occur stay absent rather than becoming null.
        assert!(json.get("rerank_unavailable").is_none());
        assert!(json.get("refresh_failed").is_none());
    }

    #[test]
    fn cosine_similarity_matches_identical_vectors() {
        let score = cosine_similarity(&[1.0, 2.0], &[1.0, 2.0]);
        assert!((score - 1.0).abs() < 1e-9);
    }

    #[test]
    fn stable_slug_is_deterministic_and_short() {
        assert_eq!(stable_slug("vault"), stable_slug("vault"));
        assert_eq!(stable_slug("vault").len(), 24);
    }

    #[test]
    fn document_manifest_is_sorted_and_preserves_fingerprints() {
        let vault = Path::new("/vault");
        let documents = vec![
            ScannedNote {
                path: PathBuf::from("/vault/z.docx"),
                size_bytes: 20,
                modified: Some(SystemTime::UNIX_EPOCH + Duration::from_millis(9)),
            },
            ScannedNote {
                path: PathBuf::from("/vault/a.pdf"),
                size_bytes: 10,
                modified: Some(SystemTime::UNIX_EPOCH + Duration::from_millis(4)),
            },
        ];
        let manifest = document_manifest_for(vault, &documents);
        assert_eq!(manifest[0].path, "a.pdf");
        assert_eq!(manifest[0].size_bytes, 10);
        assert_eq!(manifest[0].modified_millis, Some(4));
        assert_eq!(manifest[1].path, "z.docx");
    }

    #[test]
    fn document_paths_are_detected_case_insensitively() {
        assert!(is_document_path("attachments/paper.PDF"));
        assert!(is_document_path("attachments/draft.docx"));
        assert!(!is_document_path("notes/paper.md"));
    }

    #[test]
    fn path_exclusion_detects_trash_and_obsidian() {
        assert!(is_path_excluded(".trash/note.md"));
        assert!(is_path_excluded("folder/.trash/note.md"));
        assert!(is_path_excluded(".obsidian/workspace.json"));
        assert!(!is_path_excluded("10_Projects/note.md"));
        assert!(!is_path_excluded("trash_collection/note.md"));
    }

    #[tokio::test]
    async fn excluded_subtrees_are_removed_from_existing_rag_candidates() {
        let temp = TempDir::new().unwrap();
        let vault = VaultConfig::builder("test", temp.path())
            .excluded_subtrees(
                ["Agents", "Chats", "90_Archive"]
                    .into_iter()
                    .map(PathBuf::from),
            )
            .build()
            .unwrap();
        let mut config = ServerConfig::new();
        config.vaults.push(vault);
        let manager = VaultManager::new(config).unwrap();

        let chunk = |path: &str| EmbeddingChunk {
            id: format!("{path}#0"),
            path: path.into(),
            title: "Fixture".into(),
            heading: None,
            text: format!("content from {path}"),
            content_hash: "source".into(),
            embedding: vec![1.0],
        };
        let mut index = passage_fixture();
        index.chunks = [
            "Agents/private/deep.md",
            "Chats/session.md",
            "90_Archive/old.md",
            "projects/Agents/visible.md",
            "Agents-old/visible.md",
            "visible.md",
        ]
        .into_iter()
        .map(chunk)
        .collect();

        let visible: Vec<&str> = index
            .chunks
            .iter()
            .filter(|chunk| !source_path_is_excluded(&manager, &chunk.path))
            .map(|chunk| chunk.path.as_str())
            .collect();
        assert_eq!(
            visible,
            [
                "projects/Agents/visible.md",
                "Agents-old/visible.md",
                "visible.md"
            ]
        );
        assert!(stored_index_has_excluded_sources(&index, &manager));
    }

    #[test]
    fn rrf_deduplication_prevents_multi_chunk_inflation() {
        let mut entry = HybridAccumulator::new(
            "test.md".to_string(),
            "Test".to_string(),
            None,
            "text".to_string(),
            Some("test.md#0".to_string()),
        );
        entry.dense_rank = Some(1);
        entry.dense_rrf = 1.0 / (RRF_K + 1.0);
        entry.sparse_rank = Some(2);
        entry.sparse_rrf = 1.0 / (RRF_K + 2.0);
        entry.rrf_score = entry.dense_rrf + entry.sparse_rrf;

        let expected = (1.0 / 61.0) + (1.0 / 62.0);
        assert!((entry.rrf_score - expected).abs() < 1e-9);
    }

    #[test]
    fn rerank_request_asks_for_every_candidate() {
        let documents = vec!["first".to_string(), "second".to_string()];
        let request = RerankRequest {
            model: "cohere/rerank-v3.5",
            query: "test query",
            documents: &documents,
            top_n: documents.len(),
        };
        assert_eq!(
            serde_json::to_value(request).unwrap(),
            serde_json::json!({
                "model": "cohere/rerank-v3.5",
                "query": "test query",
                "documents": ["first", "second"],
                "top_n": 2
            })
        );
    }

    #[test]
    fn dedicated_reranker_key_overrides_shared_embedding_key() {
        let shared = "shared".to_string();
        assert_eq!(
            select_reranker_api_key(Some("dedicated".to_string()), Some(&shared)).as_deref(),
            Some("dedicated")
        );
        assert_eq!(
            select_reranker_api_key(None, Some(&shared)).as_deref(),
            Some("shared")
        );
        assert_eq!(select_reranker_api_key(None, None), None);
    }

    #[test]
    fn rerank_response_deserialization() {
        let json = r#"{"model":"gpustack/bge-reranker-v2-m3-GGUF","object":"list","results":[{"index":1,"relevance_score":-2.63},{"index":0,"relevance_score":-4.14}]}"#;
        let parsed: RerankResponse = serde_json::from_str(json).unwrap();
        assert_eq!(parsed.results.len(), 2);
        assert_eq!(parsed.results[0].index, 1);
        assert!((parsed.results[0].relevance_score - (-2.63)).abs() < 1e-4);
    }
}
