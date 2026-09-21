//! Dense and hybrid retrieval for Markdown vaults.
//!
//! The engine deliberately keeps embedding inference outside TurboVault. It
//! talks to an OpenAI-compatible `/embeddings` endpoint (normally the local
//! Qwen3 service used by the Zotero MCP fork), persists a versioned local
//! vector index, and combines dense results with TurboVault's sparse search.
//!
//! The index is not stored in the vault. It is derived state keyed by the
//! vault path, embedding model, and chunker version. Vault writes mark the
//! index stale; callers must explicitly run `reindex_embeddings` before using
//! dense retrieval again.

use reqwest::Client;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::cmp::Ordering;
use std::collections::HashMap;
use std::env;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};
use std::time::Duration;
use tokio::sync::RwLock;
use turbovault_core::prelude::*;
use turbovault_parser::to_plain_text;
use turbovault_vault::VaultManager;

use crate::search_engine::{SearchEngine, SearchQuery};

const INDEX_SCHEMA_VERSION: u32 = 1;
const CHUNKER_VERSION: &str = "markdown-heading-v1";
const DEFAULT_ENDPOINT: &str = "http://127.0.0.1:8082/v1/embeddings";
const DEFAULT_MODEL: &str = "Qwen/Qwen3-Embedding-8B-GGUF";
const DEFAULT_CHUNK_CHARS: usize = 2400;
const DEFAULT_CHUNK_OVERLAP: usize = 200;
const DEFAULT_BATCH_SIZE: usize = 16;
const DEFAULT_RERANKER_ENDPOINT: &str = "http://127.0.0.1:8083/v1/rerank";
const DEFAULT_RERANKER_MODEL: &str = "BAAI/bge-reranker-v2-m3";
const DEFAULT_RERANK_BATCH_SIZE: usize = 12;
const RRF_K: f64 = 60.0;

/// Runtime configuration for the external embedding endpoint and local reranker.
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
    pub reranker_endpoint: String,
    pub reranker_model: String,
    pub reranker_enabled: bool,
    pub reranker_batch_size: usize,
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

        let reranker_endpoint = env::var("TURBOVAULT_RERANKER_ENDPOINT")
            .unwrap_or_else(|_| DEFAULT_RERANKER_ENDPOINT.to_string());
        let reranker_model = env::var("TURBOVAULT_RERANKER_MODEL")
            .unwrap_or_else(|_| DEFAULT_RERANKER_MODEL.to_string());
        let reranker_enabled = env::var("TURBOVAULT_RERANKER_ENABLED")
            .map(|val| !val.trim().is_empty() && val != "0" && !val.eq_ignore_ascii_case("false"))
            .unwrap_or(!reranker_endpoint.trim().is_empty());
        let reranker_batch_size =
            env_usize("TURBOVAULT_RERANKER_BATCH_SIZE", DEFAULT_RERANK_BATCH_SIZE)?;

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
            reranker_endpoint,
            reranker_model,
            reranker_enabled,
            reranker_batch_size,
        })
    }
}

fn env_usize(name: &str, default: usize) -> Result<usize> {
    match env::var(name) {
        Ok(value) => value.parse::<usize>().map_err(|_| {
            Error::config_error(format!("{name} must be a positive integer, got {value:?}"))
        }),
        Err(_) => Ok(default),
    }
}

fn default_index_root() -> PathBuf {
    if let Some(cache) = env::var_os("XDG_CACHE_HOME") {
        return PathBuf::from(cache).join("turbovault/embeddings");
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

#[derive(Debug, Clone, Serialize, Deserialize)]
struct StoredIndex {
    schema_version: u32,
    chunker_version: String,
    model: String,
    vault_path: String,
    dimension: usize,
    built_at: String,
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
}

#[derive(Debug, Clone, Serialize)]
pub struct EmbeddingIndexStatus {
    pub configured: bool,
    pub endpoint: String,
    pub model: String,
    pub api_key_configured: bool,
    pub index_path: String,
    pub stale: bool,
    pub exists: bool,
    pub chunks: usize,
    pub dimensions: usize,
    pub built_at: Option<String>,
    pub schema_version: Option<u32>,
    pub chunker_version: Option<String>,
    pub reranker_endpoint: String,
    pub reranker_model: String,
    pub reranker_enabled: bool,
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

/// Cached dense index and its endpoint client.
pub struct EmbeddingEngine {
    manager: Arc<VaultManager>,
    config: EmbeddingConfig,
    client: Client,
    index_path: PathBuf,
    stale_path: PathBuf,
    state: RwLock<Option<StoredIndex>>,
    stale: AtomicBool,
}

impl EmbeddingEngine {
    pub async fn new(manager: Arc<VaultManager>) -> Result<Self> {
        let config = EmbeddingConfig::from_environment(manager.vault_path())?;
        tokio::fs::create_dir_all(&config.index_dir).await?;
        let index_path = config.index_dir.join("index.bin");
        let stale_path = config.index_dir.join("index.stale");
        let mut stale = tokio::fs::try_exists(&stale_path).await.unwrap_or(false);
        let state = if tokio::fs::try_exists(&index_path).await.unwrap_or(false) {
            let bytes = tokio::fs::read(&index_path).await.map_err(|error| {
                Error::other(format!("failed to read embedding index: {error}"))
            })?;
            let parsed: StoredIndex = bincode::deserialize(&bytes).map_err(|error| {
                Error::other(format!(
                    "failed to decode embedding index {}; run reindex_embeddings: {error}",
                    index_path.display()
                ))
            })?;
            if parsed.schema_version != INDEX_SCHEMA_VERSION
                || parsed.chunker_version != CHUNKER_VERSION
                || parsed.model != config.model
                || parsed.vault_path != manager.vault_path().to_string_lossy()
            {
                stale = true;
            }
            Some(parsed)
        } else {
            None
        };

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
        })
    }

    pub async fn status(&self) -> EmbeddingIndexStatus {
        let state = self.state.read().await;
        EmbeddingIndexStatus {
            configured: !self.config.endpoint.is_empty() && !self.config.model.is_empty(),
            endpoint: self.config.endpoint.clone(),
            model: self.config.model.clone(),
            api_key_configured: self.config.api_key_configured,
            index_path: self.index_path.display().to_string(),
            stale: self.stale.load(AtomicOrdering::Acquire),
            exists: state.is_some(),
            chunks: state.as_ref().map(|index| index.chunks.len()).unwrap_or(0),
            dimensions: state.as_ref().map(|index| index.dimension).unwrap_or(0),
            built_at: state.as_ref().map(|index| index.built_at.clone()),
            schema_version: state.as_ref().map(|index| index.schema_version),
            chunker_version: state.as_ref().map(|index| index.chunker_version.clone()),
            reranker_endpoint: self.config.reranker_endpoint.clone(),
            reranker_model: self.config.reranker_model.clone(),
            reranker_enabled: self.config.reranker_enabled,
        }
    }

    /// Mark derived vectors unusable after any vault mutation. The marker is
    /// persisted so a process restart cannot accidentally serve stale vectors.
    pub async fn mark_stale(&self) -> Result<()> {
        self.stale.store(true, AtomicOrdering::Release);
        tokio::fs::write(&self.stale_path, b"stale\n").await?;
        Ok(())
    }

    pub async fn reindex(&self) -> Result<EmbeddingIndexStatus> {
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

            for (chunk_number, chunk) in chunk_markdown(
                &vault_file.content,
                self.config.chunk_chars,
                self.config.chunk_overlap,
            )
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
                let input =
                    format!("Title: {title}\n{heading_prefix}Path: {relative}\nContent:\n{plain}");
                pending.push(PendingChunk {
                    id: format!("{relative}#{chunk_number}"),
                    path: relative.clone(),
                    title: title.clone(),
                    heading: chunk.heading,
                    text: plain,
                    input,
                    content_hash: content_hash.clone(),
                });
            }
        }

        if !pending.is_empty() {
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
            }
        }

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
        Ok(self.status().await)
    }

    pub async fn search(&self, query: &str, limit: usize) -> Result<Vec<EmbeddingSearchResult>> {
        if self.stale.load(AtomicOrdering::Acquire) {
            return Err(Error::config_error(
                "embedding index is stale; run reindex_embeddings before dense search",
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

    pub async fn hybrid_search(
        &self,
        search_engine: &SearchEngine,
        query: &str,
        limit: usize,
    ) -> Result<Vec<HybridSearchResult>> {
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
            Ok(results) => results,
            Err(error) => {
                log::warn!("sparse search unavailable in hybrid retrieval, using dense: {error}");
                sparse_error = Some(error);
                Vec::new()
            }
        };
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
        Ok(results.into_iter().map(HybridAccumulator::finish).collect())
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
            };
            let mut builder = self
                .client
                .post(&self.config.reranker_endpoint)
                .json(&request);
            if let Some(api_key) = &self.config.api_key {
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

#[derive(Debug)]
struct ChunkText {
    heading: Option<String>,
    markdown: String,
}

fn chunk_markdown(markdown: &str, max_chars: usize, overlap: usize) -> Vec<ChunkText> {
    let mut paragraphs: Vec<(Option<String>, String)> = Vec::new();
    let mut heading: Option<String> = None;
    let mut paragraph = String::new();

    let mut flush_paragraph = |paragraph: &mut String, heading: &Option<String>| {
        let text = paragraph.trim();
        if !text.is_empty() {
            paragraphs.push((heading.clone(), text.to_string()));
        }
        paragraph.clear();
    };

    for line in markdown.lines() {
        if let Some(next_heading) = parse_heading(line) {
            flush_paragraph(&mut paragraph, &heading);
            heading = Some(next_heading);
            continue;
        }
        if line.trim().is_empty() {
            flush_paragraph(&mut paragraph, &heading);
        } else {
            if !paragraph.is_empty() {
                paragraph.push('\n');
            }
            paragraph.push_str(line);
        }
    }
    flush_paragraph(&mut paragraph, &heading);

    let mut chunks = Vec::new();
    let mut current_heading: Option<String> = None;
    let mut current = String::new();
    for (paragraph_heading, text) in paragraphs {
        if current_heading != paragraph_heading && !current.trim().is_empty() {
            chunks.push(ChunkText {
                heading: current_heading.take(),
                markdown: current.trim().to_string(),
            });
            current.clear();
        }
        current_heading = paragraph_heading.clone();
        let candidate = if current.is_empty() {
            text.clone()
        } else {
            format!("{current}\n\n{text}")
        };
        if candidate.chars().count() <= max_chars || current.is_empty() {
            current = candidate;
            if current.chars().count() <= max_chars {
                continue;
            }
        }
        chunks.push(ChunkText {
            heading: current_heading.clone(),
            markdown: current.chars().take(max_chars).collect(),
        });
        let tail: String = current
            .chars()
            .rev()
            .take(overlap)
            .collect::<String>()
            .chars()
            .rev()
            .collect();
        current = format!("{tail}\n\n{text}");
        while current.chars().count() > max_chars {
            chunks.push(ChunkText {
                heading: current_heading.clone(),
                markdown: current.chars().take(max_chars).collect(),
            });
            let tail: String = current
                .chars()
                .rev()
                .take(overlap)
                .collect::<String>()
                .chars()
                .rev()
                .collect();
            current = tail;
        }
    }
    if !current.trim().is_empty() {
        chunks.push(ChunkText {
            heading: current_heading,
            markdown: current.trim().to_string(),
        });
    }
    chunks
}

fn parse_heading(line: &str) -> Option<String> {
    let trimmed = line.trim_start();
    let hashes = trimmed
        .chars()
        .take_while(|character| *character == '#')
        .count();
    if (1..=6).contains(&hashes) && trimmed.chars().nth(hashes) == Some(' ') {
        Some(
            trimmed[hashes..]
                .trim()
                .trim_end_matches('#')
                .trim()
                .to_string(),
        )
    } else {
        None
    }
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
            chunk_id: self.chunk_id,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn heading_chunker_preserves_sections_and_overlap() {
        let chunks = chunk_markdown(
            "# One\n\nalpha beta gamma\n\n## Two\n\ndelta epsilon zeta",
            40,
            5,
        );
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].heading.as_deref(), Some("One"));
        assert_eq!(chunks[1].heading.as_deref(), Some("Two"));
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
    fn path_exclusion_detects_trash_and_obsidian() {
        assert!(is_path_excluded(".trash/note.md"));
        assert!(is_path_excluded("folder/.trash/note.md"));
        assert!(is_path_excluded(".obsidian/workspace.json"));
        assert!(!is_path_excluded("10_Projects/note.md"));
        assert!(!is_path_excluded("trash_collection/note.md"));
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
    fn rerank_response_deserialization() {
        let json = r#"{"model":"gpustack/bge-reranker-v2-m3-GGUF","object":"list","results":[{"index":1,"relevance_score":-2.63},{"index":0,"relevance_score":-4.14}]}"#;
        let parsed: RerankResponse = serde_json::from_str(json).unwrap();
        assert_eq!(parsed.results.len(), 2);
        assert_eq!(parsed.results[0].index, 1);
        assert!((parsed.results[0].relevance_score - (-2.63)).abs() < 1e-4);
    }
}
