//! [`arbvis::FormatPlugin`] impls — one per supported model file format.
//!
//! Each plugin claims its file extension (`detects_path`), reads the
//! header (sync from a `Path` in `populate_local`, async from a `Data`
//! handle in `populate_remote`), and stuffs a [`ModelInfo`] into the
//! source's [`arbvis::Extensions`] map. Downstream the architectural
//! layout / arch tile loader / renderer read it back via
//! `extensions.get::<ModelInfo>()`.
//!
//! Failures are non-fatal: arbvis's `prepare_sources` logs the plugin
//! error and continues with no `ModelInfo` populated — the file then
//! falls through to the byte-Hilbert path the same way an `.iso` would.

use arbvis::{Data, Extensions, FormatPlugin};
use futures::future::BoxFuture;
use std::path::Path;

use crate::data::{load_model_info, load_model_info_async};
use crate::format::{ModelInfo, SourceFormat};

/// `.safetensors` header parser — produces a [`ModelInfo`] with one
/// [`crate::format::TensorMeta`] per tensor entry.
pub struct SafetensorsFormatPlugin;

impl FormatPlugin for SafetensorsFormatPlugin {
    fn id(&self) -> &'static str {
        "safetensors"
    }
    fn detects_path(&self, path: &Path) -> bool {
        matches!(
            SourceFormat::from_path(path),
            Some(SourceFormat::Safetensors)
        )
    }
    fn populate_local(
        &self,
        path: &Path,
        file_size: u64,
        exts: &mut Extensions,
    ) -> anyhow::Result<()> {
        let info: ModelInfo = load_model_info(path, file_size, SourceFormat::Safetensors)?;
        exts.insert(info);
        Ok(())
    }
    fn populate_remote<'a>(
        &'a self,
        data: &'a Data,
        byte_size: u64,
        exts: &'a mut Extensions,
    ) -> BoxFuture<'a, anyhow::Result<()>> {
        Box::pin(async move {
            let info = load_model_info_async(data, byte_size, SourceFormat::Safetensors).await?;
            exts.insert(info);
            Ok(())
        })
    }
}

/// `.gguf` header parser — produces a [`ModelInfo`] with one
/// [`crate::format::TensorMeta`] per tensor and pre-built color ranges
/// for the metadata + tensor regions.
pub struct GgufFormatPlugin;

impl FormatPlugin for GgufFormatPlugin {
    fn id(&self) -> &'static str {
        "gguf"
    }
    fn detects_path(&self, path: &Path) -> bool {
        matches!(SourceFormat::from_path(path), Some(SourceFormat::Gguf))
    }
    fn populate_local(
        &self,
        path: &Path,
        file_size: u64,
        exts: &mut Extensions,
    ) -> anyhow::Result<()> {
        let info = load_model_info(path, file_size, SourceFormat::Gguf)?;
        exts.insert(info);
        Ok(())
    }
    fn populate_remote<'a>(
        &'a self,
        data: &'a Data,
        byte_size: u64,
        exts: &'a mut Extensions,
    ) -> BoxFuture<'a, anyhow::Result<()>> {
        Box::pin(async move {
            let info = load_model_info_async(data, byte_size, SourceFormat::Gguf).await?;
            exts.insert(info);
            Ok(())
        })
    }
}

/// PyTorch pickle (`.bin` / `.pth` / `.pt`) header parser.
///
/// Local: full `candle_core::pickle` parse of the zip-packed opcode stream.
/// Remote: errors — pickle's zip end-of-central-directory record lives at
/// the END of the file, so a head-prefix range fetch can't parse it. The
/// caller treats remote pickle as plain bytes; downloading first
/// re-enables the local path.
pub struct PickleFormatPlugin;

impl FormatPlugin for PickleFormatPlugin {
    fn id(&self) -> &'static str {
        "pickle"
    }
    fn detects_path(&self, path: &Path) -> bool {
        matches!(SourceFormat::from_path(path), Some(SourceFormat::Pickle))
    }
    fn populate_local(
        &self,
        path: &Path,
        file_size: u64,
        exts: &mut Extensions,
    ) -> anyhow::Result<()> {
        let info = load_model_info(path, file_size, SourceFormat::Pickle)?;
        exts.insert(info);
        Ok(())
    }
    fn populate_remote<'a>(
        &'a self,
        _data: &'a Data,
        _byte_size: u64,
        _exts: &'a mut Extensions,
    ) -> BoxFuture<'a, anyhow::Result<()>> {
        Box::pin(async move {
            anyhow::bail!("pickle: remote header fetch not yet supported — download the file first")
        })
    }
}
