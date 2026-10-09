//! Shared test-only format-plugin mocks, used by the `registry` and `data`
//! test suites to exercise first-match-wins and failure-fallback behavior.

use crate::data::{Data, Extensions};
use crate::registry::FormatPlugin;
use futures::future::BoxFuture;
use std::path::Path;

/// Marker type a fake format plugin stuffs into `Extensions` so tests can
/// observe which plugin ran.
pub(crate) struct Tag(pub String);

/// Minimal format plugin: matches by extension suffix, optionally fails in
/// `populate_local`, and tags the extensions map so tests can verify
/// first-plugin-wins and failure-fallback behaviour.
pub(crate) struct FakePlugin {
    pub ext: &'static str,
    pub tag: &'static str,
    pub fail: bool,
}

impl FormatPlugin for FakePlugin {
    fn id(&self) -> &'static str {
        "fake"
    }
    fn detects_path(&self, path: &Path) -> bool {
        path.extension().and_then(|e| e.to_str()) == Some(self.ext)
    }
    fn populate_local(
        &self,
        _path: &Path,
        _file_size: u64,
        exts: &mut Extensions,
    ) -> anyhow::Result<()> {
        if self.fail {
            anyhow::bail!("fake plugin parse failure");
        }
        exts.insert(Tag(self.tag.to_string()));
        Ok(())
    }
    fn populate_remote<'a>(
        &'a self,
        _data: &'a Data,
        _byte_size: u64,
        _exts: &'a mut Extensions,
    ) -> BoxFuture<'a, anyhow::Result<()>> {
        Box::pin(async { Ok(()) })
    }
}
