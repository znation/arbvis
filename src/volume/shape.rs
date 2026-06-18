//! 3D placement seam: the volume analog of [`crate::layout`]'s `LayoutShape` /
//! `LayoutPlugin` / `select_layout`.
//!
//! Where a [`crate::LayoutShape`] maps the concatenated byte stream onto a 2D
//! canvas, a [`VolumeShape`] maps it into a bounded `grid_side³` voxel cube. The
//! byte-Hilbert floor ([`HilbertVolume`], priority `i32::MIN`) reproduces
//! today's blind whole-stream Hilbert fill, so byte inputs render identically
//! to before the seam existed. A downstream specialization registers a
//! higher-priority [`VolumeShapePlugin`] that returns a list of placed
//! [`VolumeEntity`]s (e.g. modelweightvis stacking transformer blocks along Z).
//!
//! Selection mirrors [`crate::layout::select_layout`] exactly: descending
//! priority, the `i32::MIN` floor guarantees termination, and a `Forced(id)`
//! that doesn't win logs the same diagnostic. A downstream plugin gates on
//! [`crate::LayoutMode`] in its `applicable` (e.g. bow out under
//! [`LayoutMode::Hilbert`]) just like the 2D arch layout does.

use std::any::Any;
use std::sync::Arc;

use serde::Serialize;

use crate::data::Source;
use crate::layout::LayoutMode;
use crate::registry::{LayoutBuildCtx, Registry, VolumeShapePlugin};

/// Axis-aligned voxel box `[min, max)` in grid coordinates (each axis in
/// `0..grid_side`). The upper bounds are exclusive.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct VoxelBox {
    pub x0: u32,
    pub y0: u32,
    pub z0: u32,
    pub x1: u32,
    pub y1: u32,
    pub z1: u32,
}

/// One entry in the viewer's click-to-pick manifest (written to `meta.json`):
/// a human-facing label + cube box for an entity. Lets the viewer name what the
/// user clicked without arbvis ever interpreting the opaque
/// [`VolumeEntity::extra`].
#[derive(Clone, Debug, Serialize)]
pub struct VolumeLabel {
    /// Display name (e.g. the tensor name).
    pub name: String,
    /// Coarse grouping shown alongside the name (e.g. `"layer 3"`, `"top-level"`).
    pub group: String,
    /// Box in grid voxels — same coordinate space as [`VolumeEntity::bbox`].
    pub bbox: VoxelBox,
}

/// One placed entity in the cube. arbvis does not interpret `extra` or
/// `renderer_id`; it fetches `[byte_start, byte_start + byte_len)` from
/// `sources[source_idx]` and hands the bytes to the [`VoxelRenderer`] named by
/// `renderer_id`, exactly as the 2D path routes a `LeafTile` to a `LeafRenderer`.
pub struct VolumeEntity {
    /// Index into the run's `sources` list.
    pub source_idx: usize,
    /// Byte offset *within that source* (not the concatenated stream).
    pub byte_start: u64,
    pub byte_len: u64,
    /// Target box in the bounded grid.
    pub bbox: VoxelBox,
    /// Names the [`VoxelRenderer`] that decodes + colors this entity.
    pub renderer_id: &'static str,
    /// Opaque per-entity payload the renderer downcasts (dtype, element shape,
    /// colormap choice, diff partner span, …).
    pub extra: Box<dyn Any + Send + Sync>,
}

/// 3D analog of [`crate::LayoutShape`]: how the byte stream maps into a bounded
/// voxel cube.
pub trait VolumeShape: Send + Sync {
    /// Stable id; also the default [`VoxelRenderer`] id for entities that don't
    /// override it. Mirrors [`crate::LayoutShape::id`].
    fn id(&self) -> &'static str;

    /// The cube side this shape renders into (a power of two).
    fn grid_side(&self) -> u32;

    /// `true` ⇒ arbvis ignores [`entities`](VolumeShape::entities) and runs the
    /// legacy whole-stream byte→Hilbert fill (the `i32::MIN` floor). The 3D
    /// analog of `LayoutShape::is_byte_layout`.
    fn is_byte_volume(&self) -> bool {
        false
    }

    /// The placed entities. `None`/empty for the byte floor.
    fn entities(&self) -> Option<Vec<VolumeEntity>> {
        None
    }

    /// Optional camera framing override (cube-space center + radius). When
    /// `None`, arbvis frames from grid occupancy.
    fn focus(&self) -> Option<([f32; 3], f32)> {
        None
    }

    /// Per-entity labels for the viewer's click-to-pick manifest, written
    /// verbatim to `meta.json`. Empty for the byte floor (nothing pickable).
    fn manifest(&self) -> Vec<VolumeLabel> {
        Vec::new()
    }

    fn as_any(&self) -> &dyn Any;
}

/// The byte-Hilbert floor — reproduces today's blind whole-stream aggregation.
pub struct HilbertVolume {
    side: u32,
}

impl VolumeShape for HilbertVolume {
    fn id(&self) -> &'static str {
        "hilbert-bytes"
    }
    fn grid_side(&self) -> u32 {
        self.side
    }
    fn is_byte_volume(&self) -> bool {
        true
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Floor plugin (`i32::MIN`), always applicable — guarantees selection
/// terminates, mirroring `HilbertLayoutPlugin`.
pub struct HilbertVolumePlugin;

impl VolumeShapePlugin for HilbertVolumePlugin {
    fn id(&self) -> &'static str {
        "hilbert-bytes"
    }
    fn priority(&self) -> i32 {
        i32::MIN
    }
    fn applicable(&self, _ctx: &LayoutBuildCtx<'_>) -> bool {
        true
    }
    fn build(&self, ctx: &LayoutBuildCtx<'_>) -> Option<Box<dyn VolumeShape>> {
        Some(Box::new(HilbertVolume {
            side: ctx.grid_side,
        }))
    }
}

/// Pick the highest-priority applicable [`VolumeShapePlugin`]. Mirrors
/// [`crate::layout::select_layout`]: descending priority, `i32::MIN` floor
/// guarantees a result, and a non-winning `Forced(id)` warns + falls back —
/// or, under [`Registry::strict_layout`], hard-errors instead.
pub fn select_volume_shape(
    sources: &[Source],
    cumulative_offsets: &[u64],
    total_bytes: u64,
    mode: LayoutMode,
    diff_mode: bool,
    grid_side: u32,
    registry: &Registry,
) -> anyhow::Result<Box<dyn VolumeShape>> {
    let ctx = LayoutBuildCtx {
        sources,
        cumulative_offsets,
        total_bytes,
        mode,
        diff_mode,
        grid_side,
    };

    let mut sorted: Vec<&Arc<dyn VolumeShapePlugin>> = registry.volume_shapes.iter().collect();
    sorted.sort_by_key(|p| std::cmp::Reverse(p.priority()));

    let forced_id = match mode {
        LayoutMode::Forced(id) => Some(id),
        _ => None,
    };
    let mut chosen: Option<Box<dyn VolumeShape>> = None;
    let mut forced_was_applicable = false;
    for plugin in &sorted {
        if !plugin.applicable(&ctx) {
            continue;
        }
        let pid = plugin.id();
        match plugin.build(&ctx) {
            Some(shape) => {
                chosen = Some(shape);
                break;
            }
            None => {
                if Some(pid) == forced_id {
                    forced_was_applicable = true;
                }
            }
        }
    }
    let chosen = chosen.expect(
        "registry.volume_shapes must include a floor plugin (HilbertVolumePlugin at i32::MIN)",
    );

    if let Some(id) = forced_id {
        if chosen.id() != id {
            // Mirrors `select_layout`'s strict handling: applicable-but-couldn't
            // -build vs. never-applicable differ only in the reason. Under
            // strict mode each is a hard error rather than a silent fallback.
            let reason = if forced_was_applicable {
                "it could not build a 3D volume for these inputs"
            } else {
                "no registered 3D volume layout matched"
            };
            if registry.strict_layout {
                anyhow::bail!(
                    "layout `{id}` requested but {reason}; strict layout mode refuses to \
                     fall back to `{}`",
                    chosen.id()
                );
            }
            log::warn!(
                "layout `{id}` requested but {reason}; falling back to `{}`",
                chosen.id()
            );
        }
    }

    Ok(chosen)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The 3D analog of `select_layout`'s strict test: a forced id no volume
    /// plugin claims falls back to the Hilbert floor — a warn (`Ok`) normally,
    /// a hard error under strict mode.
    #[test]
    fn strict_layout_turns_forced_volume_fallback_into_error() {
        let mut registry = Registry::with_defaults();
        registry.layout_mode = LayoutMode::Forced("nonexistent");

        let relaxed = select_volume_shape(&[], &[], 0, registry.layout_mode, false, 64, &registry)
            .expect("non-strict select_volume_shape must fall back, not error");
        assert_eq!(relaxed.id(), "hilbert-bytes");

        registry.strict_layout = true;
        assert!(
            select_volume_shape(&[], &[], 0, registry.layout_mode, false, 64, &registry).is_err(),
            "strict select_volume_shape must error when a forced layout falls back"
        );
    }
}
