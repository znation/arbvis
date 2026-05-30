//! Tensor-format-aware visualization built on `arbvis`.
//!
//! Today this crate is a thin shell over arbvis: its binary calls
//! `arbvis::run` with the default registry. Step 12d moves the tensor-format
//! parsers, architectural layout, MoE-diff layout, tensor-diff source builder,
//! and dtype-aware element colorizers out of arbvis into here, and exposes a
//! `register_all(&mut Registry)` that the binary calls before dispatch.
