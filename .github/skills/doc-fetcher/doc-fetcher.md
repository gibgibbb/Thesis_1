---
name: doc-fetcher
description: Mandates the retrieval of up-to-date API documentation before writing complex spatial or mathematical logic.
---

# Skill: Documentation Fetcher

As an AI, your training data may contain outdated API signatures. To prevent hallucinating incorrect syntax for this thesis, you must utilize this skill before writing core simulation code.

## Core Directives
Whenever tasked with implementing logic using `scipy.ndimage`, `rasterio`, or `scikit-learn`, you MUST use your available tools (`#context7`, `#web`, or Copilot Search) to look up the current official documentation.

## Required Checks:
1. **Convolutions:** Before writing CA neighborhood logic, verify the syntax for `scipy.ndimage.convolve` or `scipy.signal.convolve2d`, specifically checking how `mode` and `cval` parameters handle matrix edges (the borders of Lapu-Lapu City).
2. **Raster I/O:** Before writing save/load functions, verify `rasterio.open` profile updating techniques for memory-efficient dtypes (`int8`, `float32`).
3. **Scikit-Learn:** Verify the `.predict_proba()` output shape for binary classification to ensure it aligns perfectly with the CA probability matrices.

**Do not guess API parameters. Fetch them.**
