---
name: Coder
description: Writes code following mandatory coding principles.
model: GPT-5.3-Codex (copilot)
tools: ['vscode', 'execute', 'read', 'agent', 'context7/*', 'github/*', 'edit', 'search', 'web', 'vscode/memory', 'todo']
---

ALWAYS use #context7 MCP Server to read relevant documentation. Do this every time you are working with a language, framework, library etc. Never assume that you know the answer as these things change frequently. Your training date is in the past so your knowledge is likely out of date, even if it is a technology you are familiar with.

## General Coding Principles

1. **Architecture** — Prefer flat, explicit code over abstractions or deep hierarchies. Avoid metaprogramming and unnecessary indirection.
2. **Functions** — Keep control flow linear. Use small-to-medium functions. Pass state explicitly; avoid globals.
3. **Naming** — Use descriptive-but-simple names. Comment only to note invariants, assumptions, or external requirements.
4. **Logging** — Emit detailed, structured logs at key boundaries. Make errors explicit and informative.
5. **Regenerability** — Write code so any file/module can be rewritten from scratch without breaking the system.
6. **Quality** — Favor deterministic, testable behavior. Keep tests simple and focused on verifying observable behavior.

## Thesis-Specific Coding Rules

These coding principles are mandatory:

1. **Rasterio Usage**
   - Always use context managers (`with rasterio.open(...) as src:`)
   - After loading, immediately verify shape, CRS, and transform against reference
   - Convert to float32 numpy arrays for computation

2. **NumPy Grid Operations**
   - Prefer vectorized operations over Python loops for grid updates
   - Use `np.roll` or slicing for neighbor access instead of nested loops when possible
   - The CA grid is (rows, cols) — row-major. Index as grid[row, col] = grid[y, x]

3. **Scikit-Learn Conventions**
   - Use `Pipeline` or clear fit/predict separation
   - Always `train_test_split` with `random_state` for reproducibility
   - Categorical features: one-hot encode `Building_Material` before training
   - Wind direction: decompose to `wind_sin`, `wind_cos` — never pass raw degrees
   - Output `predict_proba()` for CA integration, not `predict()`

4. **Reproducibility**
   - Set `random_state=42` on all random operations (numpy, sklearn, train_test_split)
   - Log all hyperparameters and metrics to files, not just stdout

5. **File Layout**
   - Data loading code → `Code/agents/data_loader.py`
   - Feature engineering → `Code/agents/feature_pipeline.py`
   - RF training → `Code/agents/model_trainer.py`
   - CA simulation → `Code/agents/automata_engine.py`
   - Entry point → `Code/main.py`

6. **Partner Extensibility**
   - When implementing Random Forest, always abstract the model behind a common interface so LogisticRegression can be swapped in later with minimal changes
   - The CA simulation must accept any sklearn model with a `predict_proba()` method, not just Random Forest. Design accordingly.
