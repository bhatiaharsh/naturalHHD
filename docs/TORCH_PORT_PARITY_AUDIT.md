# numpy vs. torch parity audit

Scope: line-by-line comparison of `pynhhd-v1.1/pynhhd/` (numpy reference) against
`pynhhd-torch-v1.0/pynhhd/` (torch port) on `feature/pytorch-port`, checking for
structural equivalence (same files/modules/function/variable names) and
functional equivalence (same calculations). Every difference found below is
called out with a justification verdict.

## Structural comparison

| numpy (`pynhhd-v1.1/pynhhd/`) | torch (`pynhhd-torch-v1.0/pynhhd/`) | Verdict |
|---|---|---|
| `nHHD.py` (class `nHHD`) | `nhhd.py` (classes `nHHD`, `DecompositionResult`) | Justified, but worth flagging: filename case-change plus a new module-level export. |
| `greens.py`, `poisson.py`, `structured.py`, `unstructured.py`, `spherical.py` | same names | Match. |
| `utils/timer.py`, `utils/logger.py` | same; `logger.py` is byte-identical | Match. |
| — | `utils/device.py` (new) | Justified: shared dtype/device/batch helpers; numpy has no analogue need. |
| `examples/` (plotting script + drawing/fields utils) | `tests/` (pytest suite) | Justified but a real capability gap: no runnable demo script was ported 1:1; torch substitutes automated tests instead. |
| `setup.py` | `pyproject.toml` | Packaging modernization, immaterial. |

Class/method names line up 1:1 (`StructuredGrid`, `UnstructuredGrid`,
`StructuredSphericalGrid`, `PoissonSolver`, `GreensFunction`, and their
`divcurl`/`gradient`/`rotated_gradient`/`curl3D`/`solve`/`prepare`/`compute`/
`create_grid`/`create_points` methods all match by name). Internal variable
names (`dudx`, `dvdy`, `nRu`, `dxy`, etc.) are also kept consistent.

## Functional differences found

1. **Two silent correctness bugs in numpy, fixed only in the torch port (not
   in the numpy source on this branch).**
   - `structured.py` 3D `divcurl`: numpy computes
     `div = du/dx + dv/dy + dv/dz` (reuses `dv/dz`; `dw/dz` is computed but
     never referenced) — torch correctly uses `dw/dz`.
     (`pynhhd-v1.1/pynhhd/structured.py:90`, `docs/CODE_REVIEW_REPORT.md` §2.1)
   - `unstructured.py` boundary-curl correction: numpy accumulates a stale
     `tn` left over from the interior per-triangle loop instead of the
     freshly-computed `dt` for that boundary edge — torch reuses the
     correctly-indexed per-(face,corner) tensor, eliminating the bug class
     structurally rather than patching one call site.
     (`pynhhd-v1.1/pynhhd/unstructured.py:328-329`, `docs/CODE_REVIEW_REPORT.md` §2.2)
   - **Verdict:** correct fixes, but these are numpy-source bugs, not
     porting decisions. They belong on `fix/pynhhd-numpy-bugs`, not
     `feature/pytorch-port` — see "Numpy bugfix branch" below.

2. **Default precision: float64 → float32.** numpy's `PoissonSolver` sets
   `self.dtype = np.float` (== Python `float` == float64); torch's
   `DEFAULT_DTYPE = torch.float32` (`utils/device.py`). float64 remains
   available via `dtype=torch.float64`, but the two libraries do **not**
   produce numerically identical results by default — they diverge by more
   than floating-point noise, particularly near the Green's-function
   self-term. Justified as a default (GPU throughput) and documented, but
   not opt-out-transparent — always pass `dtype=torch.float64` when
   comparing outputs directly.

3. **Spherical grid: `glob=False` (regional/non-periodic longitude) mode was
   dropped, undocumented.** numpy's `spherical.py` `divcurl`/`gradient`/
   `rotated_gradient` each take a `glob` parameter; `glob=False` falls back
   to one-sided derivatives at the longitude edges for a non-global,
   regional domain. Torch's `StructuredSphericalGrid` methods have **no**
   `glob` parameter — they always behave as `glob=True`. The torch
   `spherical.py` module docstring explains the periodic-wraparound port but
   never mentions the regional mode existed and was removed.
   **Verdict: not clearly justified.** A real, if narrow, feature loss.
   `nHHD` never exposed `glob` as a pass-through itself, so impact is low,
   but it's an undocumented silent capability drop, unlike every other
   difference in this list. Either port `glob=False` or add a line noting
   regional (non-global) spherical grids aren't supported.

4. **Spherical solver: O((ny·nx)²) joblib loop → FFT/einsum
   reformulation.** Torch replaces the per-latitude, per-point Python loop
   with a precomputed `(ny, ny, nx)` kernel, FFT'd once, contracted via
   `einsum` per solve. The torch docstring derives (via a
   correlation-vs-convolution argument) why this must be implemented as a
   circular *correlation*, not a naive convolution, to match numpy's
   semantics even for non-global (<360°) longitude grids. **Justified**, and
   unusually well-documented for a nontrivial reformulation.

5. **`decompose()` return shape.** numpy mutates `self.div`/`self.curlw`/
   `self.d`/`self.r`/`self.h` in place and returns nothing. Torch returns an
   explicit `DecompositionResult` namedtuple (while still setting the same
   `self.*` attributes for the unbatched case, as a compatibility shim).
   **Justified** — required once a batch dimension exists (`self.h` would be
   ambiguous for a batch of B fields) and backward-compatible for
   single-field callers.

6. **`PoissonSolver.prepare()` timing.** numpy prepares lazily/optionally
   and has a live bug where the lazy self-heal path in `_solve()` calls
   `self.prepare(num_cores)` with an undefined `num_cores`
   (`pynhhd-v1.1/pynhhd/poisson.py:349`, `docs/CODE_REVIEW_REPORT.md` §2.4).
   Torch makes `prepare()` mandatory, constructor-time work; `solve()`
   raises a clear `RuntimeError` if called first. **Justified fix**,
   explicitly documented in torch's `poisson.py` module docstring.

7. **`nHHD.__init__` argument validation.** numpy's "grid without spacings"
   check ANDs two independent mismatch tests together, so it never actually
   fires (`pynhhd-v1.1/pynhhd/nHHD.py:43-44`, `docs/CODE_REVIEW_REPORT.md`
   §2.3 — dead validation code). Torch's `sum(modes) != 1` / explicit
   `spacings is None` checks are simply correct. **Justified fix**, though
   not explicitly cross-referenced to §2.3 anywhere in torch's comments (a
   minor documentation gap, not a functional problem).

8. **Unstructured/spherical 3D: explicit `NotImplementedError` vs. numpy's
   silent stub/never-finished path.** numpy's `unstructured.py` 3D branch
   references an undefined `self.sdim` (would `NameError` if ever reached)
   and `spherical.py`'s 3D branch is an explicit
   "NOT IMPLEMENTED — WILL RETURN THE SAME AS CARTESIAN DIV" print-and-
   fallback stub. Torch raises `NotImplementedError` up front in both
   cases. **Justified** — strictly safer, and matches the numpy version's
   *actual* (broken) capability rather than its aspirational one.

## Numpy bugfix branch

Items 1, 6, 7, and 8 above describe bugs/dead-code in `pynhhd-v1.1` itself,
not porting decisions. `feature/pytorch-port`'s copy of `pynhhd-v1.1/` is
untouched from the `7a8efd4` ("cleaned spherical") base — none of these
fixes have leaked into this branch's numpy source.

A `fix/pynhhd-numpy-bugs` branch already exists (checked out as a worktree
at `.claude/worktrees/fix-pynhhd-numpy-bugs/`), branched from the same
`7a8efd4` base, and already carries exactly this set of fixes as three
commits:

- `67a7f4d` — Fix silent correctness bugs in 3D divergence and unstructured
  boundary curl. Also fix numpy>=1.24 compatibility (`np.float`/`np.int`)
  and dead input validation. (covers items 1, 7, and the numpy>=1.24 /
  `gsw` import issues from `docs/CODE_REVIEW_REPORT.md` §2.5-2.6)
- `012c611` — Make 3D spherical grids fail with `NotImplementedError`
  instead of silently computing wrong values.
- `a583526` — Raise `NotImplementedError` (not `ValueError`) for
  unimplemented 3D unstructured meshes, matching the 3D-spherical treatment.

So there is nothing to move: `feature/pytorch-port` never picked up any
numpy-side fix, and `fix/pynhhd-numpy-bugs` already has the complete set.
This branch can continue to focus solely on the torch port.
