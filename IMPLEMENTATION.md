# NNEditor Implementation Plan

**Source:** `I-want-to-build-a-NN-editor-th-results.md`
**Plan created:** 2026-07-28
**Last updated:** 2026-07-29
**Implementation state:** In progress
**Product progress:** 67 of 76 tasks complete (88%)

## 1. Goal

Build an ONNX-first neural-network viewer and editor with:

- Semantic pan, zoom, search, minimap, and hierarchical collapse.
- Explainable automatic block detection and persistent manual overrides.
- Lazy graph, metadata, statistics, and tensor loading for large models.
- A Flet desktop and web shell with a replaceable graph renderer.
- Transactional, validated model editing with undo/redo.
- Capability-aware import and export for ONNX, PyTorch exported artifacts,
  and JAX/StableHLO artifacts.
- Consent-gated inference tracing with per-node activation inspection and
  intuitive tensor visualization.
- Safe handling of untrusted model files and explicit isolation for model code.

The first release is a read-only ONNX viewer. Editing, quantization, pruning,
PyTorch, and JAX support follow only after the foundational risks have been
validated.

## 2. Scope and Guardrails

### In scope for the first usable release

- ONNX models with embedded or external tensor data.
- Safe, non-executable artifact loading.
- Architecture, block, layer, and operator levels of detail.
- Lazy tensor slices and derived statistics.
- Desktop and web deployment.
- Session persistence and cached graph layouts.

### Explicitly deferred

- Loading arbitrary Python model code without an isolated trusted-code flow.
- Reconstructing original PyTorch or JAX source code.
- Universal or lossless cross-format export.
- Arbitrary shape-changing graph surgery.
- Claims that pruning necessarily reduces storage or improves runtime.
- Learned block detection before sufficient correction data exists.

### Architectural rules

1. Keep original artifacts immutable; exports always create a new artifact.
2. Treat format support as an artifact-and-capability matrix, not a file
   extension list.
3. Keep topology, layout, metadata, statistics, and tensor payloads separate.
4. Never materialize tensor values during initial model import unless required
   to parse the source artifact.
5. Hide graph rendering and graph algorithms behind replaceable interfaces.
6. Preserve source-specific metadata in namespaced IR extensions.
7. Make every edit a reversible command applied to a working revision.
8. Report export fidelity as lossless, semantically equivalent, lossy,
   weights-only, or unavailable.
9. Default to safe artifact mode; executable tracing requires explicit trust
   and isolation.

## 3. Progress Tracking

### Status conventions

- An unchecked task is pending.
- Append `— IN PROGRESS (owner, YYYY-MM-DD)` to an active task.
- Append `— BLOCKED: reason` when work cannot proceed.
- Check a task only after its acceptance criteria pass and evidence is linked
  in the progress log.
- A phase is complete only when all tasks and its exit gate are complete.
- Update the counts and percentage in this document in the same change that
  completes a task.

### Phase summary

| Phase | Status | Complete | Depends on | Exit gate |
|:--|:--|--:|:--|:--|
| 0. Feasibility and foundations | Complete | 7/7 | None | Architecture decision record approved |
| 1. Read-only ONNX vertical slice | Complete | 10/10 | Phase 0 | Large external-data model opens without eager tensor loading |
| 2. Hierarchy and semantic zoom | Complete | 8/8 | Phase 1 | CNN and transformer fixtures collapse interactively |
| 3. Tensor inspection and revisions | Complete | 8/8 | Phase 1 | Lazy inspection and reversible copy-on-write edit work |
| 4. ONNX surgery and export | Complete | 8/8 | Phases 2-3 | Golden round trips and validation pass |
| 5. Quantization and pruning | Complete | 7/7 | Phase 4 | Supported edits export with complete scheme reports |
| 6. PyTorch exported artifacts | Complete | 6/6 | Phase 4 | Imports and exports match the capability matrix |
| 7. JAX and StableHLO artifacts | In progress | 5/6 | Phase 4 | Supported artifacts import with documented fidelity |
| 8. Web hardening and plugins | Not started | 0/8 | Phases 2-7 as applicable | Isolation, quotas, remote storage, and extension APIs pass |
| 9. Inference tracing and activation inspection | Complete (desktop; web gated on P8.3) | 8/8 | Phases 3-4; P8.3 for web mode | Approved inputs trace in isolation and every captured activation is inspectable |
| **Total** | **In progress** | **67/76** |  |  |

### Progress log

| Date | Change | Evidence | Progress |
|:--|:--|:--|--:|
| 2026-07-28 | Read the source analysis and created the implementation baseline. No product code exists yet. | `IMPLEMENTATION.md` | 0/68 |
| 2026-07-28 | Began P0.1 project bootstrap and compatibility verification. | `pyproject.toml`, `docs/platform-support.md` | 0/68 |
| 2026-07-28 | Completed P0.1: bootstrapped the Python/Flet package, pinned and locked Flet 0.86.4, recorded Python 3.12-3.14 and desktop/browser targets, added CI and local quality gates, and built both distributions. Formatting, linting, strict typing, and 4 tests with 100% coverage pass; the tests pass on Python 3.12.13, 3.13.13, and 3.14.6. | `pyproject.toml`, `uv.lock`, `.github/workflows/ci.yml`, `docs/platform-support.md`, `src/`, `tests/`; `uv run ruff format --check .`; `uv run ruff check .`; `uv run mypy`; `uv run pytest`; `uv build` | 1/68 |
| 2026-07-28 | Initialized the Git repository on `main` with repository-local author and committer identity `Imed Bouazizi <ibouazizi@gmail.com>`. | Initial Git commit | 1/68 |
| 2026-07-28 | Completed P0.2: defined eight artifact contracts answering all six capabilities with a mandatory reason each, plus availability values, loading modes, and export-fidelity ceilings. The registry is executable and generates the document, so the two cannot drift. | `src/nneditor/ir/capabilities.py`, `docs/artifact-capabilities.md`, `tools/render_docs.py`, `tests/unit/test_capabilities.py` | 2/68 |
| 2026-07-28 | Completed P0.3: specified documents, graphs, nodes, ports, values, symbolic shapes, tensor references, subgraphs, groups, provenance, revisions, and extension namespaces; implemented and tested the `major.minor` read-planning rules and the stable-ID derivation rules including the named/output-derived/positional stability ladder. | `docs/ir-schema.md`, `src/nneditor/ir/schema.py`, `src/nneditor/ir/identity.py`, `tests/unit/test_schema.py`, `tests/unit/test_identity.py` | 3/68 |
| 2026-07-28 | Completed P0.4: built the ONNX lazy indexer on a hand-written protobuf wire reader, so import records tensor *locations* and never materializes a tensor. A 2 MiB-weight model indexes while reading under 1% of the file logically and under 10% physically, and no logical read overlaps a payload. External references are confined to an approved root, bounds-checked against the real file size, cross-checked against the shape-implied length, and checksum-verified only on request. Verified differentially against `onnx` 1.22 on five fixture shapes. | `src/nneditor/adapters/onnx/`, `src/nneditor/storage/reader.py`, `src/nneditor/storage/paths.py`, `src/nneditor/diagnostics.py`, `tests/integration/test_onnx_indexer.py`, `tests/integration/test_onnx_indexer_edge_cases.py`, `tests/unit/test_wire.py`, `tests/unit/test_reader.py`, `tests/unit/test_paths.py`; 238 tests, 99% coverage on Python 3.12.13, 3.13.13, and 3.14.6 | 4/68 |
| 2026-07-28 | Fixed a Windows portability defect in the P0.4 artifact reader: `os.pread` does not exist on Windows and the file was opened without `O_BINARY`, failing 66 tests on the supported Windows desktop target. Added a portable positional-read fallback; the full suite now passes on Windows 11. | `src/nneditor/storage/reader.py`; `uv run pytest` (238 tests, 2 platform-gated skips) | 4/68 |
| 2026-07-28 | Completed P0.5: defined the renderer-agnostic scene model, `GraphRenderer` contract, uniform-grid culling/hit-test index, and deterministic synthetic 1k/10k-node graphs; implemented the Flet Canvas adapter and a browser-side HTML canvas port; measured all three candidate approaches on Windows desktop and web. Key findings: naive Flet shape-list replacement misses 30 FPS by 1-2 orders of magnitude on both modes; shape mutation plus GPU-offset panning passes desktop (18/25 ms p50/p95) and is borderline on web (23/34 ms); the hosted HTML canvas renderer achieves 219 FPS at 10,000 nodes but Flet's WebView gap on Windows/Linux blocks it for desktop. Hit tests are microseconds; collapse/expand p95 is 73 ms at 10k nodes. | `src/nneditor/rendering/`, `benchmarks/`, `benchmarks/results/*.json`, `docs/renderer-benchmark.md`, `tests/unit/test_scene.py`, `tests/unit/test_spatial.py`, `tests/unit/test_synthetic.py`, `tests/unit/test_flet_canvas.py`; 293 tests, 99% coverage; ruff and strict mypy clean | 5/68 |
| 2026-07-28 | Completed P0.6: prototyped copy-on-write revisions. A `WorkingRevision` overlays same-length byte-span edits on read-only base tensors with deterministic undo/redo; a one-float edit against a 4 KiB tensor costs an 8-byte delta. Export splices edited spans into a byte-for-byte copy of the source (embedded raw and external storage), verifies recorded before-bytes against the actual artifact, writes atomically with rollback, and never touches source files. Exports reopen through both the lazy indexer and `onnx.checker`; a no-op export is byte-identical to its source. | `src/nneditor/editing/cow.py`, `src/nneditor/adapters/onnx/splice.py`, `tests/unit/test_cow.py`, `tests/integration/test_cow_export.py`; 319 tests, 98% coverage; ruff and strict mypy clean | 6/68 |
| 2026-07-28 | Drafted the P0.7 architecture decision record from the measured spike results: Flet Canvas under a mandatory mutate/offset/LOD update regime, hosted web canvas as the web-mode escalation path, binding performance budgets, rejected alternatives, and the security boundary. Status is Proposed; Phase 0's exit gate closes on approval. | `docs/adr/0001-phase0-architecture.md` | 6/68 |
| 2026-07-28 | ADR 0001 approved by Imed Bouazizi; P0.7 and Phase 0 complete. Phase 1 begins. | `docs/adr/0001-phase0-architecture.md` (Status: Accepted) | 7/68 |
| 2026-07-28 | Completed P1.1: implemented the typed IR core per `docs/ir-schema.md` — validated immutable documents, graphs (with derived producer/consumer links and single-producer enforcement), nodes with positional ports and typed attributes, values with symbolic shapes, location-only tensor references, provenance without timestamps, and per-entity capability notes — plus canonical JSON serialization whose bytes are order-independent and round-trip exactly; degraded reads park unknown fields under `x-nneditor.unknown` and newer majors are refused. | `src/nneditor/ir/core.py`, `src/nneditor/ir/serialize.py`, `tests/unit/test_ir_core.py`, `tests/unit/test_ir_serialize.py`; 362 tests, 97% coverage | 8/68 |
| 2026-07-28 | Completed P1.2: the indexer now captures typed attribute values (scalars, packed and unpacked lists, strings with lenient decoding; sparse/type-proto/function-reference attributes are diagnosed as unsupported), and `index_to_document` converts a `ModelIndex` into a validated IR document — synthesizing values at node ports, keeping omitted optional ports positional via placeholder values, attaching control-flow subgraphs and function bodies as graphs, carrying model metadata and function signatures in `x-onnx.*` extensions, and narrowing weights/editing capabilities with per-entity notes. Conversion is deterministic: converting the same artifact twice yields identical document bytes. | `src/nneditor/adapters/onnx/to_ir.py`, `src/nneditor/adapters/onnx/indexer.py`, `src/nneditor/adapters/onnx/index.py`, `tests/integration/test_onnx_to_ir.py`; 380 tests, 96% coverage | 9/68 |
| 2026-07-28 | Completed P1.3: registered all 20 diagnostic codes in an executable registry with category, title, and user guidance (`describe()` falls back visibly for unregistered codes); added `onnx.custom-domain` and `onnx.incomplete-io-shape` conversion diagnostics; `docs/import-diagnostics.md` is generated from the registry, and drift-catcher tests fail if a code is emitted without registration or registered without being emitted. | `src/nneditor/diagnostics.py`, `src/nneditor/adapters/onnx/to_ir.py`, `docs/import-diagnostics.md`, `tools/render_docs.py`, `tests/unit/test_diagnostic_registry.py`; 389 tests, 96% coverage | 10/68 |
| 2026-07-28 | Completed P1.4: added the cooperative `CancellationToken` and the `TensorStore` — the only component that turns tensor references into bytes. Metadata access never opens a file; reads are payload-relative, bounds-checked, chunked with cancellation checkpoints, cached under a byte budget with observable hit/miss/eviction counters, and external locations resolve strictly inside the approved root. Closing the store releases every file handle. | `src/nneditor/cancellation.py`, `src/nneditor/storage/store.py`, `tests/unit/test_tensor_store.py`; 405 tests, 96% coverage | 11/68 |
| 2026-07-28 | Completed P1.6 (core): deterministic layered layout (longest-path layering, barycenter ordering, centered grid placement; cycles in malformed artifacts are appended and reported instead of hanging) producing renderer-agnostic scenes, plus the `GraphSlicer` with bounded layout caching keyed by content hash, graph id, and settings, search across every graph, and hop-bounded neighborhood queries. Root-group/detail-level slicing arrives with Phase 2 hierarchy. | `src/nneditor/analysis/layout.py`, `src/nneditor/application/slices.py`, `tests/unit/test_layout_slices.py` | 12/68 |
| 2026-07-28 | Completed P1.5: thread-pool `JobManager` with observable job states, cooperative cancellation (including jobs cancelled before they start), and listener callbacks; `ModelSession` (document + tensor store + slice/search/neighborhood queries) and `ApplicationService` (open sync/async, session registry, shared layout cache, orderly shutdown). Two sessions on the same artifact share cached layouts by content hash; a failed async open leaves no session behind. | `src/nneditor/application/jobs.py`, `src/nneditor/application/session.py`, `tests/unit/test_jobs.py`, `tests/integration/test_session.py`; 432 tests, 96% coverage | 13/68 |
| 2026-07-28 | Completed P1.7: extracted the shared culled shape builder (with the ADR's LOD cap: labels drop first, then edges, nodes never) and implemented `ManagedCanvasRenderer`, the production adapter under the mandatory update regime — pans inside the culled region mutate persistent shape objects in place (the same list object survives, verified by tests), while rebuilds are confined to cull-boundary crossings, zoom changes, patches, and selection; includes a focus-viewport helper for the shell. The naive adapter now shares the same builder and remains the benchmark baseline. | `src/nneditor/rendering/flet_shapes.py`, `src/nneditor/rendering/flet_canvas_managed.py`, `src/nneditor/rendering/flet_canvas.py`, `tests/unit/test_flet_canvas_managed.py`; 442 tests, 96% coverage | 14/68 |
| 2026-07-28 | Completed P1.8: the Flet shell — open button with file picker and async job watching, graph panel with nested subgraphs and function bodies, search with jump-to-node focus, the managed renderer as the graph surface with pan/zoom gestures, a metadata/capability/findings inspector fed by a Flet-free viewmodel layer, an error banner for failed opens, and side panels that collapse below 900 px. The viewmodel and shell wiring are unit-tested headlessly against a stub page; the desktop app launches without errors. | `src/nneditor/ui/app.py`, `src/nneditor/ui/viewmodel.py`, `tests/unit/test_viewmodel.py`, `tests/unit/test_shell.py`, `tests/unit/test_app.py`; 458 tests, 96% coverage | 15/68 |
| 2026-07-28 | Completed P1.9: user-chosen open paths are validated at the application boundary (missing, non-file, and unreadable targets become explainable session errors); session state — recent models and per-artifact last view keyed by content hash — persists atomically in the user state directory, never beside artifacts; corrupt or newer-versioned state files are discarded, not migrated. The shell restores the saved graph, viewport, and surviving selection when the same artifact reopens. Traversal and range safety were already enforced at the storage layer (P0.4/P1.4). | `src/nneditor/application/persistence.py`, `src/nneditor/application/session.py`, `src/nneditor/ui/app.py`, `tests/unit/test_persistence.py`; 470 tests | 16/68 |
| 2026-07-28 | Completed P1.10 and passed the Phase 1 exit gate: the vertical-slice suite proves a 2 MiB external-weights model opens, renders, searches, and inspects metadata with the weights file never opened (tensor-store file accounting) and structural reads under 1% of a payload-dominated file (read instrumentation); explicit inspection reads only the requested slice. Repeated open/close cycles release resources; cancelled opens leave no session; the desktop app was launched interactively without errors and the Flet web server boot smoke (env-gated `NNEDITOR_SMOKE=1`) serves the shell over HTTP. Embedded/external data, symbolic shapes, custom operators, nested graphs, malformed offsets, and missing data were already covered by the P0.4/P1.2 suites. **Phase 1 complete — milestone M1 (ONNX Viewer Alpha) reached.** | `tests/integration/test_vertical_slice.py`; full suite 474 passed + 3 gated/platform skips, 96% coverage; ruff and strict mypy clean | 17/68 |
| 2026-07-28 | Completed P2.1: added the replaceable `GroupDetector` contract and source-hierarchy detector. Exporter node-name/source-location prefixes produce nested candidates with retained scope evidence and confidence; control-flow body graphs receive explicit containment candidates. | `src/nneditor/analysis/hierarchy.py`, `src/nneditor/analysis/detectors.py`, `tests/unit/test_hierarchy.py` | 18/68 |
| 2026-07-28 | Completed P2.2: added derived graph adjacency, reachability, dominators/post-dominators, branch/merge single-entry/single-exit regions, residual-Add regions, and conservative cycle handling. Exact dominator analysis is bounded at 2,000 nodes so hostile or huge flat graphs cannot force quadratic state. | `src/nneditor/analysis/detectors.py`, `tests/unit/test_hierarchy.py` | 19/68 |
| 2026-07-28 | Completed P2.3: implemented a versioned deterministic motif library for convolution/norm/activation, attention cores, feed-forward blocks, and normalization/activation layers. Every match retains its operator path and a human-readable versioned explanation. | `src/nneditor/analysis/detectors.py`, `tests/unit/test_hierarchy.py` | 20/68 |
| 2026-07-28 | Completed P2.4: added declaration-order repeated-subgraph detection using canonical operator/domain, attribute, input/output shape, and internal-edge signatures. Hashes are bucket keys only and full signatures are compared after hashing; a forced-collision test proves different structures never group. Exact scanning is bounded at 5,000 nodes. | `src/nneditor/analysis/detectors.py`, `tests/unit/test_hierarchy.py` | 21/68 |
| 2026-07-28 | Completed P2.5: candidate reconciliation pools independent evidence for exact memberships, retains strict nesting, deterministically resolves partial overlaps by confidence and semantic priority, assigns membership-stable group IDs and nearest parents, and produces content-derived hierarchy revisions for caches. | `src/nneditor/analysis/hierarchy.py`, `tests/unit/test_hierarchy.py` | 22/68 |
| 2026-07-28 | Completed P2.6: implemented group, split, merge, rename, lock/unlock, reject, and reset commands; manual and locked decisions override automatic overlaps. Corrections save immediately and atomically in a versioned per-artifact sidecar under the user state directory; corrupt/newer sidecars are discarded. The shell exposes explicit multi-selection and all correction actions. | `src/nneditor/application/hierarchy.py`, `src/nneditor/rendering/flet_canvas_managed.py`, `src/nneditor/ui/app.py`, `tests/unit/test_hierarchy.py`, `tests/unit/test_shell.py` | 23/68 |
| 2026-07-28 | Completed P2.7: implemented architecture, block, layer, and operator representations with zoom thresholds and explicit selection; internal edges collapse, parallel edges aggregate deterministically, and every glyph maps back to its operators. Slice caches now include hierarchy revision, detail, and root group; a deterministic architecture fallback caps overviews at 1,000 glyphs. | `src/nneditor/analysis/lod.py`, `src/nneditor/application/slices.py`, `tests/unit/test_semantic_navigation.py`, `tests/performance/test_phase2_hierarchy_perf.py` | 24/68 |
| 2026-07-28 | Completed P2.8 and passed the Phase 2 exit gate: added graph-aware search/jump, graph/group breadcrumbs, clickable minimap geometry, root-group navigation, arrow-key navigation, viewport slices, additive selection, stable logical selection across collapse/expand, and cooperative cancellation throughout hierarchy analysis. CNN, attention, feed-forward, residual, repeated-block, nested-control-flow, collision, correction, persistence, cancellation, and 10k-node fixtures pass. On the Phase 0 machine, 10k collapse is 39.6/40.8 ms p50/p95 and expand is 98.7/102.3 ms, both below the 250 ms budget. **Phase 2 complete — milestone M2 (Semantic Viewer Beta) reached.** | `docs/hierarchy-and-semantic-zoom.md`, `tests/unit/test_hierarchy.py`, `tests/unit/test_semantic_navigation.py`, `tests/integration/test_phase2_hierarchy.py`, `tests/performance/test_phase2_hierarchy_perf.py`; full suite 514 passed + 3 gated/platform skips, 93% coverage; ruff and strict mypy clean | 25/68 |
| 2026-07-28 | Completed P3.1: introduced the shared `BudgetedCache` — thread-safe LRU under a cost budget with hit/miss/eviction counters and consistent snapshots; oversized values are never admitted. Refactored the tensor store's slice cache and the slicer's semantic-slice and base-layout caches onto it, added a typed-materialization cache, and made the tensor store thread-safe for concurrent job reads. | `src/nneditor/storage/cache.py`, `src/nneditor/storage/store.py`, `src/nneditor/application/slices.py`, `tests/unit/test_cache.py` | 26/68 |
| 2026-07-28 | Completed P3.2: the indexer records each embedded-typed tensor's message span without reading it; `TensorRef.typed_span` carries it (IR schema 1.0→1.1, older documents read directly and disclose typed tensors as unavailable); `materialize_typed_tensor` decodes typed protobuf fields (float/double/int64/uint64/int32-family incl. float16/bfloat16 bit patterns) into packed bytes with cancellation checkpoints; the store's `materialization()` discloses range vs full-parse vs unavailable *before* any read, and one materialization serves every later slice from cache. Strings stay unavailable with the reason stated. | `src/nneditor/adapters/onnx/typed_data.py`, `src/nneditor/ir/core.py`, `src/nneditor/ir/serialize.py`, `src/nneditor/ir/schema.py`, `tests/unit/test_typed_data.py`; 540 tests, 93% coverage | 27/68 |
| 2026-07-28 | Completed P3.3: pure-Python streaming statistics — extrema, mean/std, sparsity, NaN/Inf counts (excluded from extrema and binning, never propagated), and histograms — computed in fixed chunks with a cancellation checkpoint per chunk; verified differentially against numpy including bin-exact histogram equality. Results persist in an atomic sidecar keyed by content hash and statistics version (stale versions and corrupt files are discarded, not migrated) and are exposed on sessions as cached lookups plus cancellable background jobs; a reopened artifact reuses persisted summaries with zero recomputation. | `src/nneditor/analysis/statistics.py`, `src/nneditor/application/statistics_store.py`, `src/nneditor/application/session.py`, `tests/unit/test_statistics.py`; 555 tests, 93% coverage | 28/68 |
| 2026-07-28 | Completed P3.5: implemented the revision model per IR spec §4 — immutable revisions identified by a hash of parent id plus command manifest (forged ids and broken chains are rejected at construction), the `ReplaceTensorBytes` command with JSON manifests, validation states that must carry findings on failure, and `RevisionChain` whose undo/redo move a cursor between revisions; applying after undo discards the redo tail, and a failed validation raises with every finding and creates nothing. | `src/nneditor/editing/revisions.py`, `tests/unit/test_revisions.py` | 29/68 |
| 2026-07-28 | Completed P3.6: revision chains persist atomically in a content-hash-keyed sidecar after every mutation, so a process crash loses at most the in-flight edit; reopening restores the chain and cursor exactly (including redo history). Corrupt or tampered sidecars — including a forged delta, caught because the revision id no longer matches its content — are quarantined to `*.corrupt` and the session recovers at base. `EditingController` gives sessions edited-view reads over the untouched store, and `discard_all` provably returns to source bytes. | `src/nneditor/application/edits_store.py`, `src/nneditor/application/editing.py`, `src/nneditor/application/session.py` | 30/68 |
| 2026-07-28 | Completed P3.7: `preview_diff` summarizes the applied chain purely from its deltas — changed tensors, span counts, bytes changed, element-aligned changed-element counts, and bounded decoded before/after values via the same decoders statistics uses; unaligned spans stay byte-level rather than guessing. No tensor is re-read and nothing is copied. | `src/nneditor/editing/diff.py`, `src/nneditor/analysis/statistics.py` (`decode_packed`/`element_width`); 573 tests, 93% coverage | 31/68 |
| 2026-07-28 | Completed P3.4: the inspector now shows, for every weight tensor of the selected node, its dtype/shape/element count, storage kind with the store's access disclosure (range vs full-parse vs unavailable) *before* any read, byte and memory estimates, quantization placeholder, import provenance, a decoded preview slice with an explicit error state for undecodable dtypes, and — once a cancellable background job finishes — statistics with sparsity, non-finite counts, and a merged-bin text histogram. All presentation logic is Flet-free and headlessly tested; the desktop app boots clean. | `src/nneditor/ui/viewmodel.py`, `src/nneditor/ui/app.py`, `tests/unit/test_tensor_inspector.py` | 32/68 |
| 2026-07-28 | Completed P3.8 and passed the Phase 3 exit gate: against an external-weights model, slices and statistics are inspected lazily (zero files open before the first read), one copy-on-write weight replacement is applied, the diff preview names the changed element and values, and undo/redo round-trip exactly — with both source files byte-identical throughout and every cache within budget; a fresh service then recovers the committed edit and persisted statistics from sidecars alone. Hardening adds 8-thread concurrent reads under a 512-byte ceiling, mid-stream cancellation persisting nothing, COW over full-parse typed tensors, and a 20-deep undo/redo chain. **Phase 3 complete.** | `tests/integration/test_phase3_exit.py`; full suite 586 passed + 3 gated/platform skips, 93% coverage; ruff and strict mypy clean | 33/68 |
| 2026-07-28 | Hardened the Phase 3/4 boundary before surgery work: artifact identity now covers the model and every external payload; tensor reads re-verify source components; range edits no longer materialize whole tensors; typed previews require consent; browser uploads are isolated/cleaned up; import/export hashing and copying are cancellable; and ONNX is an explicit runtime dependency only for schemas and the export boundary. | `src/nneditor/adapters/onnx/indexer.py`, `src/nneditor/storage/store.py`, `src/nneditor/editing/revisions.py`, `src/nneditor/ui/app.py`, `src/nneditor/application/session.py`, `docs/platform-support.md`; identity, mutation, range-read, consent, upload, cancellation, and sidecar tests | 33/68 |
| 2026-07-28 | Completed P4.1: added reversible node-rename and schema-typed scalar/list attribute commands with deterministic JSON manifests and immutable IR application. Tensor- and graph-valued attribute changes remain read-only. | `src/nneditor/editing/commands.py`, `src/nneditor/editing/validation.py`, `tests/unit/test_phase4_validation.py` | 34/68 |
| 2026-07-28 | Completed P4.2: added constrained operator replacement within modelled compatibility families, allowlisted shape/dtype-preserving unary insertion/removal, and edge reconnection that refuses unresolved compatibility. Every command retains explicit inverse preconditions. | `src/nneditor/editing/commands.py`, `src/nneditor/editing/validation.py`, `tests/unit/test_phase4_validation.py` | 35/68 |
| 2026-07-28 | Completed P4.3: the non-mutating validation pipeline checks current revision identity, entry-graph scope, IR invariants, ONNX opset/schema availability, arity, required and typed attributes, input/output dtype constraints, known shape/symbolic compatibility, tensor bounds, and export capability. Failed or stale transactions create no revision. | `src/nneditor/editing/validation.py`, `src/nneditor/application/editing.py`, `tests/unit/test_phase4_validation.py`, `tests/unit/test_revisions.py` | 36/68 |
| 2026-07-28 | Completed P4.4: the Flet edit panel exposes every Phase 4 graph command, prepare/validate, coded findings, command and capability-impact preview, atomic commit/reject, committed graph/tensor diff, visible undo/redo, and desktop export. Browser uploads are staged safely; browser multi-file export packaging is disclosed as deferred. | `src/nneditor/ui/app.py`, `tests/unit/test_phase4_validation.py`, `tests/unit/test_shell.py` | 37/68 |
| 2026-07-28 | Completed P4.5: committed revisions materialize into a new ONNX artifact through a private sibling stage; external payloads copy in bounded chunks, tensor spans splice with before-byte verification, large embedded tensors externalize at the configured threshold, and source-colliding external paths relocate. Full checker validation and lazy reopen precede external-first/model-last publication with rollback. | `src/nneditor/adapters/onnx/exporter.py`, `tests/integration/test_phase4_export.py` | 38/68 |
| 2026-07-28 | Completed P4.6: deterministic JSON export reports record fidelity, component/source/target hashes, tool and dependency versions, revision IDs, complete command manifests, structural checks, unsupported operators, metadata and dtype/layout changes, unresolved shapes, custom-runtime needs, warnings, numerical evidence, and written files. | `src/nneditor/adapters/onnx/exporter.py`, `docs/edit-export-policy.md`, `tests/integration/test_phase4_export.py` | 39/68 |
| 2026-07-28 | Completed P4.7: optional numerical smoke comparison requires explicit approval and named inputs, runs ONNX's reference evaluator in a separate minimal-environment subprocess with a timeout and supported OS resource limits, uses configurable tolerances, and claims evidence only for supplied inputs. | `src/nneditor/adapters/onnx/numerical.py`, `src/nneditor/adapters/onnx/smoke_worker.py`, `tests/integration/test_phase4_export.py` | 40/68 |
| 2026-07-28 | Completed P4.8 and passed the Phase 4 exit gate: golden/integration coverage includes byte-identical no-op output, every supported command, embedded typed and external tensor edits, threshold externalization, source relocation/immutability, custom domains, symbolic shapes, rejected/stale edits, consent, cancellation, source mutation, destination collision, simulated mid-publication rollback, report round-trip, numerical comparison, checker validation, and lazy reopen. The desktop/web package builds and the web server boot smoke passes. **Phase 4 complete — milestone M3 (ONNX Editor Beta) reached.** | `tests/unit/test_phase4_validation.py`, `tests/integration/test_phase4_export.py`; full suite 614 passed + 3 skips, 90% coverage; ruff, strict mypy, wheel/sdist build, and web boot smoke clean | 41/68 |
| 2026-07-28 | Completed P5.1: added canonical transformation manifests and previews covering runtime, representation, storage effect, bit scheme, quantization parameters, calibration/plugin provenance, pruning mode, error, sparsity, byte estimates, and four distinct product capability claims. | `src/nneditor/transformations/schema.py`, `tests/unit/test_transformations.py` | 42/68 |
| 2026-07-28 | Completed P5.2: added validated 8-bit signed-symmetric and signed/unsigned-asymmetric dequantized-float weight conversion with per-tensor/per-channel axes, exact candidate bytes, numerical error summaries, deterministic commands, and immutable preview/commit/reject. | `src/nneditor/transformations/engine.py`, `src/nneditor/editing/commands.py`, `tests/integration/test_phase5_transformations.py` | 43/68 |
| 2026-07-28 | Completed P5.3: added entry-graph ONNX Q/DQ insertion for reference-runtime-supported opset 19+, explicit runtime targeting, generated scale/zero-point revision data, initializer/value binding, duplicate-boundary rejection, export materialization, checker validation, and reference-runtime numerical evidence. | `src/nneditor/transformations/engine.py`, `src/nneditor/adapters/onnx/exporter.py`, `tests/integration/test_phase5_transformations.py` | 44/68 |
| 2026-07-28 | Completed P5.4: added explicit masks, threshold zeroing, and exact N:M logical pruning with mask provenance, before/after sparsity, unchanged dtype/shape/byte length, and explicit refusal to claim storage or execution gains. | `src/nneditor/transformations/engine.py`, `docs/quantization-and-pruning.md`, `tests/integration/test_phase5_transformations.py` | 45/68 |
| 2026-07-28 | Completed P5.5: added the constrained terminal-MatMul output-channel pattern with rank-2 weight and static-shape proof, atomic tensor/value/output propagation, smaller payload export, and fail-closed rejection for consumers, unresolved shapes, invalid channels, and all other patterns. | `src/nneditor/transformations/engine.py`, `src/nneditor/editing/commands.py`, `src/nneditor/adapters/onnx/exporter.py`, `tests/integration/test_phase5_transformations.py` | 46/68 |
| 2026-07-28 | Completed P5.6: added an external calibration provider protocol and job boundary with cooperative cancellation, sample/byte ceilings, finite-result checks, and provider/version/sample/byte provenance; provider objects and datasets never enter revision commands. | `src/nneditor/transformations/calibration.py`, `src/nneditor/application/session.py`, `tests/unit/test_transformations.py` | 47/68 |
| 2026-07-28 | Completed P5.7 and passed the Phase 5 exit gate: every supported scheme has deterministic command/report round trips, preview validation, capability claims, undo/redo and recovery coverage, lossy export fidelity, full ONNX checking, lazy reopen, Q/DQ numerical comparison, and UI preview/apply/reject coverage. **Phase 5 complete — milestone M4 (Weight Surgery Release) reached.** | `tests/unit/test_transformations.py`, `tests/integration/test_phase5_transformations.py`; full suite 656 passed + 3 skips, 90.40% coverage; ruff, strict mypy, wheel/sdist build, and web boot smoke clean | 48/68 |
| 2026-07-28 | Fixed the four correctness findings from the marked-feature audit: (1) structured pruning now rejects a weight initializer with any consumer besides the target MatMul input (previously a shared weight was silently resized, corrupting the sibling consumer); (2) unary-node removal now schema-validates the rewired consumer instead of skipping `_schema_findings`, catching dtype-constraint violations the precondition layer cannot see; (3) export reports no longer assert hardcoded checks the exporter never ran, and external-tensor weight edits grade `semantically equivalent` instead of `lossless`; (4) the transformation preview no longer claims executability is "verified" before export, and `calibration_required` now reflects the scheme's needs rather than whether provenance was attached. Each fix is pinned by a new test; the P4.3 task text now states the prepare-time versus export-time validation split explicitly. | `src/nneditor/transformations/engine.py`, `src/nneditor/editing/validation.py`, `src/nneditor/adapters/onnx/exporter.py`, `src/nneditor/ui/app.py`, `tests/integration/test_phase5_transformations.py`, `tests/unit/test_phase4_validation.py`, `tests/integration/test_phase4_export.py`; 661 tests, 90% coverage | 48/68 |
| 2026-07-28 | Completed Phase 6 (P6.1-P6.6): native PyTorch ingestion with `torch` as a dev-only fixture generator. PT2 archives parse from JSON plus uncompressed zip members (schema major 8 frozen and enforced, big-endian refused), mapping ATen targets, typed arguments, symbolic dimensions, parameter/buffer signatures, and range-readable weights. Checkpoints open through a restricted, **non-executing** pickle interpreter (allow-listed opcodes and globals, `BUILD` state carried as data, hostile `__reduce__` payloads refused before any call), with strided views, shared storages, compressed members, and missing storages each diagnosed rather than guessed. Safetensors gained a native reader and a deterministic atomic writer verified against the reference package. FX modules import via a `pickletools` opcode walk plus `ast` recovery of the generated forward, staying editing/export-unavailable per contract. Artifact kind is detected by content, not extension. | `src/nneditor/adapters/pytorch/`, `src/nneditor/adapters/detect.py`, `tests/integration/test_pytorch_adapters.py`, `tests/unit/test_pickle_scan.py`, `tests/unit/test_pt2_mapping.py` | 54/68 |
| 2026-07-28 | Completed P7.1-P7.3, P7.5-P7.6: a focused textual-MLIR reader ingests StableHLO from real `jax.export` output — module attributes, function signatures, SSA values with tensor types, dynamic extents, nested regions preserved verbatim, custom calls flagged and marked uneditable, and unparseable statements degraded to opaque nodes instead of failing the import. Weights-only JAX/Orbax sessions open through the safetensors path with the parameter tree carried as an extension; `export_weights_only` writes edited tensors as safetensors labelled weights-only. Four real jax modules (static, shape-polymorphic, control-flow, composite) import with **zero** opaque nodes. P7.4 (isolated trusted tracing) remains open by design — it needs the Phase 8 worker isolation. | `src/nneditor/adapters/jax/`, `tests/integration/test_jax_adapters.py`, `tests/integration/test_phase6_7_compatibility.py`; 736 tests, 90% coverage; ruff and strict mypy clean | 59/68 |
| 2026-07-28 | Benchmarked the real product path on a 14,000-node / 35 MB ONNX model (`benchmarks/large_model_bench.py`) and fixed two scale defects it exposed. (1) Layout recomputed the widest row inside its per-row loop, making it O(layers²) — 196 M generator calls; hoisting it cut architecture slicing from **32.9 s to 1.19 s**. (2) `SpatialIndex.query` walked every grid cell in the query rectangle, so a zoomed-out viewport over a tall graph swept millions of empty cells (52 M dict lookups); it now walks stored cells when that is cheaper, cutting the zoomed-out frame from **15.1 s to 68 ms**. Also added a renderer node budget: beyond the shape cap the frame draws a deterministic subset (selection first), reports `dropped_nodes`, and the status bar names the remedy, so a partial view is never mistaken for the whole graph. | `src/nneditor/analysis/layout.py`, `src/nneditor/rendering/spatial.py`, `src/nneditor/rendering/flet_shapes.py`, `src/nneditor/ui/app.py`, `benchmarks/large_model_bench.py`, `tests/unit/test_spatial.py`, `tests/unit/test_flet_canvas_managed.py`; 740 tests, 90% coverage | 59/68 |
| 2026-07-28 | Renamed the distribution, import package, command, application title, local-state namespace, environment flags, tests, documentation, and schema extension namespace to NNEditor. Made `nneditor` PyPI-ready with dynamic single-source versioning, complete package metadata, a typed-package marker, explicit wheel/sdist contents, strict artifact validation, isolated wheel installation checks, and GitHub Actions Trusted Publishing workflows for TestPyPI and PyPI. | `pyproject.toml`, `src/nneditor/`, `.github/workflows/ci.yml`, `.github/workflows/publish.yml`, `docs/publishing.md`; full suite 658 passed + 3 skips, 90.40% coverage; ruff, strict mypy, wheel/sdist metadata, isolated install, and web boot smoke clean | 48/68 |
| 2026-07-28 | Ran a nine-subsystem adversarial audit (storage/IR, ONNX adapter, PyTorch/JAX adapters, analysis, application, editing/transformations, rendering, UI shell, plus a pan/zoom profiling pass) and fixed all ~40 verified defects in parallel waves. Highlights: asymmetric quantization now extends its range to include zero (errors were ~64× the quantization step on positive-only tensors and reached exported Q/DQ parameters); N:M pruning groups along the innermost axis; artifact hashing moved off the store-wide lock and through the reader's own descriptor; dominator analysis sweeps in reverse postorder and the structural detector lost its cubic path (3.9 s → 15 ms and 20.7 s → 0.14 s at 800 nodes); variance is computed stably in the existing second pass and bfloat16 gained decoders; the restricted pickle interpreter, MLIR reader, zip store, and safetensors reader now refuse hostile input with typed diagnostics and bounded work (exponential flatten, ReDoS regexes, negative storage offsets, forged central directories, body-less-function body theft); the Windows smoke worker gained a Job Object memory cap; sessions close on reopen, a failed open can no longer wedge the loading overlay, the Auto detail level is reachable, and sidecar stores survive concurrent saves without quarantining good history. | `tests/unit/test_fixes_*.py`, `tests/integration/test_fixes_*.py` (9 new suites, ~170 new tests); full suite 1,030 passed + 4 skips, 90.82% coverage; ruff and strict mypy clean | 59/68 |
| 2026-07-28 | Fixed the architecture-page blob and revamped semantic LOD: corrected the inverted block/layer antichain preference, made drill-down show a group's interior instead of the group itself, added drill-through of dominant roots (>80% coverage descends to children), re-laid collapsed architecture/block scenes compactly instead of inheriting operator-layout unions, and folded constant feeders into their consumer glyphs. On the 2,199-node Qwen3-VL vision tower the architecture view went from one glyph covering 100% of a 137,964×107,956 scene to 36 compact glyphs (24 numbered blocks + merger + patch embed) at 0.46% max share. Added block organization modes — Auto, Source, Repeated, Patterns, Structural — cycled by a toolbar icon, filtering the cached candidate pool before reconciliation (switching never re-detects), with manual corrections honored in every mode and the choice persisted in the v2 hierarchy sidecar (v1 files migrate losslessly). | `src/nneditor/analysis/lod.py`, `src/nneditor/analysis/layout.py`, `src/nneditor/application/hierarchy.py`, `src/nneditor/ui/app.py`, `tests/unit/test_fixes_lod.py`, `tests/unit/test_fixes_modes.py` | 59/68 |
| 2026-07-28 | Fixed the pan/zoom event-loop stall a profiling pass isolated on 5k-10k-node scenes: the minimap rebuilt one fresh control per scene node on every pointer event (a quadratic Flet list diff measured at 2.1 s per event at 5k nodes and 8.1 s at 10k), gesture events were unthrottled, and each event re-diffed the whole page. Minimap dots now persist across viewport changes with a 600-dot sampling cap and only a persistent viewport rectangle mutates; drag events are throttled to 16 ms; gesture bursts coalesce latest-wins through a single drain task; and viewport applies update only the canvas, status line, and minimap. `MiniMap.project_viewport` factors the projection out of the model so the shell and the minimap builder cannot disagree. | `src/nneditor/ui/app.py`, `src/nneditor/application/navigation.py`, `tests/unit/test_shell.py`; 765 passed + 3 skips, 90.07% coverage; ruff and strict mypy clean | 59/68 |
| 2026-07-29 | Ran a full four-subsystem code-health assessment against this plan and re-verified the quality gates on Windows 11 / Python 3.14.6: ruff clean, strict mypy clean on 150 files, 1,029 passed + 4 skipped at 90.82% coverage. The single failure — the 10k collapse/expand budget test — passes in isolation (2.4 s) and misses its wall-clock budget only under full-suite coverage load; it is a test-design flaw, not a regression. Reopened P7.5: no StableHLO writer exists, and `adapters/jax/stablehlo.py` constructs documents without tensor references, so `export_weights_only` fails for every StableHLO session while `ir/capabilities.py` still advertises re-serialization. Recorded the complete ranked findings and improvement backlog in §11. | `IMPLEMENTATION.md` §11; full suite 1,029 passed + 4 skipped, 90.82% coverage | 58/68 |
| 2026-07-29 | Added Phase 9 (inference tracing and activation inspection): consent-gated isolated execution building on the P4.7 evaluator boundary, reproducible input specifications, a budgeted activation capture store served through the existing tensor-store interface, per-node activation inspection reusing the Phase 3 statistics and inspector paths, intuitive visualizations (heatmaps, feature-map grids, histograms, attention maps), and base-versus-edited trace comparison. The task total grows from 68 to 76; milestone M7 added. | `IMPLEMENTATION.md` §5 Phase 9 | 58/76 |
| 2026-07-29 | Completed the Tier 1 multi-format lifecycle and P7.5 re-scope, Tier 2 concurrency repairs, Windows/performance CI split, adapter/command/rule dispatch seams, and the first Flet shell decomposition. Export is routed by artifact kind; web uploads are content-detected; edit capability is independent of writer availability; source-swap verification and adapter errors are format-neutral. StableHLO is honestly inspection-only, while readable checkpoint tensors export as labelled safetensors without claiming an Orbax writer. Revision/session/hierarchy state is synchronized, background reads use stable views, caches are bounded, and shared layouts survive sibling-session close. Registries now own artifact opening, command codecs/apply handlers, validation rules, and transformation providers. The shell uses an injected renderer protocol and delegates overview, pure parsing/formatting, and panel layout to focused modules. | `src/nneditor/adapters/registry.py`, `src/nneditor/application/`, `src/nneditor/editing/`, `src/nneditor/rendering/`, `src/nneditor/ui/`, `.github/workflows/`, `pyproject.toml`; ruff clean; strict mypy clean on 154 files; 1,052 ordinary tests passed + 4 skipped, 2 performance tests passed separately, 90.92% coverage | 59/76 |
| 2026-07-29 | Completed Phase 9 for desktop and passed its exit gate. ONNX now declares a distinct tracing capability; every other artifact gives an explicit unavailable reason. Each run binds a content-addressed deterministic input specification and an approval naming the model/spec/limits, materializes the selected revision, and executes only in a minimal-environment subprocess under enforced wall, memory, capture, and chunk ceilings. Captures commit atomically into a byte-budgeted range-readable store with visible complete/truncated/dropped/evicted states. Every semantic graph level marks captured nodes; the inspector exposes activation previews, streaming statistics, histograms, heatmaps, feature maps, and motif-driven attention maps. Same-input base/edited revisions compare max-absolute, relative, and cosine error per node. The desktop shell covers empty/loading/partial/error/cancel states; web tracing remains visibly unavailable until P8.3 supplies multi-user worker isolation. **Phase 9 complete (desktop) — milestone M7 reached for the desktop target.** | `src/nneditor/tracing/`, `src/nneditor/application/session.py`, `src/nneditor/ui/`, `docs/inference-tracing.md`, `tests/unit/test_tracing.py`, `tests/unit/test_trace_worker.py`, `tests/unit/test_phase9_shell.py`, `tests/integration/test_phase9_tracing.py`; 1,086 ordinary tests passed + 4 skipped at 90.03% coverage; 2 performance tests pass separately; ruff clean; strict mypy clean on 166 files | 67/76 |
| 2026-07-29 | Made tracing graph-first: model inputs and outputs are explicit boundary nodes; every input has an in-graph `.npy` picker with deterministic-random fallback; UI traces capture the full graph by default; renderer hit testing and selection now include connections; selecting an operator, semantic block, connection, input, or output opens its captured activation data in the inspector. Input changes invalidate stale approval, while narrowed captures remain available through the application API. | `src/nneditor/ui/trace_graph.py`, `src/nneditor/ui/app.py`, `src/nneditor/rendering/hit_testing.py`, `src/nneditor/tracing/inputs.py`, `tests/unit/test_trace_graph.py`, `tests/unit/test_phase9_shell.py`, `tests/unit/test_flet_canvas.py`, `tests/unit/test_flet_canvas_managed.py`; 1,094 ordinary tests passed + 4 skipped at 90.07% coverage; 2 performance tests pass separately; ruff and strict mypy clean | 67/76 |
| 2026-07-29 | Removed avoidable tracing setup friction in the desktop shell. Starting a trace now presents the approval in context instead of failing with an instruction to find it elsewhere; omitted required mask inputs receive a generated all-valid mask using resolved model dimensions; selecting a traced node, boundary, or connection builds its activation views automatically; and a dedicated action opens the visualizations in a large overlay while keeping the compact inspector summary. | `src/nneditor/ui/app.py`, `src/nneditor/ui/trace_graph.py`, `src/nneditor/tracing/inputs.py`, `tests/unit/test_phase9_shell.py`, `tests/unit/test_trace_graph.py` | 67/76 |
| 2026-07-29 | Finished the desktop release surface at version 1.0.0: the model can be closed explicitly, Save appears only while the working revision is modified, Open Model exposes format-specific extension filters, the release README includes the application icon and traced-model screenshots, and deterministic image/time-series `.npy` examples are available for first-run exploration. | `src/nneditor/ui/app.py`, `README.md`, `src/nneditor/__init__.py`, `tests/fixtures/inputs/`, `tests/unit/test_shell.py`, `tests/unit/test_package.py` | 67/76 |
| 2026-07-29 | Added an integrated **Generate test input** workspace. Images can be resized, color-converted, normalized, and laid out as model-ready arrays; mask, CSV/TSV/time-series, and generic tensor generators are bounded and save atomically; generated values can be assigned directly to a selected model input. The CLI accepts a model path, Open Model shares one extension catalog, and Windows users can register supported formats under per-user **Open with** capabilities without overwriting `UserChoice` or silently changing defaults. | `src/nneditor/input_generation.py`, `src/nneditor/ui/input_workspace.py`, `src/nneditor/artifact_formats.py`, `src/nneditor/desktop/windows_associations.py`, `src/nneditor/ui/app.py`, `tests/unit/test_input_generation.py`, `tests/unit/test_windows_associations.py`; commit `786e9e4`; 1,159 ordinary tests passed + 4 skipped + 2 performance tests deselected at 90.26% coverage; 2 performance tests passed separately; ruff, strict mypy, wheel/sdist build, and strict distribution checks clean; responsive Flet visual smoke passed | 67/76 |

## 4. Target Architecture

```text
Flet desktop/web shell
  hierarchy | graph renderer | inspector | edits | jobs
                         |
              application service
  sessions | revisions | capabilities | validation | job control
                         |
               canonical graph IR
  documents | graphs | nodes | values | blocks | provenance
                /                       \
      analysis and editing        artifact and tensor store
  shapes | regions | patterns     mmap | ranges | LRU | COW
                \                       /
     ONNX | PyTorch Export/FX | StableHLO/JAX adapters
```

### Proposed repository layout

```text
nneditor/
├── pyproject.toml
├── src/nneditor/
│   ├── adapters/
│   ├── analysis/
│   ├── application/
│   ├── editing/
│   ├── ir/
│   ├── rendering/
│   ├── storage/
│   ├── ui/
│   └── workers/
├── tests/
│   ├── fixtures/
│   ├── integration/
│   ├── performance/
│   └── unit/
├── benchmarks/
├── docs/
│   ├── adr/
│   ├── artifact-capabilities.md
│   ├── edit-export-policy.md
│   ├── ir-schema.md
│   └── renderer-benchmark.md
└── web_renderer/                 # Only if selected by Phase 0
```

The `web_renderer/` directory is conditional. Do not create or commit to a
JavaScript renderer until the Phase 0 benchmark chooses it.

## 5. Step-by-Step Work Plan

### Phase 0 — Feasibility and Foundations

**Purpose:** Resolve the three largest risks before building the product:
rendering scale, lazy artifact access, and IR fidelity.

- [x] **P0.1 — Bootstrap the project.** Select supported Python and Flet
  versions after checking current official compatibility; create the package,
  dependency groups, linting, type checking, tests, and CI. Record supported
  desktop and browser targets. — COMPLETE (Codex, 2026-07-28)
- [x] **P0.2 — Define artifact and capability contracts.** Document accepted
  ONNX, PyTorch, JAX/StableHLO, checkpoint, and safetensors artifacts. For each,
  state whether topology, weights, metadata, editing, execution, and export are
  available and why. — COMPLETE (Claude, 2026-07-28)
- [x] **P0.3 — Specify the versioned IR.** Define documents, graphs, nodes,
  ports, values, symbolic shapes, tensor references, nested subgraphs, groups,
  provenance, revisions, extension namespaces, and capability reasons. Specify
  stable-ID and migration rules. — COMPLETE (Claude, 2026-07-28)
- [x] **P0.4 — Prototype the ONNX lazy indexer.** Index topology and embedded or
  external tensor locations without converting tensors to arrays. Validate
  paths, offsets, lengths, integer arithmetic, and checksums. Instrument file
  reads so tests can prove which byte ranges were accessed. — COMPLETE (Claude,
  2026-07-28)
- [x] **P0.5 — Benchmark rendering alternatives.** Implement the minimal
  `GraphRenderer` contract and compare viable Flet Canvas, custom Flet/Flutter
  control, and hosted web renderer approaches on desktop and web. Test 1,000
  and 10,000 visible nodes, tens of thousands of edges, hit testing,
  collapse/expand, incremental patches, viewport culling, and browser memory.
  — COMPLETE (Claude, 2026-07-28)
- [x] **P0.6 — Prototype revision-based copy-on-write.** Change one weight
  without modifying the source artifact; store a compact delta, undo and redo
  it, then export and structurally validate a new ONNX artifact. — COMPLETE
  (Claude, 2026-07-28)
- [x] **P0.7 — Record the architecture decision.** Compare spike results,
  choose the renderer and topology representation, finalize provisional
  performance budgets, list rejected alternatives, and update the repository
  layout and dependency policy. — COMPLETE (Claude, 2026-07-28; ADR approved
  by Imed Bouazizi)

**Provisional benchmark targets**

These are decision thresholds, not product promises. Adjust them in P0.7 using
measured hardware and browser data.

- A compact index for a graph much larger than the visible region remains
  usable without sending the entire graph to the renderer.
- A 10,000-node synthetic visible graph is rendered without UI lockups.
- Pan/zoom maintains at least 30 frames per second at the agreed normal working
  set.
- Selection/hit testing responds within 100 ms at p95.
- Collapse/expand produces visible feedback within 250 ms at p95.
- Initial ONNX import performs no tensor-to-array conversion.
- External tensor reads are range-bound and observable in tests.

**Exit gate:** Approve an architecture decision record containing renderer
choice, fallback, IR versioning approach, artifact contracts, security boundary,
and measured benchmark results. Stop and redesign if no renderer meets the
working-set target on both required deployment modes.

### Phase 1 — Read-Only ONNX Vertical Slice

**Purpose:** Deliver an end-to-end viewer before adding automatic grouping or
editing.

- [x] **P1.1 — Implement IR core types.** Add validated immutable model,
  graph, node, value, tensor-reference, provenance, and capability types with
  deterministic serialization. — COMPLETE (Claude, 2026-07-28)
- [x] **P1.2 — Implement the ONNX adapter.** Import operator domains,
  attributes, inputs/outputs, initializers, functions, control-flow subgraphs,
  symbolic shapes, external data, producer metadata, and source extensions.
  — COMPLETE (Claude, 2026-07-28)
- [x] **P1.3 — Add import diagnostics.** Produce warnings and errors for
  unsupported operators, malformed tensor references, missing external files,
  incomplete shapes, custom domains, and lost metadata. — COMPLETE (Claude,
  2026-07-28)
- [x] **P1.4 — Build the artifact and tensor-store interfaces.** Support local
  files, safe range reads, metadata-only tensor access, cancellation, cache
  limits, and explicit resource cleanup. — COMPLETE (Claude, 2026-07-28)
- [x] **P1.5 — Build session and application services.** Open and close models,
  track capabilities, own background jobs, isolate UI state from the IR, and
  expose renderer-facing graph-slice queries. — COMPLETE (Claude, 2026-07-28)
- [x] **P1.6 — Implement graph slicing and layout caching.** Return only the
  requested root group, detail level, viewport, search result, or graph
  neighborhood. Cache geometry by model hash, hierarchy revision, and layout
  settings. — COMPLETE (Claude, 2026-07-28; root-group and detail-level
  slicing joins the cache key when Phase 2 introduces hierarchy)
- [x] **P1.7 — Implement the selected renderer adapter.** Support pan, zoom,
  selection, focus, incremental patching, viewport updates, and replacement of
  the visible graph without coupling UI code to a specific renderer.
  — COMPLETE (Claude, 2026-07-28)
- [x] **P1.8 — Build the Flet shell.** Add file/open status, hierarchy panel,
  graph surface, metadata inspector, job status, errors, and responsive desktop
  and browser layouts. — COMPLETE (Claude, 2026-07-28)
- [x] **P1.9 — Add safe artifact loading and session persistence.** Treat paths
  and archives as untrusted; reject traversal and invalid ranges; store session
  state and cached layout separately from source artifacts. — COMPLETE
  (Claude, 2026-07-28)
- [x] **P1.10 — Complete the ONNX vertical-slice test suite.** Cover embedded
  and external data, symbolic shapes, custom operators, nested graphs, malformed
  offsets, missing data, repeated opens, cancellation, and desktop/web smoke
  tests. — COMPLETE (Claude, 2026-07-28)

**Exit gate:** Open a representative large ONNX model with external data,
navigate and inspect its graph on desktop and web, and demonstrate through read
instrumentation that tensor payloads were not eagerly materialized.

### Phase 2 — Hierarchy, Block Detection, and Semantic Zoom

**Purpose:** Make large architectures understandable rather than merely
renderable.

- [x] **P2.1 — Import source hierarchy.** Convert exporter scopes, names, and
  graph nesting into candidate groups while retaining confidence and source
  evidence. — COMPLETE (Codex, 2026-07-28)
- [x] **P2.2 — Detect structural regions.** Add single-entry/single-exit
  regions, dominators/post-dominators, branch/merge boundaries, residual
  regions, and control-flow containment behind a replaceable graph-algorithm
  interface. — COMPLETE (Codex, 2026-07-28)
- [x] **P2.3 — Add a deterministic pattern library.** Detect initial
  convolution/norm/activation, attention, feed-forward, normalization, and
  residual-add motifs with versioned patterns and unit fixtures. — COMPLETE
  (Codex, 2026-07-28)
- [x] **P2.4 — Add repeated-subgraph detection.** Compute canonical,
  shape-aware, and attribute-aware structural hashes; find repeated transformer
  and residual blocks; guard against hash collisions. — COMPLETE (Codex,
  2026-07-28)
- [x] **P2.5 — Reconcile group candidates.** Resolve overlaps and nesting,
  assign stable group IDs, calculate confidence, and retain a human-readable
  explanation for every automatic grouping decision. — COMPLETE (Codex,
  2026-07-28)
- [x] **P2.6 — Add manual hierarchy controls.** Support group, split, merge,
  rename, lock, reject, reset, and persistence of overrides in a sidecar view
  document rather than the model. — COMPLETE (Codex, 2026-07-28)
- [x] **P2.7 — Implement semantic levels of detail.** Define architecture,
  block, layer, and operator representations; aggregate edges and metrics; swap
  representations according to zoom and explicit user navigation. — COMPLETE
  (Codex, 2026-07-28)
- [x] **P2.8 — Complete navigation and hierarchy tests.** Add search/jump,
  breadcrumbs, minimap, neighborhood expansion, keyboard navigation, stable
  selection across levels, and correctness/performance fixtures for CNNs and
  transformers. — COMPLETE (Codex, 2026-07-28)

**Exit gate:** Automatically detect and explain useful groups in representative
CNN and transformer fixtures; allow users to correct them; collapse and expand
without losing selection or exceeding the Phase 0 latency budgets.

### Phase 3 — Tensor Inspection and Revision Foundation

**Purpose:** Inspect large weights safely and establish the mechanics needed for
editing.

- [x] **P3.1 — Implement size-bounded caches.** Add separate metadata,
  statistics, tensor-slice, and layout caches with observable hit/miss/eviction
  behavior and configurable memory budgets. — COMPLETE (Claude, 2026-07-28;
  model metadata lives in the in-memory document and needs no separate cache)
- [x] **P3.2 — Implement tensor slice access.** Read supported local tensor
  ranges without loading unrelated tensors, and disclose when a source
  container forces full-tensor materialization. — COMPLETE (Claude, 2026-07-28)
- [x] **P3.3 — Add cancellable statistics jobs.** Stream min/max/mean,
  histogram, sparsity, non-finite count, and memory estimates; persist results
  by artifact hash and statistic version. — COMPLETE (Claude, 2026-07-28)
- [x] **P3.4 — Build the tensor inspector.** Show shape, dtype, layout,
  quantization, storage, preview slices, histograms, sparsity, provenance, and
  estimated parameter/memory costs with loading and error states. — COMPLETE
  (Claude, 2026-07-28)
- [x] **P3.5 — Implement revision and command types.** Add a working revision,
  parent links, command manifests, validation state, compact tensor deltas, and
  deterministic undo/redo. — COMPLETE (Claude, 2026-07-28)
- [x] **P3.6 — Implement copy-on-write tensor storage.** Overlay edits on
  immutable source tensors, manage temporary and persisted deltas, and recover
  cleanly from cancellation or process failure. — COMPLETE (Claude, 2026-07-28)
- [x] **P3.7 — Add a read-only diff preview.** Compare graph metadata and tensor
  summaries between revisions without requiring full tensor copies. — COMPLETE
  (Claude, 2026-07-28)
- [x] **P3.8 — Complete storage and revision tests.** Test cache pressure,
  concurrent requests, cancellation, corrupt sidecars, non-finite data, undo
  chains, cleanup, recovery, and memory ceilings. — COMPLETE (Claude,
  2026-07-28)

**Exit gate:** Inspect slices and statistics lazily, apply one copy-on-write
weight replacement, preview the diff, and undo/redo it without altering the
source or exceeding the configured cache budget.

### Phase 4 — ONNX Model Surgery and Export

**Purpose:** Add a deliberately constrained and verifiable editing workflow.

- [x] **P4.1 — Implement attribute-edit commands.** Initially allow renaming
  and edits to schema-validated attributes that do not invalidate graph
  topology. — COMPLETE (Codex, 2026-07-28)
- [x] **P4.2 — Implement compatible graph commands.** Add supported
  schema-compatible operator replacement, simple unary insertion/removal, and
  compatible edge reconnection with explicit preconditions. — COMPLETE
  (Codex, 2026-07-28)
- [x] **P4.3 — Build the validation pipeline.** Validate command preconditions,
  graph invariants, ONNX schemas, operator versions, shapes, dtypes, symbolic
  constraints, tensor sizes, and target-export capability. — COMPLETE (Codex,
  2026-07-28; scope note 2026-07-28: prepare-time shape checks compare
  *declared* value shapes — full shape inference runs at export via
  `onnx.checker(full_check=True)`; tensor byte/size bounds are enforced in the
  revision chain and exporter rather than this pipeline. Audit fix: removals
  now schema-validate the rewired consumer.)
- [x] **P4.4 — Build the transactional edit UI.** Present command parameters,
  validation errors, metadata/tensor diff, capability changes, and atomic
  commit/reject actions; keep undo/redo visible. — COMPLETE (Codex, 2026-07-28)
- [x] **P4.5 — Implement ONNX export.** Materialize a new artifact from a
  committed revision, preserve untouched source metadata where possible,
  externalize large tensors, write atomically, and never overwrite the source.
  — COMPLETE (Codex, 2026-07-28)
- [x] **P4.6 — Generate an export report.** Include fidelity level, structural
  checks, unsupported operators, metadata changes, dtype/layout changes,
  unresolved shapes, custom runtime needs, and the complete edit manifest.
  — COMPLETE (Codex, 2026-07-28)
- [x] **P4.7 — Add optional numerical smoke comparison.** When an approved
  runtime and valid inputs are available, compare original and edited outputs
  in an isolated worker with configurable tolerances; never claim equivalence
  from structural validation alone. — COMPLETE (Codex, 2026-07-28)
- [x] **P4.8 — Complete golden export tests.** Cover no-op round trips, each
  supported edit, external data, custom domains, symbolic shapes, rejected
  edits, atomic-write failures, and reopening every exported artifact.
  — COMPLETE (Codex, 2026-07-28)

**Exit gate:** All supported edits either fail without mutation or commit
atomically; exported ONNX artifacts reopen and validate; no-op golden round
trips meet the documented fidelity policy.

### Phase 5 — Quantization and Pruning

**Purpose:** Add weight transformations whose representation and runtime
implications are explicit.

- [x] **P5.1 — Define transformation schemas.** Record target runtime,
  bit-width, signedness, scale, zero point, axis, granularity, symmetry,
  calibration needs, operator representation, pruning mode, and storage
  implications. — COMPLETE (Codex, 2026-07-28)
- [x] **P5.2 — Implement previewable weight-only quantization.** Start with a
  small supported set of per-tensor and per-channel schemes; calculate error
  summaries and estimated storage changes before commit.
  — COMPLETE (Codex, 2026-07-28)
- [x] **P5.3 — Add graph-aware ONNX quantization.** Insert or replace only
  supported quantization operators, validate opset/runtime requirements, and
  mark unsupported paths unavailable rather than approximating them silently.
  — COMPLETE (Codex, 2026-07-28)
- [x] **P5.4 — Implement logical pruning.** Support masks, zeroing, and selected
  N:M policies with before/after sparsity and an explicit statement that
  logical sparsity may not shrink files or improve execution.
  — COMPLETE (Codex, 2026-07-28)
- [x] **P5.5 — Add constrained structured pruning.** Support only explicitly
  modeled channel or head patterns; propagate shape changes and reject any
  rewrite that cannot be proven consistent.
  — COMPLETE (Codex, 2026-07-28)
- [x] **P5.6 — Add calibration and plugin boundaries.** Keep activation-aware
  calibration and backend-specific quantizers outside the core command model;
  run data-processing jobs with cancellation, provenance, and resource limits.
  — COMPLETE (Codex, 2026-07-28)
- [x] **P5.7 — Complete numerical and export tests.** Measure transformation
  error, validate quantization metadata and tensor shapes, test undo/redo, and
  verify each supported runtime/export combination.
  — COMPLETE (Codex, 2026-07-28)

**Exit gate:** Every supported transformation has a reproducible manifest,
diff/error preview, validation result, capability report, and reopenable export.
The UI clearly distinguishes mathematical conversion, executable graph support,
storage reduction, and expected acceleration.

### Phase 6 — PyTorch Exported Artifacts

**Purpose:** Add PyTorch support without implying that a checkpoint contains a
recoverable architecture.

- [x] **P6.1 — Verify and freeze supported contracts.** Confirm current
  `torch.export` serialization, FX fallback, control-flow, mutation, custom-op,
  and version-compatibility behavior against official documentation and test
  artifacts.
- [x] **P6.2 — Implement exported-program ingestion.** Map graphs, signatures,
  symbolic constraints, parameters, buffers, source/module metadata, and custom
  operators into the IR with provenance and diagnostic reasons.
- [x] **P6.3 — Implement the constrained FX adapter.** Mark it as secondary,
  version-sensitive support and report features that cannot be represented
  safely.
- [x] **P6.4 — Add weights-only sessions.** Open `state_dict` or paired
  safetensors as weights-only when compatible topology is unavailable; do not
  invent nodes or module structure.
- [x] **P6.5 — Implement supported export paths.** Export updated weights or a
  supported exported graph artifact; use ONNX conversion only when its loss and
  capability report is accepted. Never claim reconstruction of original Python.
- [x] **P6.6 — Complete the PyTorch compatibility suite.** Test multiple
  supported versions, symbolic shapes, common architectures, custom operators,
  weights-only behavior, unsupported mutations, and import/export diagnostics.

**Exit gate:** Supported exported programs produce accurate topology and
capabilities; weights-only artifacts are labeled correctly; exported results
meet the documented version and fidelity matrix.

### Phase 7 — JAX and StableHLO Artifacts

**Purpose:** Add JAX ecosystem support through explicit graph artifacts or an
isolated trusted tracing workflow.

- [x] **P7.1 — Verify and freeze supported contracts.** Confirm current JAX
  export and StableHLO serialization, symbolic shape, custom-call, checkpoint,
  and version behavior using official documentation and fixtures.
- [x] **P7.2 — Implement StableHLO ingestion.** Map functions, regions,
  operations, values, symbolic shapes, constants, attributes, and custom calls
  into the IR while retaining source extensions.
- [x] **P7.3 — Add JAX/Flax/Orbax weights-only sessions.** Display checkpoint
  trees and tensors without claiming topology unless they are paired with a
  compatible exported graph.
- [ ] **P7.4 — Implement trusted tracing as an isolated optional flow.** Require
  explicit consent and input specifications; use a constrained worker with CPU,
  memory, file, network, and time controls; never execute model code in the UI
  process.
- [x] **P7.5 — Implement supported export paths.** Export compatible StableHLO
  or updated checkpoints and report when edits cannot be represented; never
  claim original JAX source reconstruction. — COMPLETE, RE-SCOPED
  (2026-07-29): StableHLO remains inspection-only because the reader preserves
  constants as operation syntax rather than addressable tensors and no
  StableHLO writer exists. Its weights, editing, and export capabilities are
  now explicitly unavailable. Readable JAX/Orbax checkpoint arrays use the
  implemented, labelled weights-only safetensors export; the capability
  contract explicitly says that no Orbax writer exists and that checkpoint
  tree/sharding metadata is not re-serialized.
- [x] **P7.6 — Complete the JAX compatibility suite.** Cover supported versions,
  nested parameter trees, symbolic dimensions, custom calls, control-flow
  regions, weights-only sessions, sandbox limits, and fidelity diagnostics.

**Exit gate:** Supported StableHLO artifacts import with documented fidelity,
weights-only checkpoints are labeled correctly, and all tracing occurs through
the explicit isolated trust boundary.

### Phase 8 — Web Hardening and Plugin Platform

**Purpose:** Make multi-user web deployment and third-party extension safe,
observable, and maintainable.

- [ ] **P8.1 — Add remote artifact storage.** Use content-addressed immutable
  objects, range reads, checksums, deduplication, expiring workspaces, and
  lifecycle cleanup.
- [ ] **P8.2 — Add upload and import controls.** Enforce size, file-count,
  archive, path, decompression, storage, CPU, memory, and duration limits before
  and during processing.
- [ ] **P8.3 — Isolate background workers.** Separate statistics, layout,
  validation, execution, and tracing jobs; support cancellation, timeouts,
  cleanup, and per-session resource ownership.
- [ ] **P8.4 — Add authentication and authorization boundaries.** Protect
  sessions, source artifacts, revisions, exports, jobs, and signed download
  links; prevent cross-session object access.
- [ ] **P8.5 — Add operational observability.** Record structured logs, traces,
  metrics, cache behavior, range-read volume, job resource use, failures, and
  security events without logging tensor contents or secrets.
- [ ] **P8.6 — Define a versioned plugin API.** Start with declarative operator
  metadata, block patterns, import adapters, and transformation providers;
  declare trust level, capabilities, compatibility, and isolation requirements.
- [ ] **P8.7 — Complete accessibility and resilience work.** Test keyboard
  navigation, focus, contrast, screen-reader labels, reconnect behavior,
  resumable jobs where appropriate, error recovery, and responsive layouts.
- [ ] **P8.8 — Complete production readiness tests.** Run threat modeling,
  dependency and artifact scanning, load tests, quota tests, multi-tenant
  isolation tests, backup/recovery exercises, deployment smoke tests, and
  rollback rehearsal.

**Exit gate:** Production web deployment enforces isolation and quotas, large
artifacts use remote range access, jobs are cancellable and observable, and
plugin compatibility/security contracts are versioned and tested.

### Phase 9 — Inference Tracing and Activation Inspection

**Purpose:** Let users push a concrete input through a supported runtime and
inspect the tensors that flow through every node — lazily, within explicit
resource budgets, and without weakening the safe-artifact boundary. Execution
builds on the P4.7 isolated-evaluator seam (separate minimal-environment
subprocess, timeout, OS resource limits) and reuses the Phase 3 tensor-store,
statistics, and inspector infrastructure for captured activations. Desktop
tracing needs only Phases 3-4; web-mode tracing additionally requires the
Phase 8 worker isolation (P8.3).

- [x] **P9.1 — Define the tracing contract and consent boundary.** Add an
  execution-tracing capability per artifact kind with a mandatory reason,
  starting with the ONNX reference evaluator; every other kind discloses why
  tracing is unavailable rather than approximating. Tracing always requires
  explicit per-run user approval naming the model, the input specification,
  and the resource limits; no trace ever runs in the UI process.
- [x] **P9.2 — Implement input specification and binding.** Bind named graph
  inputs from seeded deterministic random data, user-supplied tensor files,
  or constants; validate dtype and shape against declared values; require
  explicit user-provided extents for symbolic dimensions rather than
  guessing. Persist input specifications in an atomic sidecar keyed by
  content hash so traces are reproducible and re-runnable.
- [x] **P9.3 — Implement the isolated trace runner.** Extend the P4.7 worker
  to capture intermediate values by augmenting the in-memory model's outputs
  (the source artifact is never mutated), honoring an explicit node/value
  selection when the user narrows scope. Stream captures to disk in bounded
  chunks with cancellation checkpoints; enforce wall-clock, memory, and
  capture-byte ceilings; a cancelled or failed trace leaves no partial state
  behind.
- [x] **P9.4 — Build the budgeted activation store.** Key captures by
  artifact hash, revision id, input-specification hash, and value id; serve
  reads through the same lazy tensor-store interface as weights (range reads,
  slice cache, statistics jobs); enforce byte budgets with eviction; disclose
  before any read when a capture was dropped, truncated, or evicted so a
  partial trace is never mistaken for a complete one.
- [x] **P9.5 — Wire traces into the graph and inspector.** Mark nodes whose
  values hold captures at every detail level; extend the tensor inspector so
  a selected node shows its captured inputs/outputs alongside its weights,
  with the existing preview-slice, statistics, and histogram paths running on
  captured tensors; keep loading, partial-capture, and error states explicit.
- [x] **P9.6 — Build intuitive activation visualizations.** Add renderer-
  agnostic view builders for heatmaps of 2-D slices, feature-map grids for
  convolutional activations, histograms and line plots for vectors, and
  attention-map views for attention cores detected by the Phase 2 motif
  library. Downsampling is deterministic and disclosed; colormap and
  normalization choices are stated on the view; all presentation logic stays
  Flet-free and headlessly tested.
- [x] **P9.7 — Add trace comparison.** Compare two traces of the same input
  specification — base versus edited revision, or two revisions — with
  per-node error metrics (max-abs, relative, cosine) rendered as a graph
  overlay, so quantization and pruning effects are visible at the node where
  they arise rather than only at the model outputs.
- [x] **P9.8 — Complete tracing tests.** Cover consent refusal, isolation and
  resource-limit enforcement, cancellation mid-trace, capture-budget
  eviction, symbolic shapes, custom domains and unsupported operators
  degrading to partial traces with diagnostics, seeded determinism across
  runs, visualization golden fixtures, trace comparison across revisions, and
  UI loading/partial/error/empty states.

**Exit gate:** A user-approved input traces through a supported ONNX model in
an isolated worker within the configured resource budgets; every captured
activation is inspectable and visualizable lazily through the same budgeted
store discipline as weights; traces of base and edited revisions compare
per node; unsupported executions are disclosed as unavailable rather than
approximated; and partial captures are always visibly partial.

## 6. Testing Strategy

### Fixture corpus

Maintain small, deterministic fixtures for fast tests and separately managed
large artifacts for performance tests:

- ONNX: embedded tensors, external tensors, symbolic shapes, custom ops,
  nested `If`/`Loop` graphs, CNN, transformer, and malformed artifacts.
- PyTorch: supported exported programs, FX edge cases, parameters/buffers,
  dynamic shapes, custom ops, and weights-only checkpoints.
- JAX/StableHLO: functions and regions, symbolic shapes, custom calls, nested
  checkpoint trees, and weights-only artifacts.
- Synthetic graphs: deep chains, wide fan-out, dense edges, repeated blocks,
  nested groups, and graphs far larger than the visible viewport.

Do not commit multi-gigabyte fixtures to normal Git history. Store their
content hashes, generation instructions, licenses, and expected results.

### Required test layers

| Layer | Required coverage |
|:--|:--|
| Unit | IR invariants, IDs, graph slicing, caches, patterns, commands, schemas |
| Property | Graph serialization, undo/redo, offset bounds, structural hashes |
| Golden | Import summaries, layouts where stable, no-op exports, export reports |
| Integration | Open → inspect → edit → validate → export → reopen |
| Security | Traversal, malformed offsets, archive bombs, unsafe code, isolation |
| Performance | Import reads, memory ceilings, render latency, cache pressure |
| Compatibility | Supported framework/runtime/browser/desktop version matrix |
| UI | Navigation, selection, semantic zoom, error/loading states, accessibility |

### Measurement rules

- Capture test hardware, operating system, browser, artifact hash, dependency
  versions, cold/warm cache state, and measurement method.
- Track p50 and p95 latency rather than only averages.
- Distinguish process memory, browser memory, mapped bytes, bytes read, cache
  capacity, and artifact size.
- Fail regression checks only against budgets approved in the Phase 0
  architecture decision; do not silently relax a budget.

## 7. Security and Data Integrity Checklist

These requirements apply throughout all phases:

- [x] Source artifacts remain immutable and are identified by a cryptographic
  content hash.
- [x] External tensor paths are normalized, confined to approved roots, and
  protected from traversal and link-based escapes.
- [x] Tensor offsets and lengths are bounds-checked with overflow-safe integer
  logic before reads or allocations.
- [x] Pickle and arbitrary Python execution are disabled in safe artifact mode.
- [ ] Trusted-code execution is isolated and requires explicit user consent.
- [ ] Long-running reads, layouts, statistics, validation, and execution jobs
  support cancellation and resource limits.
- [x] Temporary files and incomplete exports use atomic lifecycle rules and are
  recoverable or safely cleaned up.
- [x] Export reports retain artifact, tool, dependency, command, and revision
  provenance.
- [ ] Logs exclude model tensor contents, credentials, signed URLs, and other
  sensitive values.

This checklist is cross-cutting and is not included in the 76 task count; each
item must be covered by the relevant phase's implementation and tests.

## 8. Key Risks and Responses

| Risk | Early signal | Response |
|:--|:--|:--|
| Flet renderer cannot meet scale or parity needs | Phase 0 frame time, memory, or integration failure | Keep `GraphRenderer` replaceable; select custom control or hosted renderer from measured results |
| ONNX parsing still consumes too much memory | Import RSS or read instrumentation exceeds budget | Favor external data, compact indexes, streaming/range-aware parsing where the format permits |
| Graph layout dominates interaction time | Collapse/expand or navigation misses p95 budget | Cache hierarchy-aware layouts, compute asynchronously, patch locally, and aggregate edges |
| Block detection produces confusing groups | Low fixture precision or frequent user rejection | Use deterministic evidence, confidence, explanations, locks, and persistent corrections |
| Symbolic shapes prevent safe edits | Validation cannot prove compatibility | Disable the edit or require explicit constraints; never infer compatibility silently |
| Framework API churn breaks adapters | Compatibility test failures on new versions | Pin tested versions, isolate adapters, publish a matrix, and add migrations intentionally |
| Quantization export is not executable on target | Runtime validation or capability check fails | Make schemes target-specific and report unavailable combinations before commit |
| Web workloads exhaust resources | Queues, storage, or worker limits spike | Apply quotas, cancellation, isolation, expiring storage, and content-addressed deduplication |
| Unsafe artifacts trigger code or path access | Security tests or threat modeling reveal a path | Keep safe and trusted modes separate; validate before allocation or execution |
| Activation capture exhausts storage or memory | Capture store hits its byte budget on real models | Budgeted, evictable capture store; explicit node/value selection; partial-trace disclosure |

## 9. Release Milestones

| Milestone | Includes | Excludes |
|:--|:--|:--|
| **M0 — Architecture proven** | Phase 0 spikes and decisions | Product UI |
| **M1 — ONNX Viewer Alpha** | Phase 1 read-only vertical slice | Automatic grouping and edits |
| **M2 — Semantic Viewer Beta** | Phase 2 hierarchy and navigation | Model surgery |
| **M3 — ONNX Editor Beta** | Phases 3-4 inspection, revisions, constrained edits | Quantization/pruning and other frameworks |
| **M4 — Weight Surgery Release** | Phase 5 supported quantization/pruning | Universal runtime support |
| **M5 — Multi-artifact Release** | Phases 6-7 supported PyTorch and JAX artifacts | Original source reconstruction |
| **M6 — Hosted/Extensible Release** | Phase 8 hardened web deployment and plugins | Untrusted in-process extensions |
| **M7 — Inference Tracing Release** | Phase 9 consent-gated tracing, activation inspection, visualization, and trace comparison | Arbitrary runtime plugins; training-time introspection |

## 10. Definition of Done

A task is done only when:

1. Its behavior and failure modes match the relevant design document.
2. Tests cover the success path, malformed input, cancellation where relevant,
   and capability/error reporting.
3. Security, memory, and performance effects have been considered and measured
   where the task affects them.
4. User-visible behavior includes loading, empty, unsupported, error, and
   recovery states.
5. Documentation and the compatibility/capability matrix are updated.
6. Evidence is added to the progress log and the phase/task counts are updated.

A release is done only when its phase exit gates pass on the supported desktop
and web matrix, exported artifacts reopen successfully, and known fidelity or
capability limitations are visible to users.

## 11. Code Health Assessment and Improvement Backlog (2026-07-29)

A four-subsystem assessment (adapters/storage/IR, application/editing/
transformations/analysis, rendering/UI, tests/process) was run against this
plan on 2026-07-29. Like §7, this backlog is cross-cutting and is **not**
included in the 76-task count; items should be folded into the phase that
naturally owns them (most Tier 3 items are natural Phase 8 precursors).

### Verified state

- Quality gates on Windows 11 / Python 3.14.6: ruff clean, strict mypy clean,
  1,159 ordinary tests passed + 4 skipped + 2 performance tests deselected,
  2 performance tests passed in their isolated run, and 90.26%
  non-performance coverage. The wheel and source distribution build and pass
  strict metadata validation.
- The headline mechanisms are real, not scaffolding: the lazy ONNX indexer
  records byte ranges without reading payloads (proven by read
  instrumentation), the pickle scanner is a genuinely non-executing opcode
  interpreter, revisions are content-addressed and self-verifying with
  rollback-safe restore, and transformations commit through the same revision
  chain as hand edits. Zero TODO/FIXME markers across ~26,700 source lines.
- Layering is clean in the load-bearing directions: `analysis/` does not
  import `application/`, `application/` does not import `ui/`, and `flet`
  imports are confined to `rendering/` and focused `ui/` modules.

### Overall judgment

The ONNX core earns its claims. The 2026-07-29 implementation pass closed the
identified Phase 6-7 lifecycle and Tier 2 thread-safety gaps. StableHLO export
was deliberately declared unavailable rather than simulated; Phase 7 remains
open only for the explicitly isolated trusted-tracing flow in P7.4. The
desktop 1.0.0 surface now includes graph-first tracing, activation
visualization, lifecycle-aware open/close/save controls, test-input generation,
and opt-in Windows file-opening integration. This does not change the honest
status of Phase 8: multi-user web hardening and the public plugin API have not
started.

### Tier 1 — User-visible breakage in the multi-format path

Closed on 2026-07-29 with application- and UI-boundary regression tests.
Phase 7's remaining exit-gate work is P7.4, not an undocumented writer.

- [x] **Route export by artifact kind.** The Export button
  (`ui/app.py:~2861`) hardcodes ONNX export (`.onnx` filter,
  `export_revision`, which re-runs `index_model` on the source), so exporting
  a safetensors/PT2/StableHLO session fails with an ONNX parse error.
  `ModelSession.export_weights_only` (`application/session.py:468`) has no UI
  call site. Dispatch on artifact kind or export capability.
- [x] **Accept non-ONNX web uploads.** `application/session.py:688,756`
  reject any upload not ending in `.onnx` — an extension check that
  contradicts architectural rule 2 and blocks Phase 6-7 formats in web mode.
  Detect the staged bytes by content instead.
- [x] **Reconcile the validation pipeline with the capability registry.**
  `editing/validation.py:506` refuses any graph edit unless *export* is fully
  available, so PT2's declared partial editing and StableHLO's "constant
  edits are supported" (`ir/capabilities.py`) are unreachable, and the
  refusal message quotes the export reason for an editing refusal. Relax the
  precondition or narrow the declared capabilities — the contract and the
  code must agree.
- [x] **Make artifact-swap detection format-agnostic.**
  `storage/store.py:419` reads the expected hash from the `x-onnx.model`
  extension; for every other format it returns `None` and re-verification is
  silently skipped. Fall back to `document.source.content_hash`.
- [x] **Close two raw-exception leaks in `_read_document`.**
  `application/session.py:796` is the only dispatch branch without
  `SessionError` wrapping (malformed ONNX surfaces raw), and the FX fallback
  does not catch `FxError`.
- [x] **Fix or re-scope P7.5** (tracked in Phase 7 above).

### Tier 2 — Concurrency: locking has not kept pace with background jobs

- [x] **Serialize access to `RevisionChain` or make jobs snapshot-based.**
  The chain has no lock (`editing/revisions.py:401,461-475`), yet statistics
  jobs on pool threads read it (`application/session.py:319-330`) while the
  UI thread commits/undoes. A commit landing mid-read yields wrong bytes or
  `IndexError`. Cheapest fix: an `RLock`. Better: capture
  `(document, revision_id)` at job start — the immutable IR already supports
  it.
- [x] **Guard the `ModelSession.hierarchy` rebind.**
  `application/session.py:441-445` reassigns the controller on every graph
  edit while job threads read it — a silently stale render, not a crash.
- [x] **Guard `HierarchyOverrides` mutation.** Controllers mutate the shared
  overrides object outside the store's lock
  (`application/hierarchy.py:392` vs. the save iteration at `:189`); two
  sessions on one artifact can race into "dictionary changed size during
  iteration."
- [x] **Bound `_edited_stats` and narrow slicer invalidation.** The
  edited-statistics cache is unbounded (`application/session.py:169`; the
  `BudgetedCache` infrastructure already exists), and closing one session
  invalidates the shared layout cache for every session on the artifact
  (`session.py:562`).
- [x] **Narrow two risky handlers.** `editing/revisions.py:303-306` collapses
  every failure to "length unknown," silently changing bounds validation;
  `application/editing.py:103-107` quarantines the user's edit history on a
  transient I/O error as if it were corruption.

### Tier 3 — Architecture: propagate the pattern the repo already has

`GroupDetector` + `DetectionPipeline` (`analysis/hierarchy.py:97`,
`analysis/detectors.py:727`) is the correct extensibility pattern and already
exists in-tree. Three subsystems should copy it; all three are natural Phase 8
plugin-API precursors.

- [x] **Introduce an `ArtifactAdapter` protocol and registry** replacing the
  `if/elif` over `ArtifactKind` in `application/session.py:783`. This is why
  cancellation, scan limits, and read instrumentation are ONNX-only today
  (opening a 2 GB safetensors file is uncancellable), and it deletes six
  verbatim-duplicated `open_X` wrappers across the PyTorch/JAX adapters.
- [x] **Register `apply`/`to_json`/`from_json` handlers per command** and
  collapse the three parallel `isinstance` ladders in `editing/commands.py`
  (~420 lines; adding a command currently requires five synchronized edits
  that mypy cannot check). Resolved with typed `CommandDescriptor` entries
  holding target, summary, codec, and apply handlers.
- [x] **Turn `ValidationPipeline` into a rule registry and
  `TransformationEngine` into a handler map**, replacing the 224-line
  (`editing/validation.py:574`) and 4-branch (`transformations/engine.py:349`)
  `isinstance` chains. This finally gives the `backend_plugin` manifest field
  (declared, validated, never dispatched on) a purpose.
- [ ] **Break the `storage ↔ adapters.onnx` import cycle**
  (`storage/store.py:205,299` function-local imports) by injecting a
  materializer at document-construction time.
- [ ] **Create one dtype authority** (`ir/dtypes.py`): three width tables
  (`adapters/onnx/dtypes.py`, `adapters/pytorch/scalar_types.py`,
  `analysis/statistics.py`) disagree on which dtypes exist — a float8 tensor
  validates in the indexer but has no width in statistics — and the
  element-count product loop is copy-pasted eight times.
- [ ] **Unify the two node-search implementations.** `GraphSlicer.search`
  (`application/slices.py:180`) and `NavigationModel.search`
  (`application/navigation.py:100`) match different fields for the same
  user query and can return different results.

### Tier 4 — The UI monolith and dead code

- [x] **Decompose `ui/app.py`** (previously 3,813 lines, one class, a 673-line
  `__init__`, 78% coverage — the largest and worst-covered file in the repo).
  Mechanical, low-risk steps: extract panel builders from `__init__`; move
  pure logic (the transformation-request parser at `app.py:~2446`, capability
  report formatting, `_humanize_identifier`, `_compact_bytes`) into
  `viewmodel.py` where it becomes headlessly testable. Overview controls,
  shell layout, graph tracing, tensor helpers, and the 934-line test-input
  workspace now live in focused modules. Later Phase 9 and release-surface
  work grew the controller to 4,982 lines, so further decomposition remains a
  code-health opportunity even though the original seam-restoration task is
  complete.
- [x] **Honor the renderer seam.** The shell imports the concrete
  `ManagedCanvasRenderer` directly (violating the rule stated in
  `rendering/__init__.py`) and uses four off-contract members at 18 call
  sites, so `GraphRenderer` is not currently replaceable. Extend the
  contract, type the attribute as `GraphRenderer`, and inject a factory.
  Resolved with `InteractiveGraphRenderer`, a lazy production factory, and
  constructor injection; `ui/` has no concrete renderer import.
- [ ] **Collapse or delete the naive canvas adapter.** `flet_canvas.py`
  duplicates ~half of `flet_canvas_managed.py` verbatim (identical
  `apply_patch`, `replace_scene`, `set_selection`, `hit_test`, `_flush`) and
  survives only for a Phase 0 comparison ADR 0001 already decided; it also
  enforces no shape cap, so the two adapters silently diverge.
- [ ] **Decide the fate of ~1,000 lines of production-dead code:**
  `ir/serialize.py` (518 lines, zero production callers — either wire
  document caching into `application/persistence.py` for faster reopen, or
  quarantine it as spec-only), the superseded `adapters/onnx/splice.py`
  prototype (loads two full copies of the model into RAM), and
  `open_orbax_checkpoint` (unreachable: detection is single-file-only, so a
  checkpoint directory can never be opened).
- [ ] **Stream `write_safetensors`.** It accumulates every tensor's bytes in
  memory before writing (`adapters/pytorch/safetensors.py:328,354`) and is on
  the live `export_weights_only` path; offsets are computable from lengths
  alone.

### Tier 5 — Process guards so the above does not regrow

- [x] **Add Windows to the CI matrix.** CI is Ubuntu-only while Windows is a
  primary desktop target, a Windows-only defect already shipped once
  (`os.pread`, P0.4), and the two Job-Object tests never run in CI. The quality
  matrix now includes Windows latest on Python 3.14.
- [ ] **Tighten ruff:** add `C901`, `PLR0912`, `PLR0915`, and `BLE` with a
  short baseline ignore list. The current ruleset is exactly what lets a
  395-line undocumented function (`adapters/pytorch/pt2.py:241`) and 16
  blanket handlers pass clean. Highest value per minute on this list.
- [x] **Register a `performance` marker and exclude it from default runs.**
  The 10k wall-clock assertions run in every `pytest` invocation and fail
  under coverage load — the suite's one flake. Move `--cov-fail-under=90`
  to a CI-only invocation so narrowed local runs stop failing on coverage.
  Ordinary, coverage, and performance runs are now three explicit gates.
- [ ] **Merge the `test_fixes_*.py` files into their subsystem files before
  committing them.** Three naming schemes (by-module, by-phase,
  by-audit-batch) now coexist; once committed, the incident-keyed stratum
  becomes permanent. Keep the per-test "pins this fix" docstrings.
- [ ] **Add read instrumentation to the PyTorch/JAX adapters** so the
  laziness discipline proven for ONNX becomes enforceable for all formats
  (no PyTorch/JAX import test currently asserts payload bytes stayed
  untouched; StableHLO import reads the whole file into memory).

### Suggested sequencing

1. Land the verified Tier 1-2, registry, CI, and shell-seam changes before
   starting Phase 8 multi-user work.
2. Finish the remaining Tier 3 storage/dtype/search consolidation before
   exposing a public Phase 8 plugin API.
3. Resolve the remaining Tier 4 naive-renderer/dead-code/streaming items.
4. Add the remaining Tier 5 complexity and adapter-laziness process guards.
