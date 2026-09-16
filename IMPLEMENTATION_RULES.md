# Implementation rules

This file is the repo-wide contract for implementing new experiments and features. Read it before changing code. The point is to keep the paper code auditable and prevent hidden defaults, duplicate execution paths, patch-local geometry, and other research-repo folklore from quietly changing results.

## 1. YAML is the single source of experiment behavior

- Paper-facing commands accept **one YAML path**. Do not add behavioral CLI flags such as `--lr`, `--k`, `--tau`, `--batch-size`, `--seed`, `--resume`, or `--checkpoint`.
- Runnable experiment YAMLs are **self-contained**. Do not make a paper experiment depend on a maze of inherited model/loss/dataset fragments.
- Unknown YAML keys are errors. Missing required keys are errors. Types are strict; for example, `"48"` is not accepted as integer `48`.
- Paper-facing Python functions should receive typed config objects rather than expose long lists of numerical keyword defaults. Internal pure functions may take ordinary arguments when those arguments are mathematical inputs, not hidden experiment policy.
- Do not rely on library constructor defaults for paper behavior. Optimizer, scheduler, model, loss, metric, checkpoint, and inference settings that affect results must be explicit in YAML and validated.
- Sweep overrides may modify only existing keys. A typo must fail before the first GPU run.
- Every run archives the source YAML, resolved YAML, environment metadata, and first-party code snapshot.

`CONFIG_POLICY.md` contains the strict configuration details.

## 2. Keep one current implementation path

- `inverse_diagrams/` is the current geometry/direct-inverse implementation.
- `core/`, `models/`, `engine/`, `losses/`, `metrics/`, `topology/`, and `cvlab/` provide the current CV/training framework.
- `old/` is frozen historical/reference/oracle code. New experiments must not depend on old solvers at runtime.
- Do not keep a second active implementation of the same experiment “just in case.” Remove obsolete entrypoints/config trees instead of leaving contradictory paths around.
- If an API is replaced, update scripts, configs, docs, and tests in the same change.

## 3. The representation is global even when computation is local

For a sample there is one global generator set

\[
\Theta = \{\theta_i\}_{i=1}^{K}.
\]

Do not create independent generator sets for image patches when measuring compact global representation or minimum generator count.

The binary diagram margin is

\[
F_\Theta(x)=\min_{j\in -}q_j(x)-\min_{i\in +}q_i(x).
\]

Foreground is defined by

\[
F_\Theta(x)\ge 0.
\]

Patching/chunking is an implementation strategy for evaluating this same global function. **Patch the computation, not the representation.**

## 4. Use true minima as the canonical decoder

- The canonical Power/GBPD decoder uses real `min`/`amin` competition within each class.
- Do not replace class minima with log-sum-exp/softmin in the main model. Softmin moves the zero boundary and changes the representation.
- The forward learned-segmentation path (`losses/diagram.py`, `models/diagram_models.py`)
  may train on

\[
P(x)=\sigma(F_\Theta(x)/\tau),
\]

but hard semantics are always determined by the sign of `F`.
- The direct-inverse path (`inverse_diagrams/inverse/losses.py`) does **not** form a
  probability. It differentiates a hinge on the raw margin `F` directly, so there is no
  temperature anywhere in it and no `sigmoid(F/tau)` helper on that side of the repo.
- Softmin can exist only as an explicitly named ablation with separate configs/results.

## 5. Keep Power and GBPD mathematically consistent

Power uses affine features:

- 2D: \(\eta(x,y)=(x,y,1)\)
- 3D: \(\eta(x,y,z)=(x,y,z,1)\)

GBPD uses quadratic features:

- 2D: \((x^2,xy,y^2,x,y,1)\)
- 3D: \((x^2,xy,xz,y^2,yz,z^2,x,y,z,1)\)

The implementation must preserve

\[
\theta_i^T\eta(x)=(x-p_i)^T M_i(x-p_i)-w_i
\]

when converting between coefficient and physical-generator forms.

Power must remain a tested special case of GBPD with \(M_i=I\).

GBPD optimization may use unconstrained quadratic coefficients, but physical export must apply the shared SPD gauge shift before interpreting matrices as \(M_i\succ0\).

## 6. 2D and 3D are first-class, not separate projects

- New geometry/runtime features should support both 2D and 3D unless a method is mathematically or dependency-limited to one dimension.
- Do not duplicate entire decoders for 2D and 3D. Put dimension-specific behavior in feature maps, coordinate generation, or small adapters.
- Physical voxel spacing must be preserved. Do not independently normalize axes in a way that destroys anisotropic spacing.
- For anisotropic volumes, choose patches/windows by physical field of view when practical, not blindly by equal voxel counts.
- If a topology metric/loss is only 2D, mark it explicitly and fail clearly in 3D rather than silently substituting something else.

## 7. Large masks: stream scores and coordinates

Design for native 2D masks around 1500x1500 and large 3D volumes without changing the public geometry API.

- Never make full `[N,K]` score tensors a required execution path.
- Generate coordinates on demand for spatial blocks when possible; do not require a persistent full `[N,d]` coordinate array for native rendering.
- Direct Power/GBPD rendering should use non-overlapping blocks because the diagram score has no receptive-field edge effect.
- If `K` becomes large, generator chunks may be added so memory scales with spatial-chunk-size times generator-chunk-size.
- Full native-resolution hard verification is mandatory even if optimization uses patches, subsampling, or lower-resolution stages.
- Do not claim constant-memory learned 3D training until the loss/backward path is genuinely streaming. `paper/CODEBASE_AUDIT.md` records current limitations.

## 8. Sliding windows and halos are for context-dependent operations

For CNNs, morphology, skeletonization, or local topology operators, use halo/core windows because output near a crop edge can depend on neighboring input.

The intended pattern is

\[
\text{window size} = \text{core size} + 2\times\text{halo}.
\]

Example 2D starting point:

- window: 512x512
- halo: 100x100
- core/stride: 312x312

For 3D, use the same idea but not necessarily the same literal size; examples such as 96^3 or 128^3 are more realistic.

Window utilities must handle:

- images smaller than the configured window,
- irregular final windows,
- exact one-write destination coverage,
- explicit model output spatial-shape validation,
- output dtype from the model rather than silent casting,
- batch/channel leading dimensions.

For direct diagram rendering, overlap is unnecessary. Use blocks.

## 9. Optimization patches may be local; verification and topology are global

For very large masks, direct inverse fitting may optimize on sampled blocks/patches. Prefer a mixture of:

- boundary regions,
- current hard errors,
- foreground samples,
- background samples,
- topology-critical thin structures/components when known.

Uniform random sampling alone is not sufficient for thin bridges, tunnels, vessels, tiny components, or holes.

Periodically render the whole native mask and refresh the hard-error set.

Local patch topology loss is only an auxiliary signal. It does **not** prove global connectivity, holes, or cavities. Global topology evaluation/loss must see the full object/domain or a justified global ROI/global reduced representation.

## 10. Losses must respect the geometric objective

The real representation objective is hard classification:

\[
y(x)F_\Theta(x)>0.
\]

The two paths that optimize `F` answer to this objective differently, and the rule is stated separately for each.

- **Forward learned segmentation** (`losses/diagram.py`): recommended baseline losses include margin/hinge (`DiagramMarginLoss`) and BCE on `sigmoid(F/tau)` (`DiagramBCEDiceLoss`). Dice and MSE may be ablated, but they are not substitutes for hard native verification.
- **Direct inverse** (`inverse_diagrams/inverse/losses.py`): the objective is a hinge on the true margin plus a term over the currently-wrong voxels,

\[
L=\operatorname{mean}_x\,\mathrm{relu}(m-y(x)F_\Theta(x))\;+\;\lambda\,\operatorname{mean}_{x\,\mathrm{wrong}}\,\mathrm{relu}(m-y(x)F_\Theta(x)),
\]

  with `m = decision_margin` and `λ = hard_weight`. BCE on `sigmoid(F/tau)` is **not** permitted here: measured on a target exactly representable at the `K` being fitted, it never reached `wrong_voxels == 0` over four restarts, because a mean over the raster gives the last misclassified voxel a vanishing share of the gradient. `paper/ABLATIONS.md` records the measurement. Dice remains a reported metric on this path, never an objective.

- Do not treat Dice=1 approximately as proof of exact reconstruction.
- `wrong_voxels == 0` is the exact raster success condition.
- MSE is allowed as an explicitly configured ablation, not a hidden replacement objective.
- Signed-distance supervision can be investigated, but do not assume `F` itself is a Euclidean signed-distance field.

## 11. Minimum-K claims must be empirical and auditable

- Report the smallest **observed exact** generator count as empirical \(K^*\).
- If no exact solution was found, `empirical_min_k` is `None`/NA. Never report the best approximate `K` as a minimum.
- Failure at one `K` is optimizer failure, not proof of mathematical impossibility.
- Do not stop the search merely because one lower `K` failed unless an explicitly configured research policy says so.
- Preserve every restart result, not only the winner.
- Report exact-success frequency over restarts as well as best error.
- Separate foreground/background split, seed, optimizer settings, and family in result tables.

A count may be called a **minimum** only against a proved lower bound. `mode: minimize`
reports \(L \le K_{\min} \le U\) with `CERTIFIED_OPTIMUM` iff \(L = U\); every other mode
reports \(U\) alone, which is an upper bound and must be written as one.

- A lower bound must be proved for the family actually being reported. The convex-cover
  bound holds for Power, where a cell is an intersection of half-spaces; it does **not**
  hold for anisotropic cells, and reusing it there would let a search stop above the true
  minimum and call it certified.
- \(L > U\) is a contradiction, not a close call. It raises.
- An anisotropic count is reported with its \(\Gamma\). With \(\kappa\) unbounded a cell
  becomes an arbitrarily thin sliver spanning the domain and the family has no smallest
  member, so \(K_{\min}(\Gamma)\) is the well-posed quantity and a count without its
  \(\Gamma\) is not comparable to anything.
- \(K_{\min}\) is non-increasing in \(\Gamma\); a *found* upper bound need not be, because
  a wider budget is a wider search space. Do not read non-monotone \(U\) as a violation.

The paper should distinguish:

1. representation/capacity failure,
2. optimization failure,
3. learned prediction failure.

Do not mix these into one number.

## 12. GBPD warm starts should preserve restart diversity

When GBPD is initialized from Power, do not fit many Power restarts, choose one winner, and perturb that same basin repeatedly.

Prefer a one-to-one or otherwise explicit mapping:

\[
\text{Power restart }r \rightarrow \text{GBPD restart }r.
\]

This preserves geometric basin diversity and makes restart statistics meaningful.

## 13. Viscosity is auxiliary softness, not geometry

Spatial viscosity is defined as a positive field shared by all generators at a location:

\[
P(x)=\sigma\left(\frac{F_\Theta(x)}{\nu(x)}\right),\qquad \nu(x)>0.
\]

Because \(\nu(x)>0\),

\[
P(x)\ge 0.5 \iff F_\Theta(x)\ge0.
\]

Therefore viscosity may change optimization confidence/gradient scale but must not change the hard generator-defined boundary.

Implementation rules:

- parameterize positivity, e.g. `nu = nu_min + softplus(raw_nu)`,
- start with fixed/annealed/global learned temperature before spatial viscosity,
- regularize spatial viscosity toward a simple field (for example on `log(nu)` and optionally with TV),
- treat viscosity as an auxiliary dense field, not part of the compressed generator count,
- do **not** use per-generator viscosity in the main model; it can change generator competition and secretly sculpt geometry,
- viscosity must be introduced as a controlled ablation after the baseline representation/optimization frontier is established.

Full rationale is in `paper/CONCEPTS_AND_METHODS.md`.

## 14. Topology is staged, dimension-aware, and separately evaluated

- Exact native raster reconstruction already preserves native raster topology under the same connectivity convention. Do not add topology loss merely to an already exact oracle reconstruction experiment.
- Add topology supervision when fitting approximately or predicting from images.
- In 2D, existing PH/Betti/TopoLoss tools may be used according to their documented capabilities.
- In 3D, do not pretend the current repo has a general differentiable Betti/PH loss. clDice is appropriate for tubular structures; connected-component metrics are available; general 3D PH remains a roadmap item.
- Do not infer global topology from independently correct local patches.
- Power has a special future direction: topology from the nerve of convex foreground cells, which may scale with `K` rather than raster size. GBPD does not inherit the same convex-good-cover guarantee automatically.

## 15. Keep dense baselines fair

Dense UNet baselines must use ordinary dense-segmentation semantics. Do not accidentally inherit diagram-specific temperature scaling, generator regularization, or geometry-only assumptions.

The baseline should answer whether compact diagram prediction is competitive with a conventional segmentation model, not whether the conventional model survives being forced through diagram hyperparameters.

## 16. Reproducibility beats convenience

For paper runs:

- seeds are explicit in YAML,
- dataset splitting has one explicit seed source,
- checkpoint selection is explicit,
- sweeps evaluate the configured checkpoint, not whichever weights remain after the last epoch,
- code/config/environment snapshots are archived,
- failed sweep cells are recorded and cause a non-zero sweep result,
- test data is not used for hyperparameter selection.

If behavior can change a result, it must be visible in the config or in versioned code, not in an undocumented default.

## 17. Tests are part of the implementation contract

New geometry/runtime changes should add tests that cover the relevant invariants. Important existing/required test patterns include:

- Power/GBPD feature dimensions in 2D and 3D,
- coefficient-to-physical score equivalence,
- Power embedded as GBPD,
- old exact decoder/oracle agreement where applicable,
- full render versus block render equivalence,
- sliding-window coverage equals exactly one write per destination element,
- identity reconstruction through the tiler,
- 2D and 3D coordinate-ramp tests to expose off-by-one shifts,
- images/volumes smaller than the configured window,
- irregular spatial sizes,
- strict YAML rejection of unknown/missing/wrongly typed values,
- sweep expansion validation before execution,
- repository-hygiene tests preventing removed stale paths from returning.

A code path is not “paper ready” because it imports. Run unit tests plus at least one end-to-end smoke path for the behavior being changed.

## 18. New research ideas must enter through an ablation, not a silent rewrite

When adding viscosity, topology, activity regularization, merge/refit, a new decoder, a different loss, or a new initialization:

1. preserve the existing baseline,
2. add explicit YAML fields/configs,
3. state the hypothesis in `paper/ROADMAP.md` or `paper/ABLATIONS.md`,
4. add the smallest tests needed for the invariant,
5. compare against the baseline using the same data/seeds/evaluation policy,
6. only promote it to the default after evidence justifies doing so.

Do not let an experimental idea silently redefine the canonical model.

## 19. Prefer simple, inspectable code over framework cleverness

- Keep the geometry core small enough to audit mathematically.
- Reuse the CV framework for training, datasets, metrics, callbacks, and backbones rather than duplicating those systems inside `inverse_diagrams`.
- Avoid custom CUDA/fused kernels until profiling demonstrates that blockwise PyTorch/GEMM is the bottleneck.
- Prefer explicit typed dataclasses/config schemas to magic dictionaries inside implementation code.
- Fail loudly when assumptions are violated. Silent coercion and silent fallback are not acceptable in paper code.

## 20. Before an expensive experiment

Run, at minimum:

```bash
pytest
python scripts/check_paper_setup.py
cvlab validate <experiment-or-sweep.yaml>
```

For a sweep, ensure every expanded cell validates before the first cell starts. For a new model/loss/runtime path, run a tiny end-to-end training or inference smoke test first.

The practical rule is simple: if future-you could reasonably forget a detail and obtain a different result, that detail belongs in YAML, a strict invariant, or this document.
