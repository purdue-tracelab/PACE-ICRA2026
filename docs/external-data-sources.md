# External data and parameter provenance

This register records the external evidence used by PACE code and identifies
values inherited from PACE or chosen for the model. Source links point to a
specific revision or edition where available. Keep the register and audit log
current under the [repository maintenance rules](../AGENTS.md).

## Standards used in the model

| Source and locator | Value used | Implementation |
| --- | --- | --- |
| [ITTF Statutes 2026, Law 2.3](https://documents.ittf.sport/sites/default/files/public/2026-02/2026_Statutes_v1_consolidated_clean.pdf#page=39) | Nominal diameter 40 mm and mass 2.7 g. The model uses matte white from the permitted appearances. | [Ball asset](../legged_lab/assets/table_tennis/ball.py) uses radius 0.020 m and mass 0.0027 kg. [TTEnv](../legged_lab/envs/base/tt_env.py) reads these from the asset for drag and geometric calculations. |
| [ITTF Statutes 2026, Laws 2.1.1 and 2.1.3](https://documents.ittf.sport/sites/default/files/public/2026-02/2026_Statutes_v1_consolidated_clean.pdf#page=39) | Tabletop height 0.760 m. A 300 mm drop should rebound about 230 mm on the playing surface. | [TTEnv](../legged_lab/envs/base/tt_env.py) adds the ball radius to 0.760 m for the predicted contact height. The drop specification is a comparison benchmark. |
| [ITTF Statutes 2026, Laws 2.4.1 and 2.4.3](https://documents.ittf.sport/sites/default/files/public/2026-02/2026_Statutes_v1_consolidated_clean.pdf#page=40) | The blade must be flat and rigid. A sponge covering including adhesive is less than 4.05 mm per playing face, subject to the applicable measurement tolerance. | The [allround paddle asset](../legged_lab/assets/booster/T1_TT/T1_TT_allround_rubber.usda) retains the rigid disk and uses a 3.3 mm nominal covering per face. The limit is a geometry constraint. The 3.3 mm value is a modeling assumption. |

### Standards consulted for comparison

- [ITTF M3 Ball Manual, October 2024](https://equipments.ittf.com/#/equipment/manuals),
  pp. 6, 13 and 14. A 305 mm drop onto the specified steel block should
  rebound 240-265 mm. The 50 N indentation limits are 0.68-0.81 mm at the poles
  for a seamed ball and 0.70-0.90 mm for a seamless ball. These characterize
  the approval test. PACE uses its existing contact settings for the
  ball-table and ball-racket pairs.
- [ITTF M4 Racket Coverings Manual, October 2024](https://sr.bttv.de/fileadmin/bttv/media/SR/pdf/M4_RC_MANUAL_OCTOBER_2024.pdf),
  pp. 10 and 16. The 50 mN friction test and its 0.50 threshold apply to
  outward-pimple and long-pimple coverings in the standards table. The
  inverted-rubber fit below uses measured ball impacts.

## Racket impact data and fit

Thomas Gossard's [racket bounce study, arXiv:2604.11349v1](https://arxiv.org/abs/2604.11349v1)
identifies the two configurations in Tables II-IV as andro Power 3 allround
inverted rubber with a 1.8 mm sponge. Racket 02 uses an offensive blade and
racket 04 uses a defensive blade. PACE uses this pair as a representative
allround reference and fits one shared set of coefficients for both faces.

The files below are pinned to dataset revision
[`159c6f040dd7832570b537b3b5fd00a97fc28c15`](https://github.com/gossardt/tt_rackets/commit/159c6f040dd7832570b537b3b5fd00a97fc28c15).
The [upstream repository includes an MIT license](https://github.com/gossardt/tt_rackets/blob/159c6f040dd7832570b537b3b5fd00a97fc28c15/LICENSE).
PACE stores the derived coefficients and links.

| Split and sample | Source file | Valid rows | Git blob SHA |
| --- | --- | ---: | --- |
| Training, racket 02 | [racket_02.csv](https://github.com/gossardt/tt_rackets/blob/159c6f040dd7832570b537b3b5fd00a97fc28c15/data/train/racket_02.csv) | 622 | `f3467213317e8133b3a3e75b33a8557976bff575` |
| Training, racket 04 | [racket_04.csv](https://github.com/gossardt/tt_rackets/blob/159c6f040dd7832570b537b3b5fd00a97fc28c15/data/train/racket_04.csv) | 660 | `2ea3e033ee257c82e089af72035aed113179199b` |
| Test, racket 02 | [racket_02.csv](https://github.com/gossardt/tt_rackets/blob/159c6f040dd7832570b537b3b5fd00a97fc28c15/data/test/racket_02.csv) | 156 | `6a1ea7f9626fad5927aa9a2ad4828c92b8243591` |
| Test, racket 04 | [racket_04.csv](https://github.com/gossardt/tt_rackets/blob/159c6f040dd7832570b537b3b5fd00a97fc28c15/data/test/racket_04.csv) | 165 | `63722ef320700354aab75531b029d5898ef8b2d7` |

### Reproducing the fit

1. Read the published CSV values in the racket frame. Keep rows with 14
   finite numeric fields, `v_in_z < 0` and `v_out_z > 0`. Pool the two
   training splits with equal weight per row. Reserve the two test splits for
   evaluation.
2. Fit normal restitution `e` by least squares through the origin for
   `v_out_z = e * (-v_in_z)`. The pooled result is `0.7008378486`.
3. Compute contact slip `s = v_in_x - r * spin_before_y` with the PACE radius
   `r = 0.020 m` and spin in rad/s. For candidate friction `mu` and
   tangential restitution `e_t`, predict tangential velocity with
   `v_out_x = v_in_x - sign(s) * min(0.4 * (1 + e_t) * abs(s), mu * (1 + e) * (-v_in_z))`.
   The factor 0.4 follows from the thin spherical shell inertia used by
   [the PACE impact model](../legged_lab/physics/ball_impact.py).
4. Minimize summed squared error in signed `v_out_x` over a friction grid
   from 0 to 1.5 in steps of 0.005. For the rigid fit, fix `e_t = 0`. For
   the offline elastic fit, also search `e_t` from 0 to 1 in steps of 0.02.
   Evaluate both fits on the 321 test rows using the fitted normal restitution.

| Fit | Normal restitution | Effective friction | Tangential restitution | Test tangential velocity RMSE |
| --- | ---: | ---: | ---: | ---: |
| Rigid contact approximation | 0.70 (rounded for use) | 0.445 | 0 (fixed) | 1.43 m/s |
| Offline elastic impulse | 0.70 (rounded for use) | 0.605 | 0.66 | 0.17 m/s |

### Mapping the data to PACE

- The [paddle asset](../legged_lab/assets/booster/T1_TT/T1_TT_allround_rubber.usda)
  authors 0.79 for both friction fields and 0.50 for material restitution.
  With the [PACE ball](../legged_lab/assets/table_tennis/ball.py) values of
  0.10 and 0.90, the average combine rule gives the fitted pair values of
  0.445 and 0.70. See the [PhysX combine rule](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/dev_guide/rigid_bodies_articulations/rigid_bodies.html#configure-rigid-body-material-combine-modes).
  Static and dynamic friction use the same value as a simplification.
- The nominal 3.3 mm total covering per face includes sponge, topsheet and
  adhesive. Only the 1.8 mm sponge thickness comes from the study. The
  14.5 mm collision disk adds the two assumed coverings to the existing
  7.9 mm disk. The link mass and inertia come from the existing asset.
- The elastic fit is available for offline use with `resolve_ball_impact`
  and documented in the [physics README](../legged_lab/physics/README.md).
  The live PhysX setup uses the rigid material approximation. The reported
  errors are from the analytical fit on published data. PhysX contact
  response and transfer to the PACE racket are still unverified in Isaac Sim.

## Inherited settings and analytical assumptions

| Value or behavior | Provenance and status |
| --- | --- |
| Active PACE air density 1.225 kg/m³ and drag coefficient 0.4378 | Inherited from the upstream [PACE training environment](https://github.com/ohowo/PACE-ICRA2026/blob/859b26b411d442704c7b590a94d54cbfb3ba9bb6/legged_lab/envs/base/tt_env.py#L205-L213). The upstream comment describes real-world testing. A source dataset or fit record for 0.4378 was unavailable in the reviewed branch. |
| Analytical drag constant 0.1248038406 m⁻¹ | Derived in [the force adapter](../legged_lab/physics/aerodynamics.py) as `rho * pi * r**2 * Cd / (2 * mass)`. Uses the values above and the ITTF nominal ball. The CPU test checks the constant against the force. [Conti et al., arXiv:2606.28805v2](https://arxiv.org/abs/2606.28805v2) provides context on speed and spin dependence, without supplying a coefficient to PACE. |
| Table material friction 0.40 and restitution 0.95 | Inherited from the [table USD physics layer](../legged_lab/assets/table_tennis/table/configuration/ping_pong_table_urdf_physics.usd). These are authored material inputs. |
| Analytical tabletop restitution 0.90 and horizontal velocity multiplier 0.70 | Inherited from the [PACE predictor](https://github.com/ohowo/PACE-ICRA2026/blob/859b26b411d442704c7b590a94d54cbfb3ba9bb6/legged_lab/envs/base/tt_env.py#L981-L1022). In the idealized vertical quadratic-drag calculation, 0.90 gives a 227.5 mm rebound from a 300 mm drop. That is a derived comparison, with the actual PhysX rebound pending verification. |
| Still air, zero initial serve spin, zero Magnus factor, and thin-shell inertia `I/m = 2r²/3` | Model defaults and an analytical geometry assumption. The optional airflow and spin settings accept values supplied by the caller. |

## Update audit log

| Date | Code or change reviewed | Provenance check |
| --- | --- | --- |
| 2026-11-04 | Baseline `77e8017`, covering [PR #2](https://github.com/ohowo/PACE-ICRA2026/pull/2), [PR #3](https://github.com/ohowo/PACE-ICRA2026/pull/3), [PR #4](https://github.com/ohowo/PACE-ICRA2026/pull/4) and [PR #5](https://github.com/ohowo/PACE-ICRA2026/pull/5) | Created the register, checked the four dataset blobs at the pinned revision and reran the coefficient fit and test-set evaluation. Recorded the standards, inherited values and remaining validation gaps. This change updates the documentation and maintenance rules. |
