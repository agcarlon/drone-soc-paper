# Release provenance and verification

The five archives under [artifacts/paper_results](../artifacts/paper_results/)
contain the saved paper experiments. The run configurations record original
revision `f1ff89cba6c8d46b350503219450a8a59ff71c3f`. The
[results source snapshot](results-source-f1ff89c.tar.gz) contains the
implementation, experiment and plotting scripts, tests, and environment files
exported from that revision. The saved runs and data are distributed separately
in the five archives.

The v0.1 release tree contains the revised controller and updated manuscript
and plotting code. The explicit collision time scale was introduced in
original revision `674889d04a83042217fb57f216eedf6e15ee5ba9`. The
[controller diff](collision-time-scale.patch) shows the changes in the
runtime correction paths, default, and serialized configuration.

## Experiment environment

The release environment matches the archived versions of Python (3.12.13),
NumPy (2.4.6), SciPy (1.18.0), Matplotlib (3.11.0), and pandas (3.0.3).
These five version fields reproduce the environment fingerprint recorded in
the run configurations. The archives record these versions without the
original package builds or transitive dependencies. The release locks pin the
builds and dependencies used for validation on Linux x86-64. The historical
source snapshot retains its original files.

## Collision time scale

The runtime safeguard multiplies the collision-gradient correction by
`tau_col`, a time scale in seconds. The default and the paper's value are
both 1 s. Runtime planners project the nominal command to the speed limit,
apply the collision correction, and project the corrected command to the
speed limit. The pair distance is regularized as
`max(||x_i - x_j||, 1e-8 m)` in the gradient calculation. The relative
position is zero at exact coincidence, so the correction there is zero.

The [regression tests](../tests/test_review_revisions.py) compare against the
historical correction in five controller paths at separated, active-transition,
and coincident positions. They check speed saturation and compare commands
componentwise with relative and absolute tolerances of `1e-12`, with command
components expressed in m/s. These are constructed-state command checks.

With the environment from the repository README active, run from the repository
root:

```sh
MPLBACKEND=Agg python -B -m pytest -q -p no:cacheprovider \
  tests/test_review_revisions.py tests/test_drone_soc.py tests/test_infinite_horizon.py \
  tests/test_finite_horizon.py tests/test_greedy_eig_comparison.py \
  tests/test_hjb.py tests/test_fixed_point.py
```

The [historical test log](historical-pytest.log) records the original focused
verification. The [release test log](release-pytest.log) records the same
checks run from the v0.1 tree in the matched environment.

## Verify and use the saved data

Follow the [data retrieval instructions](../README.md#reproduce-the-paper).
The extraction script checks all five archives against
[SHA256SUMS](../artifacts/paper_results/SHA256SUMS).
Run `./reproduce_paper.sh --plots-only` to verify and extract the archives,
recompute the statistical tables, and regenerate the plots and manuscript.
