# Internal validations

These checks probe modeling and discretization assumptions. They are not paper
experiments and are not run by pytest or `reproduce_paper.sh`.

Run the deterministic reduced profile with

```bash
python -u experiments/validation/run_internal_validations.py --profile smoke
```

The `full` profile increases the grid and Monte Carlo sizes and may be
expensive, particularly when extended to the realistic no-continuation HJB
comparison:

```bash
python -u experiments/validation/run_internal_validations.py --profile full
```

Outputs are written under `results/internal_validation/<profile>/` by default.
