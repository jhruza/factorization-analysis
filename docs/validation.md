# Validation

CPU checks completed on 16 November 2026:

- 26 tests pass, including signed-state filtering, causal masking, factor maps, whole-history controls, paper PCA, exact resume, and freeze branches.
- The source generator and oracle agree within `8e-8` on retained reference histories.
- Seeded transformer initialization and tested intermediate activations match the reference exactly; logits agree within `4.8e-7`.
- The smoke workflow runs through Pixi: training, both analyses, figures, and four branches. All 100 tested native-hybrid controls pass.

Run `pixi run test` and `pixi run lint`. The standalone generator check is `pixi run python scripts/check_reference_process.py`. The optional model-reference script requires TransformerLens and the original checkpoint; ordinary training and analysis require neither.

Reports: [validation.json](../scripts/validation.json), [model comparison](../scripts/reference_model_check.json), and [generator fixture](../scripts/reference_process_check.json). Reference implementation: [Astera-org/factored-reps](https://github.com/Astera-org/factored-reps), commit `1dd7a8e`.

GPU execution and a complete paper-budget training run have not been tested. Smoke results verify the software pipeline, not learned factorization.
