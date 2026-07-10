# Current Goal

Repair the INTERACTION preprocessing, MapGlow model, and combined training pipeline; prove the fixes with unit and numerical tests; run 2,000 local optimization steps on real INTERACTION data with full validation; and deliver training/trajectory visualizations.

## Acceptance Criteria

- Prediction preprocessing derives context, coordinates, scaling, labels, and map selection from history only.
- Flow likelihood and inverse are mathematically consistent in train and eval modes, including masks.
- ActNorm and invertible convolutions remain nonsingular by construction.
- Training uses explicit context/loss masks, static agent/scene metadata, strict checkpoint loading, finite-gradient checks, validation metrics, and final checkpoints.
- Prediction models 30 future states as reversible deltas from the last valid history state, without model padding.
- Unit, reconstruction, Jacobian, integration, and archive-compatibility tests pass.
- A 2,000-step FP32 run completes locally on real INTERACTION scenes, evaluates the complete validation split, and emits checkpoints, metrics, samples, and visualizations.
