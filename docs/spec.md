# Current Goal

Build a separate query-centric LaneGraph route-mixture MapGlow, regenerate forecasting-safe v5 INTERACTION data with explicit lane topology, prove the exact mixture likelihood and proposal mathematics, and complete a 2,000-step real-data validation run without changing the existing v2 experiment.

## Acceptance Criteria

- Prediction preprocessing derives context, coordinates, scaling, labels, and map selection from history only.
- Flow likelihood and inverse are mathematically consistent in train and eval modes, including masks.
- ActNorm and invertible convolutions remain nonsingular by construction.
- Training uses explicit context/loss masks, static agent/scene metadata, strict checkpoint loading, finite-gradient checks, validation metrics, and final checkpoints.
- Prediction models 30 future states as reversible deltas from the last valid history state, without model padding.
- Unit, reconstruction, Jacobian, integration, and archive-compatibility tests pass.
- A 2,000-step FP32 run completes locally on real INTERACTION scenes, evaluates the complete validation split, and emits checkpoints, metrics, samples, and visualizations.
- Six explicit route proposals form an exact normalized scene mixture, use topology/history only, and are sampled deterministically by component for validation coverage.
