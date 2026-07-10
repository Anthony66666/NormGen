# Progress

- 2026-07-10: Audited the original model/training pipeline and reproduced train-mode non-invertibility from stochastic conditioner dropout.
- 2026-07-10: Completed forecasting-safe preprocessing with explicit temporal/agent masks and numeric normalization metadata; four focused leakage/archive tests pass.
- 2026-07-10: Repaired Gaussian tails, yaw units, deterministic conditioners, ActNorm, LU/dense invertible convolutions, exact channel masks, exogenous Coupling RoPE coordinates, and strict forward/reverse paths.
- 2026-07-10: Added reversible masked delta-state prediction, absolute-state sample decoding, and a 30-step no-padding prediction configuration.
- 2026-07-10: Converted scene-normalized history anchors back to metre-scale offsets before Coupling 2D RoPE; the complete 32-test suite, Python compilation, and diff checks pass.
- 2026-07-10: Trained the final FP32 configuration for 2,000 steps on 36,370 real forecasting-safe scenes and validated on all 11,794 validation scenes. Full validation: NLL/dim -6.731564, ADE 1.592654 m, FDE 3.638173 m, minADE6 1.374469 m, minFDE6 3.205546 m.
- 2026-07-10: Strictly reloaded `best.pt`; 54,685,323 model values and all optimizer tensors are finite with zero missing or unexpected keys.
