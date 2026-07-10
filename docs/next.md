# Next Actions

- No blocking work remains for the requested FP32 repair and 2,000-step validation.
- Optional: fix the RoPE SDPA attention-mask dtype and add CUDA FP16/BF16 regression tests before enabling AMP.
- Optional: continue the validated checkpoint to one or more complete epochs and monitor held-out ADE/FDE for overfitting.
- Optional: compare delta/no-padding tail displacement statistics with the earlier absolute/padded baseline as a research ablation.
