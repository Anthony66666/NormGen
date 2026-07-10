"""Invertible trajectory representations used by prediction-mode training."""

from __future__ import annotations

import numpy as np
import torch


PADDING_VALUE = -1.0


def _validate_shapes(states, history, timestep_mask, history_timestep_mask):
    if states.ndim != 4 or history.ndim != 4:
        raise ValueError("states and history must have shape [B,C,T,V]")
    if states.shape[:2] != history.shape[:2] or states.shape[3] != history.shape[3]:
        raise ValueError(
            "states and history must have matching batch, channel, and agent dimensions"
        )
    expected_mask = (states.shape[0], states.shape[2], states.shape[3])
    expected_history_mask = (history.shape[0], history.shape[2], history.shape[3])
    if tuple(timestep_mask.shape) != expected_mask:
        raise ValueError(
            f"timestep_mask must have shape {expected_mask}, got {tuple(timestep_mask.shape)}"
        )
    if tuple(history_timestep_mask.shape) != expected_history_mask:
        raise ValueError(
            "history_timestep_mask must have shape "
            f"{expected_history_mask}, got {tuple(history_timestep_mask.shape)}"
        )


def _wrap_periodic_np(value, period):
    return np.remainder(value + period / 2.0, period) - period / 2.0


def _wrap_periodic_torch(value, period):
    period_tensor = value.new_tensor(float(period))
    return torch.remainder(value + period_tensor / 2.0, period_tensor) - period_tensor / 2.0


def encode_state_deltas_np(
    future,
    history,
    timestep_mask,
    history_timestep_mask,
    *,
    yaw_channel=4,
    yaw_period=2.0,
    padding_value=PADDING_VALUE,
):
    """Encode future states as increments from the last observable state.

    Missing future observations do not move the running anchor. If an agent
    reappears later, its next delta is measured from its last valid state.
    """
    future = np.asarray(future)
    history = np.asarray(history)
    timestep_mask = np.asarray(timestep_mask, dtype=bool)
    history_timestep_mask = np.asarray(history_timestep_mask, dtype=bool)
    _validate_shapes(future, history, timestep_mask, history_timestep_mask)

    encoded = np.full_like(future, padding_value)
    previous = np.zeros(
        (future.shape[0], future.shape[1], future.shape[3]), dtype=future.dtype
    )
    has_anchor = np.zeros((future.shape[0], future.shape[3]), dtype=bool)

    for step in range(history.shape[2]):
        valid = history_timestep_mask[:, step, :]
        previous = np.where(valid[:, None, :], history[:, :, step, :], previous)
        has_anchor |= valid

    for step in range(future.shape[2]):
        valid = timestep_mask[:, step, :] & has_anchor
        current = future[:, :, step, :]
        delta = current - previous
        if yaw_channel is not None and yaw_channel < future.shape[1]:
            delta[:, yaw_channel, :] = _wrap_periodic_np(
                delta[:, yaw_channel, :], float(yaw_period)
            )
        encoded[:, :, step, :] = np.where(valid[:, None, :], delta, padding_value)
        previous = np.where(valid[:, None, :], current, previous)
        has_anchor |= valid

    return encoded


def decode_state_deltas_torch(
    deltas,
    history,
    timestep_mask,
    history_timestep_mask,
    *,
    yaw_channel=4,
    yaw_period=2.0,
    padding_value=PADDING_VALUE,
):
    """Invert :func:`encode_state_deltas_np` for model outputs."""
    _validate_shapes(deltas, history, timestep_mask, history_timestep_mask)
    timestep_mask = timestep_mask.bool()
    history_timestep_mask = history_timestep_mask.bool()

    decoded = torch.full_like(deltas, float(padding_value))
    previous = torch.zeros(
        deltas.shape[0], deltas.shape[1], deltas.shape[3],
        dtype=deltas.dtype, device=deltas.device,
    )
    has_anchor = torch.zeros(
        deltas.shape[0], deltas.shape[3], dtype=torch.bool, device=deltas.device
    )

    for step in range(history.shape[2]):
        valid = history_timestep_mask[:, step, :]
        previous = torch.where(valid.unsqueeze(1), history[:, :, step, :], previous)
        has_anchor = has_anchor | valid

    for step in range(deltas.shape[2]):
        valid = timestep_mask[:, step, :] & has_anchor
        current = previous + deltas[:, :, step, :]
        if yaw_channel is not None and yaw_channel < deltas.shape[1]:
            current_yaw = _wrap_periodic_torch(current[:, yaw_channel, :], yaw_period)
            current = current.clone()
            current[:, yaw_channel, :] = current_yaw
        decoded[:, :, step, :] = torch.where(
            valid.unsqueeze(1), current, decoded[:, :, step, :]
        )
        previous = torch.where(valid.unsqueeze(1), current, previous)
        has_anchor = has_anchor | valid

    return decoded
