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
    period_tensor = torch.as_tensor(period, dtype=value.dtype, device=value.device)
    return torch.remainder(value + period_tensor / 2.0, period_tensor) - period_tensor / 2.0


def _yaw_scale_np(yaw_scale, yaw_period, batch, dtype):
    scale = np.asarray(
        2.0 * np.pi / float(yaw_period) if yaw_scale is None else yaw_scale,
        dtype=dtype,
    )
    if scale.ndim == 0:
        return scale
    if scale.shape == (batch,):
        return scale[:, None]
    if scale.shape == (batch, 1):
        return scale
    raise ValueError(f"yaw_scale must be scalar or [{batch}], got {scale.shape}")


def _yaw_scale_torch(yaw_scale, yaw_period, value):
    if yaw_scale is None:
        return value.new_tensor(2.0 * np.pi / float(yaw_period))
    scale = torch.as_tensor(yaw_scale, dtype=value.dtype, device=value.device)
    if scale.ndim == 0:
        return scale
    if scale.shape == (value.shape[0],):
        return scale[:, None]
    if scale.shape == (value.shape[0], 1):
        return scale
    raise ValueError(
        f"yaw_scale must be scalar or [{value.shape[0]}], got {tuple(scale.shape)}"
    )


def encode_state_deltas_np(
    future,
    history,
    timestep_mask,
    history_timestep_mask,
    *,
    yaw_channel=4,
    yaw_period=2.0,
    yaw_scale=None,
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
    resolved_yaw_scale = _yaw_scale_np(
        yaw_scale, yaw_period, future.shape[0], future.dtype
    )
    resolved_yaw_period = 2.0 * np.pi / resolved_yaw_scale
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
                delta[:, yaw_channel, :], resolved_yaw_period
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
    yaw_scale=None,
    padding_value=PADDING_VALUE,
):
    """Invert :func:`encode_state_deltas_np` for model outputs."""
    _validate_shapes(deltas, history, timestep_mask, history_timestep_mask)
    timestep_mask = timestep_mask.bool()
    history_timestep_mask = history_timestep_mask.bool()

    decoded = torch.full_like(deltas, float(padding_value))
    resolved_yaw_scale = _yaw_scale_torch(
        yaw_scale, yaw_period, deltas[:, yaw_channel, 0, :]
    )
    resolved_yaw_period = 2.0 * torch.pi / resolved_yaw_scale
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
            current_yaw = _wrap_periodic_torch(
                current[:, yaw_channel, :], resolved_yaw_period
            )
            current = current.clone()
            current[:, yaw_channel, :] = current_yaw
        decoded[:, :, step, :] = torch.where(
            valid.unsqueeze(1), current, decoded[:, :, step, :]
        )
        previous = torch.where(valid.unsqueeze(1), current, previous)
        has_anchor = has_anchor | valid

    return decoded


def _last_history_state_and_motion_np(history, history_timestep_mask):
    """Return the last state and last observed per-frame position displacement."""
    batch, channels, steps, agents = history.shape
    last_state = np.zeros((batch, channels, agents), dtype=history.dtype)
    last_position = np.zeros((batch, 2, agents), dtype=history.dtype)
    last_index = np.zeros((batch, agents), dtype=np.int64)
    step_motion = np.zeros((batch, 2, agents), dtype=history.dtype)
    has_state = np.zeros((batch, agents), dtype=bool)

    for step in range(steps):
        valid = history_timestep_mask[:, step, :]
        has_previous = valid & has_state
        elapsed = np.maximum(step - last_index, 1).astype(history.dtype)
        displacement = (
            history[:, :2, step, :] - last_position
        ) / elapsed[:, None, :]
        step_motion = np.where(
            has_previous[:, None, :], displacement, step_motion
        )
        last_state = np.where(valid[:, None, :], history[:, :, step, :], last_state)
        last_position = np.where(
            valid[:, None, :], history[:, :2, step, :], last_position
        )
        last_index = np.where(valid, step, last_index)
        has_state |= valid

    return last_state, step_motion, has_state


def encode_constant_velocity_residual_np(
    future,
    history,
    timestep_mask,
    history_timestep_mask,
    *,
    yaw_channel=4,
    yaw_period=2.0,
    yaw_scale=None,
    padding_value=PADDING_VALUE,
):
    """Encode an absolute future as a history-conditioned motion residual.

    Positions are residuals around a constant-motion baseline extrapolated
    from the last two valid history positions. Position and velocity residuals
    are rotated into the last observed heading frame. Yaw is represented as a
    continuously accumulated residual from the last observed yaw.

    The transform consists only of translations and two 2-D rotations, so its
    absolute Jacobian determinant is exactly one. The model may therefore be
    trained on this representation without changing the original-data NLL.
    """
    future = np.asarray(future)
    history = np.asarray(history)
    timestep_mask = np.asarray(timestep_mask, dtype=bool)
    history_timestep_mask = np.asarray(history_timestep_mask, dtype=bool)
    _validate_shapes(future, history, timestep_mask, history_timestep_mask)
    if future.shape[1] < 5:
        raise ValueError("constant-velocity residual representation requires 5 channels")

    anchor, step_motion, has_anchor = _last_history_state_and_motion_np(
        history, history_timestep_mask
    )
    resolved_yaw_scale = _yaw_scale_np(
        yaw_scale, yaw_period, future.shape[0], future.dtype
    )
    resolved_yaw_period = 2.0 * np.pi / resolved_yaw_scale
    yaw0 = anchor[:, yaw_channel, :] * resolved_yaw_scale
    cos_yaw = np.cos(yaw0).astype(future.dtype, copy=False)
    sin_yaw = np.sin(yaw0).astype(future.dtype, copy=False)

    encoded = np.full_like(future, padding_value)
    accumulated_yaw = np.zeros_like(anchor[:, yaw_channel, :])
    previous_yaw = anchor[:, yaw_channel, :].copy()

    for step in range(future.shape[2]):
        valid = timestep_mask[:, step, :] & has_anchor
        current = future[:, :, step, :]
        baseline_xy = anchor[:, :2, :] + float(step + 1) * step_motion
        residual_xy = current[:, :2, :] - baseline_xy
        residual_v = current[:, 2:4, :] - anchor[:, 2:4, :]

        local_x = cos_yaw * residual_xy[:, 0, :] + sin_yaw * residual_xy[:, 1, :]
        local_y = -sin_yaw * residual_xy[:, 0, :] + cos_yaw * residual_xy[:, 1, :]
        local_vx = cos_yaw * residual_v[:, 0, :] + sin_yaw * residual_v[:, 1, :]
        local_vy = -sin_yaw * residual_v[:, 0, :] + cos_yaw * residual_v[:, 1, :]

        yaw_increment = _wrap_periodic_np(
            current[:, yaw_channel, :] - previous_yaw, resolved_yaw_period
        )
        accumulated_yaw = np.where(valid, accumulated_yaw + yaw_increment, accumulated_yaw)
        previous_yaw = np.where(valid, current[:, yaw_channel, :], previous_yaw)

        encoded[:, 0, step, :] = np.where(valid, local_x, padding_value)
        encoded[:, 1, step, :] = np.where(valid, local_y, padding_value)
        encoded[:, 2, step, :] = np.where(valid, local_vx, padding_value)
        encoded[:, 3, step, :] = np.where(valid, local_vy, padding_value)
        encoded[:, yaw_channel, step, :] = np.where(
            valid, accumulated_yaw, padding_value
        )

    return encoded


def _last_history_state_and_motion_torch(history, history_timestep_mask):
    batch, channels, steps, agents = history.shape
    last_state = torch.zeros(
        batch, channels, agents, dtype=history.dtype, device=history.device
    )
    last_position = torch.zeros(
        batch, 2, agents, dtype=history.dtype, device=history.device
    )
    last_index = torch.zeros(batch, agents, dtype=torch.long, device=history.device)
    step_motion = torch.zeros_like(last_position)
    has_state = torch.zeros(batch, agents, dtype=torch.bool, device=history.device)

    for step in range(steps):
        valid = history_timestep_mask[:, step, :]
        has_previous = valid & has_state
        elapsed = (step - last_index).clamp_min(1).to(history.dtype)
        displacement = (
            history[:, :2, step, :] - last_position
        ) / elapsed.unsqueeze(1)
        step_motion = torch.where(
            has_previous.unsqueeze(1), displacement, step_motion
        )
        last_state = torch.where(valid.unsqueeze(1), history[:, :, step, :], last_state)
        last_position = torch.where(
            valid.unsqueeze(1), history[:, :2, step, :], last_position
        )
        last_index = torch.where(valid, torch.full_like(last_index, step), last_index)
        has_state = has_state | valid

    return last_state, step_motion, has_state


def decode_constant_velocity_residual_torch(
    residuals,
    history,
    timestep_mask,
    history_timestep_mask,
    *,
    yaw_channel=4,
    yaw_period=2.0,
    yaw_scale=None,
    padding_value=PADDING_VALUE,
):
    """Invert :func:`encode_constant_velocity_residual_np`."""
    _validate_shapes(residuals, history, timestep_mask, history_timestep_mask)
    if residuals.shape[1] < 5:
        raise ValueError("constant-velocity residual representation requires 5 channels")
    timestep_mask = timestep_mask.bool()
    history_timestep_mask = history_timestep_mask.bool()

    anchor, step_motion, has_anchor = _last_history_state_and_motion_torch(
        history, history_timestep_mask
    )
    resolved_yaw_scale = _yaw_scale_torch(
        yaw_scale, yaw_period, residuals[:, yaw_channel, 0, :]
    )
    resolved_yaw_period = 2.0 * torch.pi / resolved_yaw_scale
    yaw0 = anchor[:, yaw_channel, :] * resolved_yaw_scale
    cos_yaw = torch.cos(yaw0)
    sin_yaw = torch.sin(yaw0)
    decoded = torch.full_like(residuals, float(padding_value))

    for step in range(residuals.shape[2]):
        valid = timestep_mask[:, step, :] & has_anchor
        local_xy = residuals[:, :2, step, :]
        local_v = residuals[:, 2:4, step, :]
        baseline_xy = anchor[:, :2, :] + float(step + 1) * step_motion

        world_x = cos_yaw * local_xy[:, 0, :] - sin_yaw * local_xy[:, 1, :]
        world_y = sin_yaw * local_xy[:, 0, :] + cos_yaw * local_xy[:, 1, :]
        world_vx = cos_yaw * local_v[:, 0, :] - sin_yaw * local_v[:, 1, :]
        world_vy = sin_yaw * local_v[:, 0, :] + cos_yaw * local_v[:, 1, :]

        current = torch.empty_like(residuals[:, :, step, :])
        current[:, 0, :] = baseline_xy[:, 0, :] + world_x
        current[:, 1, :] = baseline_xy[:, 1, :] + world_y
        current[:, 2, :] = anchor[:, 2, :] + world_vx
        current[:, 3, :] = anchor[:, 3, :] + world_vy
        current[:, yaw_channel, :] = _wrap_periodic_torch(
            anchor[:, yaw_channel, :] + residuals[:, yaw_channel, step, :],
            resolved_yaw_period,
        )
        decoded[:, :, step, :] = torch.where(
            valid.unsqueeze(1), current, decoded[:, :, step, :]
        )

    return decoded
