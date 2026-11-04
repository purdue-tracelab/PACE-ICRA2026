# SPDX-License-Identifier: MIT

import math
from typing import Optional

import torch


def resolve_ball_impact(
    linear_velocity: torch.Tensor,
    angular_velocity: torch.Tensor,
    normal: torch.Tensor,
    *,
    radius: float,
    restitution: float,
    friction: float,
    tangential_restitution: float = 0.0,
    surface_velocity: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the world-frame linear and angular velocities after one sphere impact.

    Ball velocities have shape (..., 3) in m/s and rad/s. The unit normal points
    from the surface toward the ball. The normal and optional surface velocity
    at the contact point broadcast to the ball velocity shape. Omit the surface
    velocity for a stationary table. For a racket, include its local surface
    motion at the point of impact. Surface motion is prescribed.

    The ball is a thin spherical shell. Normal restitution describes rebound
    after compression. Tangential restitution describes elastic recovery in
    shear, with zero targeting no slip and one targeting full slip reversal.
    Friction caps the tangential impulse at mu times the normal impulse.
    This instantaneous model assumes the caller has located the contact.
    Separating and grazing contacts leave both velocities unchanged.
    """
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError("radius must be positive and finite")
    if not math.isfinite(friction) or friction < 0:
        raise ValueError("friction must be nonnegative and finite")
    if not (0 <= restitution <= 1 and 0 <= tangential_restitution <= 1):
        raise ValueError("normal and tangential restitution must be in [0, 1]")

    normal = normal.expand_as(linear_velocity)
    contact_arm = -radius * normal
    relative_velocity = linear_velocity if surface_velocity is None else linear_velocity - surface_velocity
    normal_speed = (relative_velocity * normal).sum(dim=-1, keepdim=True)
    normal_delta_v = -(1 + restitution) * normal_speed.clamp(max=0)

    slip = relative_velocity - normal_speed * normal + torch.linalg.cross(angular_velocity, contact_arm)
    slip_speed = torch.linalg.vector_norm(slip, dim=-1, keepdim=True)
    safe_slip_speed = torch.where(slip_speed > 0, slip_speed, 1.0)

    # I / m = 2r²/3 for a thin shell. Express impulse per unit ball mass.
    inertia_per_mass = (2.0 / 3.0) * radius**2
    slip_gain = (1 + tangential_restitution) / (1 + radius**2 / inertia_per_mass)
    slip_gain_with_friction = (friction * normal_delta_v / safe_slip_speed).clamp(max=slip_gain)
    tangential_delta_v = -slip_gain_with_friction * slip

    return (
        linear_velocity + normal_delta_v * normal + tangential_delta_v,
        angular_velocity + torch.linalg.cross(contact_arm, tangential_delta_v) / inertia_per_mass,
    )
