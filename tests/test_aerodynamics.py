# SPDX-License-Identifier: MIT

import math
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from legged_lab.physics.aerodynamics import AeroForceField


class AeroForceFieldTest(unittest.TestCase):
    def stage_forces(self, field, velocities, spins):
        ball = Mock()
        ball.data = SimpleNamespace(
            root_lin_vel_w=torch.tensor(velocities, dtype=torch.float32, device="cpu"),
            root_ang_vel_w=torch.tensor(spins, dtype=torch.float32, device="cpu"),
        )
        field.apply_to_rigid_object(ball)
        ball.set_external_force_and_torque.assert_called_once()
        staged = ball.set_external_force_and_torque.call_args.kwargs
        self.assertTrue(staged["is_global"])
        self.assertEqual(staged["forces"].shape, (len(velocities), 1, 3))
        torch.testing.assert_close(staged["torques"], torch.zeros_like(staged["forces"]))
        return staged["forces"][:, 0]

    def test_default_still_air_preserves_quadratic_drag(self):
        field = AeroForceField(device="cpu")
        forces = self.stage_forces(field, [[2, 0, 0], [4, 0, 0], [0, 0, 0]], [[0, 0, 5]] * 3)
        drag_scale = 0.5 * 1.20 * math.pi * 0.020**2 * 0.43
        expected = forces.new_tensor([[-4 * drag_scale, 0, 0], [-16 * drag_scale, 0, 0], [0, 0, 0]])
        torch.testing.assert_close(forces, expected)

    def test_drag_opposes_motion_relative_to_air(self):
        field = AeroForceField(device="cpu", air_velocity_w=(2, 0, 0))
        forces = self.stage_forces(field, [[0, 0, 0], [2, 0, 0], [4, 0, 0]], [[0, 0, 0]] * 3)
        drag = 0.5 * 1.20 * math.pi * 0.020**2 * 0.43 * 2**2
        torch.testing.assert_close(forces, forces.new_tensor([[drag, 0, 0], [0, 0, 0], [-drag, 0, 0]]))

    def test_magnus_uses_relative_velocity_and_reverses_with_spin(self):
        field = AeroForceField(device="cpu", drag_coeff=0.0, magnus_factor=1.0, air_velocity_w=(2, 0, 0))
        forces = self.stage_forces(
            field, [[4, 0, 0], [4, 0, 0], [2, 0, 0]], [[0, 0, 5], [0, 0, -5], [0, 0, 5]]
        )
        lift = 0.5 * 1.20 * math.pi * 0.020**2 * 0.020 * 5 * 2
        torch.testing.assert_close(forces, forces.new_tensor([[0, lift, 0], [0, -lift, 0], [0, 0, 0]]))

    def test_air_velocity_requires_three_components(self):
        with self.assertRaisesRegex(ValueError, "air_velocity_w"):
            AeroForceField(device="cpu", air_velocity_w=(2, 0))

    def test_drag_acceleration_coefficient_matches_world_frame_force(self):
        field = AeroForceField(device="cpu", radius_m=0.020, air_density=1.225, drag_coeff=0.4378)
        mass_kg = 0.0027
        k = field.drag_acceleration_coefficient(mass_kg)
        self.assertAlmostEqual(k, 0.1248038406)

        forces = self.stage_forces(field, [[3, -4, 12]], [[0, 0, 0]])
        expected = -k * 13 * forces.new_tensor([[3, -4, 12]])
        torch.testing.assert_close(forces / mass_kg, expected)
