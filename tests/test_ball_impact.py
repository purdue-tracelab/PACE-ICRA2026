# SPDX-License-Identifier: MIT

import unittest

import torch

from legged_lab.physics.ball_impact import resolve_ball_impact


def vector(values):
    return torch.tensor(values, dtype=torch.float64, device="cpu")


class BallImpactTest(unittest.TestCase):
    def test_frictionless_table_rebounds_only_along_normal(self):
        spin = vector([10, 20, 30])
        velocity, outgoing_spin = resolve_ball_impact(
            vector([3, 1, -4]), spin, vector([0, 0, 1]), radius=0.02, restitution=0.75, friction=0.0
        )
        torch.testing.assert_close(velocity, vector([3, 1, 3]))
        torch.testing.assert_close(outgoing_spin, spin)

    def test_opposite_spin_deflects_table_bounce_in_opposite_directions(self):
        velocity, spin = resolve_ball_impact(
            vector([[0, 0, -4], [0, 0, -4]]),
            vector([[0, 100, 30], [0, -100, 30]]),
            vector([0, 0, 1]),
            radius=0.02,
            restitution=0.5,
            friction=1.0,
        )
        torch.testing.assert_close(velocity, vector([[0.8, 0, 2], [-0.8, 0, 2]]))
        torch.testing.assert_close(spin, vector([[0, 40, 30], [0, -40, 30]]))

    def test_sliding_impulse_is_limited_by_friction(self):
        velocity, spin = resolve_ball_impact(
            vector([4, 0, -1]), vector([0, 0, 0]), vector([0, 0, 1]),
            radius=0.02, restitution=0.5, friction=0.2, tangential_restitution=1.0,
        )
        torch.testing.assert_close(velocity, vector([3.7, 0, 0.5]))
        torch.testing.assert_close(spin, vector([0, 22.5, 0]))

    def test_moving_racket_transfers_speed_and_spin(self):
        velocity, spin = resolve_ball_impact(
            vector([-2, 0, 0]), vector([0, 0, 0]), vector([1, 0, 0]),
            radius=0.02, restitution=0.5, friction=1.0,
            surface_velocity=vector([1, 2, 0]),
        )
        torch.testing.assert_close(velocity, vector([2.5, 0.8, 0]))
        torch.testing.assert_close(spin, vector([0, 0, -60]))

    def test_tangential_elastic_recovery_can_reverse_slip_and_spin(self):
        incoming_spin = vector([0, 0, -20])
        normal = vector([1, 0, 0])
        velocity, spin = resolve_ball_impact(
            vector([-3, 2, 0]), incoming_spin, normal,
            radius=0.02, restitution=0.6, friction=1.0, tangential_restitution=0.5,
        )
        contact_velocity = velocity + torch.linalg.cross(spin, -0.02 * normal)
        torch.testing.assert_close(contact_velocity, vector([1.8, -1.2, 0]))
        self.assertGreater(spin[2].item(), 0)

    def test_separating_grazing_and_no_slip_contacts(self):
        incoming = vector([[3, 0, 2], [3, 0, 0], [2, 0, -4]])
        incoming_spin = vector([[0, 100, 0]] * 3)
        velocity, spin = resolve_ball_impact(
            incoming, incoming_spin, vector([0, 0, 1]), radius=0.02, restitution=0.5, friction=1.0
        )
        torch.testing.assert_close(velocity, vector([[3, 0, 2], [3, 0, 0], [2, 0, 2]]))
        torch.testing.assert_close(spin, incoming_spin)

    def test_invalid_contact_parameters(self):
        for invalid in ({"radius": 0}, {"friction": -0.1}, {"restitution": 1.1}, {"tangential_restitution": -0.1}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                params = {"radius": 0.02, "friction": 0.5, "restitution": 0.8} | invalid
                resolve_ball_impact(vector([0, 0, -1]), vector([0, 0, 0]), vector([0, 0, 1]), **params)
