
# Isaac Lab Ball Aerodynamics


GPU batched aerodynamic drag and Magnus forces for spinning balls in Isaac Lab. Implemented in PyTorch. Drop-in force field for `RigidObject` assets.


## Quickstart


```python
from legged_lab.physics.aerodynamics import AeroForceField
from isaaclab.assets.rigid_object import RigidObject

class DummyEnv:
    def __init__(self, sim, scene, ball: RigidObject):
        self.sim = sim
        self.scene = scene
        self.ball = ball

        # Initialize aerodynamics AFTER scene is built, BEFORE resets/buffers
        self.aero = AeroForceField(
            device="cuda:0",
            radius_m=0.020,
            air_density=1.225,
            drag_coeff=0.43,   # set 0.0 to disable
            magnus_factor=0.0
        )

    def step(self):
        # Call once per physics substep, BEFORE scene.write_data_to_sim()
        self.aero.apply_to_rigid_object(self.ball)
        self.scene.write_data_to_sim()
        self.sim.step(render=False)
        self.scene.update(dt=0.01)
```

### Example Ball Definition

The ball used in above example can be defined as follows:
```python
from isaaclab.assets.rigid_object import RigidObject, RigidObjectCfg
import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene

ball_cfg = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Ball",
    spawn=sim_utils.SphereCfg(
        radius=0.02,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.6, 0.2)),
    ),
)

scene = InteractiveScene(scene_cfg)     # scene_cfg includes ball_cfg
ball: RigidObject = scene["ball"]       # handle used by AeroForceField
```


## Features

* Drag and Magnus forces in world frame
* GPU batched PyTorch compute
* Works with `RigidObject` or collections
* Pluggable model API for custom aero



# Aerodynamic Models

Math and usage for the batched aerodynamic models used by the adapter in `aerodynamics.py`.

## Model

Let $\mathbf{v}\in\mathbb{R}^3$ be world linear velocity, $\boldsymbol{\omega}\in\mathbb{R}^3$ world angular velocity, $\rho$ air density, $A=\pi r^2$ area, $r$ radius, $C_d$ drag coefficient, $\kappa$ Magnus factor.

$$
\mathbf{F}_\text{drag} = -\tfrac{1}{2}\rho A C_d \|\mathbf{v}\| \mathbf{v}, \quad
\mathbf{F}_\text{magnus} = \tfrac{1}{2}\rho A r \kappa (\boldsymbol{\omega}\times \mathbf{v})
$$
$$
\mathbf{F}_\text{total}=\mathbf{F}_\text{drag}+\mathbf{F}_\text{magnus},\quad \boldsymbol{\tau}=\mathbf{0}
$$

Shapes: `(N, 3)` for forces and torques. World frame.


## Adapter usage

```python
from legged_lab.physics.aerodynamics import AeroForceField

aero = AeroForceField(
    device="cuda:0",
    radius_m=0.020,
    air_density=1.225,
    drag_coeff=0.47,
    magnus_factor=0.0
)

# Called once per substep before staging to the sim
aero.apply_to_rigid_object(ball_asset)   # ball_asset: isaaclab.assets.RigidObject
F_w = aero.last_forces_w    # (N, 3)
T_w = aero.last_torques_w   # (N, 3), zero in this model
```

## Where to place calls in an env

* Initialize the adapter **after** the scene is created and assets are resolved.
* Initialize **before** buffer allocation and any `reset(...)` calls.
* In the step loop, call `apply_to_rigid_object(...)` **before** `scene.write_data_to_sim()` on each physics substep.


## Quick example: defining `self.ball`


```python
from isaaclab.assets.rigid_object import RigidObject, RigidObjectCfg
import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene

ball_cfg = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Ball",
    spawn=sim_utils.SphereCfg(
        radius=0.02,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
    ),
)

scene = InteractiveScene(scene_cfg)            # scene_cfg includes ball_cfg
self.scene = scene
self.ball: RigidObject = self.scene["ball"]    # RigidObject handle used by the aero adapter

```


## Direct model API


```python
import torch
from legged_lab.physics.aero_model import MeasuredAeroModel

model = MeasuredAeroModel(device="cuda:0")
v = torch.randn(1024, 3, device="cuda")
w = torch.randn(1024, 3, device="cuda")
ball_props = {"radii": torch.full((1024,), 0.02, device="cuda"),
              "areas": torch.full((1024,), 3.1415926*0.02**2, device="cuda")}
env_props = {"air_density": 1.225}
params = {"drag_coefficients": torch.full((1024,), 0.43, device="cuda"),
          "magnus_factors":    torch.full((1024,), 0.0,  device="cuda")}
F, T = model.calculate_forces_and_torques(v, w, ball_props, env_props, params)
```


## Tuning

* `drag_coeff`: fit from deceleration vs. speed; typical spheres \~0.4–0.5.
* `magnus_factor`: fit from lateral deflection vs. spin; `0.0` disables Magnus.
* Use SI units and world-frame velocities.

## Match ball and rebound reference

The `t1_tt` ball uses the ITTF nominal diameter of 40 mm, mass of 2.7 g,
and a matte white appearance. The simulation uses quadratic vector drag with
the existing PACE values of 1.225 kg/m³ for air density and 0.4378 for `Cd`.
The analytical predictor now gets `k = ρ A Cd / (2m)` from those same inputs
and the configured ball mass, giving 0.124804 m⁻¹. Its closed-form flight
estimate still treats the axes separately. The constant `Cd` is a PACE model
parameter. Experimental drag can depend on speed and spin.

ITTF Law 2.1.3 specifies about 230 mm rebound from a 300 mm drop onto the
playing surface. In the one-dimensional quadratic-drag model above, that
corresponds to a normal restitution of about 0.905. The existing analytical
table value of 0.90 gives about 228 mm in that model. PhysX resolves contact
using the rigid sphere and the contact materials.

The separate ITTF M3 ball approval test specifies a 240–265 mm rebound from a
305 mm drop onto a standard steel block. M3 also specifies a slow 50 N
indentation test, with 0.68–0.81 mm at the poles for a seamed ball and
0.70–0.90 mm for a seamless ball. These test conditions characterize ball
hardness and steel-block rebound. Dynamic stiffness, damping, and tangential
restitution for a particular table or racket rubber require separate data.

[ITTF Laws 2.1 and 2.3](https://documents.ittf.sport/sites/default/files/public/2026-02/2026_Statutes_v1_consolidated_clean.pdf) · [ITTF M3 Ball Manual](https://equipments.ittf.com/#/equipment/manuals) · [Measured speed and spin dependence](https://arxiv.org/abs/2606.28805)

## Symmetric allround racket rubber

Both faces of the `t1_tt` racket use the same allround inverted-rubber
approximation. The reference is andro Power 3 with a 1.8 mm sponge on the two
blades used in the published racket 02 and 04 experiments. The collision
geometry adds a nominal 3.3 mm per face, including sponge, topsheet and adhesive,
as a thickness assumption. The resulting 14.5 mm cylinder uses a single contact
material, including its rim. The original link mass and inertia are retained.

A constant fit to the two published training sets gives about 0.70 normal
restitution. With tangential restitution fixed at zero, fitting the outgoing
tangential velocity gives 0.445 effective friction. The authored PhysX material
values are 0.79 friction and 0.50 restitution to obtain the fitted contact values
when averaged with the PACE ball (0.10 friction and 0.90 restitution).
Startup randomization retains this paddle material and samples the other robot
shapes, including the forearm collider on the same link.

For offline predictions with `resolve_ball_impact`, fitting with tangential
elasticity enabled gives 0.605 friction and 0.66 tangential restitution, with
the same 0.70 normal restitution. On the held-out data, the tangential-speed
RMSE was 1.43 m/s for the rigid impulse fit and 0.17 m/s with tangential
elasticity. These are fits to the published experiments. The live PhysX
material uses rigid contact and Coulomb friction.

[Racket and rubber experiments](https://arxiv.org/abs/2604.11349) · [Public training and test data](https://github.com/gossardt/tt_rackets/tree/main/data)
