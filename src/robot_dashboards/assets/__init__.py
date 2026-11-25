"""Sample assets shipped with the package.

- ``ur5e/``: the UR5e collision model (ur_description, BSD-3-Clause, LICENSE.ur_description beside it).
- ``data/workspace-dev-029.npz``: a real 3.2 s UR5e multi-waypoint trajectory (trajectory_tracking_agent's
  ur5e_alone/workspace/v2 dev task 029): ``t``, ``names``, ``q``/``qd``/``qdd`` (rad), ``tool_xyz`` (m),
  ``tool_rot`` (3x3), joint limits ``lo``/``hi``/``vmax``/``amax``.
"""

from pathlib import Path

ASSETS = Path(__file__).parent
UR5E_URDF = ASSETS / "ur5e" / "ur5e_collision.urdf"
UR5E_TRAJECTORY = ASSETS / "data" / "workspace-dev-029.npz"
