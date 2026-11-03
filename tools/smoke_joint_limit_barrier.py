# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU one-step regression smoke for the opt-in deploy joint-limit objective.

Run from this checkout with an existing matching uv environment, for example:
  uv run --no-sync python tools/smoke_joint_limit_barrier.py
"""
import json
from pathlib import Path

import numpy as np
import warp as wp
import newton.ik as ik
from newton._src.sim.ik.ik_common import IKJacobianType

from soma_retargeter.io.bvh import load_bvh
from soma_retargeter.pipelines.soma_retargeting_pipeline import SomaRetargetingPipeline
from soma_retargeter.pipelines.deploy_joint_limits import DEPLOY_LIMITS
from soma_retargeter.pipelines.ik_objectives import IKJointLimitBarrier


def check_barrier_derivative(model, pipeline):
    """Measure the exact hinge values and Jacobian, including zero inside bounds."""
    obj = IKJointLimitBarrier(pipeline.deploy_joint_limit_lower,
                              pipeline.deploy_joint_limit_upper,
                              pipeline.deploy_joint_limit_margin, weight=40.0)
    n_dofs = model.joint_dof_count
    obj.set_batch_layout(n_dofs, 0, 1)
    obj.bind_device(wp.get_device())
    obj.init_buffers(model, IKJacobianType.ANALYTIC)
    dof = next(i for i, lo in enumerate(pipeline.deploy_joint_limit_lower.numpy())
               if abs(lo - DEPLOY_LIMITS["unitree_g1"]["left_ankle_pitch_joint"][0]) < 1e-6)
    coord = obj.dof_to_coord.numpy()[dof]
    lo = pipeline.deploy_joint_limit_lower.numpy()[dof]
    margin = pipeline.deploy_joint_limit_margin.numpy()[dof]
    for q, expected, expected_jac in (
        (lo + margin + 0.01, 0.0, 0.0),
        (lo, -40.0 * margin, 40.0),
        (lo - 0.01, -40.0 * (margin + 0.01), 40.0),
    ):
        values = model.joint_q.numpy().copy().reshape(1, -1)
        values[0, coord] = q
        joint_q = wp.array(values, dtype=wp.float32)
        residual = wp.zeros((1, n_dofs), dtype=wp.float32)
        jac = wp.zeros((1, n_dofs, n_dofs), dtype=wp.float32)
        obj.compute_residuals(None, joint_q, model, residual, 0, None)
        obj.compute_jacobian_analytic(None, joint_q, model, jac, None, 0)
        assert abs(residual.numpy()[0, dof] - expected) < 2e-5, (q, residual.numpy()[0, dof], expected)
        assert abs(jac.numpy()[0, dof, dof] - expected_jac) < 1e-5
    print("barrier residual and analytic Jacobian: zero interior, quadratic approach cost, correct derivative")


def main():
    root = Path(__file__).resolve().parents[1]
    motion = root / "soma_retargeter/assets/soma/soma_zero_frame0.bvh"
    skeleton, animation = load_bvh(str(motion))
    config_path = root / "soma_retargeter/assets/robotics/unitree/g1/configs/soma_to_g1_retargeter_config.json"
    config = json.loads(config_path.read_text())
    config.update(enable_post_processing=False, enable_contact_processing=False,
                  num_initialization_frames=4, num_stabilization_frames=0, ik_iterations=1)
    for enabled in (False, True):
        pipeline = SomaRetargetingPipeline(skeleton, "soma", "unitree_g1",
                                            retarget_config=dict(config, deploy_joint_limit_barrier=enabled))
        if enabled:
            check_barrier_derivative(pipeline.ik_model, pipeline)
        pipeline.add_input_motions([animation], [wp.transform_identity()], scale_animation=True)
        output = pipeline.execute()
        assert len(output) == 1
        assert np.isfinite(output[0].data).all()
        if enabled:
            labels = [label.split("/")[-1] for label in pipeline.ik_model.joint_label]
            starts = pipeline.ik_model.joint_q_start.numpy()
            raw = output[0].data
            for index, name in enumerate(labels):
                if name in DEPLOY_LIMITS["unitree_g1"]:
                    lo, hi = DEPLOY_LIMITS["unitree_g1"][name]
                    assert np.all(raw[:, starts[index]] >= np.float32(lo) - 1e-6)
                    assert np.all(raw[:, starts[index]] <= np.float32(hi) + 1e-6)
        print(f"deploy_joint_limit_barrier={enabled}: stepped 1 G1 frame, finite and in deploy bounds")


if __name__ == "__main__":
    main()
