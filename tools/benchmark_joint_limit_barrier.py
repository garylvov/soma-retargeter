# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Paired SOMA77 BVH retarget benchmark, opt-in joint barrier vs default.

No GPU is used. Outputs only JSON aggregate metrics and selected source clip names.
Run in a memory-capped CPU job; input root must contain the original Kimodo BVHs.
"""

import argparse
import json
import time
from pathlib import Path

import newton
import numpy as np
import warp as wp

from soma_retargeter.io.bvh import load_bvh
from soma_retargeter.pipelines.deploy_joint_limits import DEPLOY_LIMITS
from soma_retargeter.pipelines.soma_retargeting_pipeline import SomaRetargetingPipeline
from soma_retargeter.pipelines.utils import get_retargeter_config
from soma_retargeter.utils.space_conversion_utils import SpaceConverter, get_facing_direction_type_from_str


POINTS = {"pelvis": "Hips", "left_wrist": "LeftHand", "right_wrist": "RightHand",
          "left_foot": "LeftFoot", "right_foot": "RightFoot"}


def _positions(pipeline, result):
    """Forward-kinematics position of matched robot bodies, in source-target frame."""
    model = pipeline.ik_model
    state = model.state()
    indices = {name: pipeline.mapped_body_link_pos_data[pipeline.mapped_joints.index(joint)][0]
               for name, joint in POINTS.items()}
    positions = {name: [] for name in indices}
    for frame in result.data:
        wp.copy(model.joint_q, wp.array(frame, dtype=wp.float32))
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)
        body = state.body_q.numpy()
        for name, index in indices.items():
            positions[name].append(body[index, :3].copy())
    return {name: np.asarray(values, dtype=np.float64) for name, values in positions.items()}


def _score(pipeline, result):
    names = [name.split("/")[-1] for name in pipeline.ik_model.joint_label]
    starts = pipeline.ik_model.joint_q_start.numpy()
    q = result.data
    contract = DEPLOY_LIMITS[pipeline.target_type]
    stop = False
    stop_frames = 0
    for name, idx in zip(names, starts):
        if name in contract:
            lo, hi = contract[name]
            hit = np.isclose(q[:, idx], lo, atol=1e-5, rtol=0) | np.isclose(q[:, idx], hi, atol=1e-5, rtol=0)
            stop |= bool(hit.any())
            stop_frames += int(hit.sum())

    targets = pipeline.input_targets[0][pipeline.num_initialization_frames + pipeline.num_stabilization_frames:]
    pos = _positions(pipeline, result)
    errors = {}
    for key, joint in POINTS.items():
        target = targets[:, pipeline.mapped_joints.index(joint), :3]
        errors[key] = np.linalg.norm(pos[key] - target, axis=1) * 100.0  # cm

    # Contact proxy measured consistently on scaled SOMA source and robot:
    # source foot horizontal speed <= 0.10 m/s and z <= own 10th-percentile z + 5 cm.
    # This is kinematic contact agreement, NOT simulated force/contact physics.
    contacts, slides = [], []
    for foot in ("left_foot", "right_foot"):
        source = targets[:, pipeline.mapped_joints.index(POINTS[foot]), :3]
        output = pos[foot]
        dt = 1.0 / result.sample_rate
        src_speed = np.linalg.norm(np.diff(source[:, :2], axis=0), axis=1) / dt
        out_speed = np.linalg.norm(np.diff(output[:, :2], axis=0), axis=1) / dt
        src_contact = (src_speed <= 0.10) & (source[1:, 2] <= np.quantile(source[:, 2], 0.1) + 0.05)
        out_contact = (out_speed <= 0.10) & (output[1:, 2] <= np.quantile(output[:, 2], 0.1) + 0.05)
        contacts.append((int((src_contact == out_contact).sum()), len(src_contact), int(src_contact.sum()), int(out_contact.sum())))
        slides.append((float(out_speed[src_contact].sum()), int(src_contact.sum())))
    return stop, stop_frames, errors, contacts, slides


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--robot", choices=("unitree_g1", "h1_2"), default="unitree_g1")
    parser.add_argument("--max-frames", type=int, default=0, help="0: evaluate full clips")
    parser.add_argument("--post-processing", action="store_true", help="match production config (enabled by default)")
    args = parser.parse_args()
    if args.robot not in DEPLOY_LIMITS:
        parser.error(f"missing deploy contract: {args.robot}")
    files = sorted(args.source.glob("soma77_*.bvh"))[:args.count]
    if len(files) != args.count:
        parser.error(f"requested {args.count} BVHs, found {len(files)}")
    converter = SpaceConverter(get_facing_direction_type_from_str("Mujoco"))
    offset = converter.transform(wp.transform_identity())
    skeleton, _ = load_bvh(str(files[0]))
    config = get_retargeter_config("soma", args.robot)
    if not args.post_processing:
        config.update(enable_post_processing=False, enable_contact_processing=False)
    scores = {"old": [], "new": []}
    for file in files:
        _, anim = load_bvh(str(file), skeleton)
        if args.max_frames:
            anim.local_transforms = anim.local_transforms[:args.max_frames]
            anim.num_frames = len(anim.local_transforms)
        for mode, enabled in (("old", False), ("new", True)):
            pipeline = SomaRetargetingPipeline(skeleton, "soma", args.robot,
                                                retarget_config=dict(config, deploy_joint_limit_barrier=enabled))
            pipeline.add_input_motions([anim], [offset], scale_animation=True)
            start = time.perf_counter()
            result = pipeline.execute()[0]
            elapsed = time.perf_counter() - start
            stop, stop_frames, errors, contacts, slides = _score(pipeline, result)
            scores[mode].append(dict(clip=file.name, frames=result.num_frames, stop=stop, stop_frames=stop_frames,
                                     errors={key: [float(value.sum()), len(value)] for key, value in errors.items()},
                                     contacts=contacts, slides=slides, ik_seconds=elapsed))
        print(f"PAIRED {len(scores['old'])}/{len(files)} {file.name}: "
              f"stop old={scores['old'][-1]['stop']} new={scores['new'][-1]['stop']}", flush=True)
        args.out.write_text(json.dumps({"robot": args.robot, "scores": scores}, indent=2))

    summary = {}
    for mode, rows in scores.items():
        summary[mode] = {
            "clips_with_stop": sum(row["stop"] for row in rows),
            "stop_frames_joint_sum": sum(row["stop_frames"] for row in rows),
            "clip_count": len(rows),
            "mean_ik_seconds_per_clip": float(np.mean([row["ik_seconds"] for row in rows])),
            "keypoint_mean_error_cm": {key: sum(row["errors"][key][0] for row in rows) /
                                       sum(row["errors"][key][1] for row in rows) for key in POINTS},
            "contact_proxy_agreement": sum(contact[0] for row in rows for contact in row["contacts"]) /
                                       sum(contact[1] for row in rows for contact in row["contacts"]),
            "source_contact_samples": sum(contact[2] for row in rows for contact in row["contacts"]),
            "retarget_contact_samples": sum(contact[3] for row in rows for contact in row["contacts"]),
            "foot_slide_m_per_s_on_source_contact": sum(s[0] for row in rows for s in row["slides"]) /
                                                    max(1, sum(s[1] for row in rows for s in row["slides"])),
        }
    args.out.write_text(json.dumps({"robot": args.robot, "scores": scores, "summary": summary}, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
