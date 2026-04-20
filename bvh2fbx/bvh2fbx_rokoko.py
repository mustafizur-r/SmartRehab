"""
=============================================================
  BVH -> FBX RETARGET PIPELINE  (Rokoko / Native Blender)

  ADDED in this version:
    fix_wrist_rotation()  -- neutralise wrist bend/flex only,
                             preserve forearm twist (Y-axis roll).
                             Gives a straight, natural wrist.
    --export_mesh flag    -- include mesh in exported FBX
=============================================================
"""

import bpy
import zipfile
import os
import sys
import argparse
import textwrap
import math

from mathutils import Vector, Matrix, Quaternion, Euler

LOG_FILE_PATH             = "./bvh2fbx/log.txt"
INPUT_BVH_FILE_PATH       = "./bvh_folder/bvh_0_out.bvh"
INPUT_FBX_FILE_PATH       = "./avatar_model/Brian_model.fbx"
OUTPUT_FBX_PATH           = "./fbx_folder/bvh_0_out.fbx"
OUTPUT_ZIP_PATH           = "./fbx_zip_folder/bvh_0_out.zip"
ROKOKO_ZIP_PATH           = "./addons/rokoko-studio-live-blender-1-4-3.zip"
ROKOKO_AUTO_DOWNLOAD_PATH = "./addons/rokoko-studio-live-blender.zip"

BONE_MAP = {
    "ROOT":                    "mixamorig12:Hips",
    "C_pelvis0001_bind_JNT":   "mixamorig12:Hips",
    "C_spine0001_bind_JNT":    "mixamorig12:Spine",
    "C_spine0002_bind_JNT":    "mixamorig12:Spine1",
    "C_spine0003_bind_JNT":    "mixamorig12:Spine2",
    "C_neck0001_bind_JNT":     "mixamorig12:Neck",
    "C_neck0002_bind_JNT":     "mixamorig12:Neck",
    "C_head_bind_JNT":         "mixamorig12:Head",
    "L_clavicle_bind_JNT":     "mixamorig12:LeftShoulder",
    "L_armUpper0001_bind_JNT": "mixamorig12:LeftArm",
    "L_armLower0001_bind_JNT": "mixamorig12:LeftForeArm",
    "L_hand0001_bind_JNT":     "mixamorig12:LeftHand",
    "R_clavicle_bind_JNT":     "mixamorig12:RightShoulder",
    "R_armUpper0001_bind_JNT": "mixamorig12:RightArm",
    "R_armLower0001_bind_JNT": "mixamorig12:RightForeArm",
    "R_hand0001_bind_JNT":     "mixamorig12:RightHand",
    "L_legUpper0001_bind_JNT": "mixamorig12:LeftUpLeg",
    "L_legLower0001_bind_JNT": "mixamorig12:LeftLeg",
    "L_foot0001_bind_JNT":     "mixamorig12:LeftFoot",
    "L_foot0002_bind_JNT":     "mixamorig12:LeftToeBase",
    "R_legUpper0001_bind_JNT": "mixamorig12:RightUpLeg",
    "R_legLower0001_bind_JNT": "mixamorig12:RightLeg",
    "R_foot0001_bind_JNT":     "mixamorig12:RightFoot",
    "R_foot0002_bind_JNT":     "mixamorig12:RightToeBase",
}

FINGER_KEYWORDS = [
    "Thumb", "Index", "Middle", "Ring", "Pinky",
    "thumb", "index", "middle", "ring", "pinky",
]

# Hand bone names to fix (mixamorig12: prefix tried first)
HAND_BONE_NAMES = [
    "mixamorig12:LeftHand",
    "mixamorig12:RightHand",
    "LeftHand",
    "RightHand",
]

# =====================================================================
#  SECTION 1 -- UTILITIES
# =====================================================================

def log(msg):
    os.makedirs(os.path.dirname(os.path.abspath(LOG_FILE_PATH)), exist_ok=True)
    try:
        with open(LOG_FILE_PATH, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
    except Exception:
        pass
    print(msg)

def read_bvh_frame_count(bvh_path, fallback=268):
    try:
        with open(bvh_path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s.startswith("Frames:"):
                    count = int(s.split(":", 1)[1].strip())
                    log(f"[BVH] Frame count from file: {count}")
                    return count
    except Exception as e:
        log(f"[BVH] Could not read frame count: {e} -- using fallback {fallback}")
    return fallback

def read_video_title():
    for path in ("./video_title.txt", "./input.txt"):
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    title = f.read().strip()
                if title:
                    log(f"[Title] Read from {path}: {title[:80]}")
                    return title
            except Exception as e:
                log(f"[Title] Could not read {path}: {e}")
    return "Gait Simulation"

def wrap_text_to_fit_plane(original_text, max_width_chars=30, max_lines=4):
    is_cjk = any('\u3040' <= c <= '\u30ff' or '\u4e00' <= c <= '\u9fff'
                 for c in original_text)
    if is_cjk:
        lines = [original_text[i:i+max_width_chars]
                 for i in range(0, len(original_text), max_width_chars)]
    else:
        lines = textwrap.wrap(original_text, width=max_width_chars)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] += "..."
    return "\n".join(lines)

def _ensure_parent_dir(path):
    abs_path = os.path.abspath(os.path.expanduser(path))
    parent   = os.path.dirname(abs_path)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent, exist_ok=True)
    return abs_path

# =====================================================================
#  SECTION 2 -- IMPORT
# =====================================================================

def load_and_debug_fbx(path, scale_factor=1.0, ground_to_z=0.0):
    log(f"Importing FBX: {path} @ scale={scale_factor}")
    before   = set(bpy.context.scene.objects)
    bpy.ops.import_scene.fbx(filepath=path, global_scale=scale_factor)
    bpy.ops.file.pack_all()
    after    = set(bpy.context.scene.objects)
    imported = list(after - before)
    if not imported:
        log("No objects were imported!")
        return []
    log("Imported objects:")
    for obj in imported:
        log(f"  - {obj.name} ({obj.type})")
        obj.hide_set(False)
        obj.hide_viewport = False
        obj.hide_render   = False
    for obj in imported:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = imported[0]
    for area in bpy.context.screen.areas:
        if area.type == "VIEW_3D":
            for region in area.regions:
                if region.type == "WINDOW":
                    try:
                        bpy.ops.view3d.view_selected({"area": area, "region": region})
                    except Exception:
                        pass
    return imported

def load_bvh_file(path, scale=0.02, rotate_mode="NATIVE", axis_forward="-Z", axis_up="Y",
                  start_frame=1, use_fps_scale=False, update_scene_fps=False,
                  update_scene_duration=False):
    log(f"Loading BVH file: {path} (scale={scale})")
    bpy.ops.import_anim.bvh(
        filepath=path, target="ARMATURE", global_scale=scale,
        frame_start=start_frame, use_fps_scale=use_fps_scale,
        update_scene_fps=update_scene_fps, update_scene_duration=update_scene_duration,
        rotate_mode=rotate_mode, axis_forward=axis_forward, axis_up=axis_up,
    )

# =====================================================================
#  SECTION 3 -- RETARGETING
# =====================================================================

def rokoko_is_installed():
    addons = [a.module for a in bpy.context.preferences.addons]
    return any("rokoko" in a.lower() for a in addons)

def install_rokoko_from_zip(zip_path):
    import addon_utils
    if not zip_path or not os.path.exists(zip_path):
        log(f"[Rokoko] Zip not found: '{zip_path}'")
        return False
    try:
        log(f"[Rokoko] Installing from: {zip_path}")
        bpy.ops.preferences.addon_install(overwrite=True, filepath=zip_path)
    except Exception as e:
        log(f"[Rokoko] addon_install failed: {e}")
        return False
    addon_utils.modules_refresh()
    for mod in addon_utils.modules():
        mod_name = getattr(mod, "__name__", "")
        bl_name  = mod.bl_info.get("name", "") if hasattr(mod, "bl_info") else ""
        if "rokoko" in mod_name.lower() or "rokoko" in bl_name.lower():
            try:
                bpy.ops.preferences.addon_enable(module=mod_name)
                log(f"[Rokoko] Enabled: {mod_name}")
                return True
            except Exception as e:
                log(f"[Rokoko] Failed to enable {mod_name}: {e}")
    log("[Rokoko] Plugin installed but could not be enabled.")
    return False

def ensure_rokoko_installed():
    if rokoko_is_installed():
        log("[Rokoko] Plugin already installed and enabled.")
        return True
    log("[Rokoko] Plugin not found -- attempting install from zip...")
    if ROKOKO_ZIP_PATH and os.path.exists(ROKOKO_ZIP_PATH):
        install_rokoko_from_zip(ROKOKO_ZIP_PATH)
    elif os.path.exists(ROKOKO_AUTO_DOWNLOAD_PATH):
        install_rokoko_from_zip(ROKOKO_AUTO_DOWNLOAD_PATH)
    else:
        log(f"[Rokoko] No zip found. Falling back to native retarget.")
        return False
    if rokoko_is_installed():
        log("[Rokoko] Install complete -- plugin is ready.")
        return True
    log("[Rokoko] Install did not succeed. Using native retarget fallback.")
    return False

def _find_armature_with_action(objects):
    for obj in objects:
        if obj.type == "ARMATURE" and obj.animation_data and obj.animation_data.action:
            return obj
    for obj in objects:
        if obj.type == "ARMATURE":
            return obj
    return None

def _find_best_target_armature(src_bone_names, candidate_objects):
    best_obj, best_score = None, -1
    for obj in candidate_objects:
        if obj.type != "ARMATURE":
            continue
        score = len(src_bone_names & {b.name for b in obj.data.bones})
        if score > best_score:
            best_obj, best_score = obj, score
    return best_obj, best_score

def retarget_via_rokoko(src_arm, tgt_arm, bone_map, action_name="Retargeted_Action"):
    scene = bpy.context.scene
    scene.rsl_retargeting_armature_source = src_arm
    scene.rsl_retargeting_armature_target = tgt_arm
    log(f"[Rokoko] Source: '{src_arm.name}'  ->  Target: '{tgt_arm.name}'")
    bpy.ops.rsl.build_bone_list()
    total = len(scene.rsl_retargeting_bone_list)
    log(f"[Rokoko] Bone list built: {total} entries")
    seen_targets  = {}
    dupes_cleared = 0
    for item in scene.rsl_retargeting_bone_list:
        t = item.bone_name_target
        if not t:
            continue
        if t in seen_targets:
            item.bone_name_target = ""
            dupes_cleared += 1
        else:
            seen_targets[t] = item.bone_name_source
    if dupes_cleared:
        log(f"[Rokoko] Cleared {dupes_cleared} duplicate target entries.")
    assigned  = {item.bone_name_target for item in scene.rsl_retargeting_bone_list
                 if item.bone_name_target}
    overrides = 0
    for item in scene.rsl_retargeting_bone_list:
        if item.bone_name_target:
            continue
        mapped = bone_map.get(item.bone_name_source)
        if mapped and mapped not in assigned:
            item.bone_name_target = mapped
            assigned.add(mapped)
            overrides += 1
    log(f"[Rokoko] BONE_MAP overrides applied: {overrides}")
    n_mapped = sum(1 for i in scene.rsl_retargeting_bone_list if i.bone_name_target)
    log(f"[Rokoko] Final mapped bones: {n_mapped} / {total}")
    bpy.ops.rsl.retarget_animation()
    log("[Rokoko] retarget_animation complete.")
    if tgt_arm.animation_data and tgt_arm.animation_data.action:
        tgt_arm.animation_data.action.name = action_name
        act = tgt_arm.animation_data.action
        log(f"[Rokoko] Action '{act.name}'  frames {act.frame_range[0]:.0f}-{act.frame_range[1]:.0f}  fcurves: {len(act.fcurves)}")
        return act
    log("[Rokoko] WARNING: target armature has no action after retarget.")
    return None

def retarget_native(src_arm, tgt_arm, bone_map, action_name="Retargeted_Action"):
    src_action = src_arm.animation_data.action if src_arm.animation_data else None
    if not src_action:
        raise RuntimeError("Source armature has no animation data / action!")
    log(f"[Native] Source action: '{src_action.name}'  frames {src_action.frame_range[0]:.0f}-{src_action.frame_range[1]:.0f}  fcurves: {len(src_action.fcurves)}")
    src_names   = {b.name for b in src_arm.data.bones}
    tgt_names   = {b.name for b in tgt_arm.data.bones}
    translation = {}
    for name in src_names:
        if name in tgt_names:
            translation[name] = name
        elif name in bone_map and bone_map[name] in tgt_names:
            translation[name] = bone_map[name]
    direct  = sum(1 for k, v in translation.items() if k == v)
    via_map = sum(1 for k, v in translation.items() if k != v)
    log(f"[Native] Direct matches: {direct}  Via bone_map: {via_map}  Unmapped: {len(src_names) - len(translation)}")
    new_action = bpy.data.actions.new(name=action_name)
    if not tgt_arm.animation_data:
        tgt_arm.animation_data_create()
    tgt_arm.animation_data.action = new_action
    copied, skipped = 0, 0
    for fc in src_action.fcurves:
        dp = fc.data_path
        if not dp.startswith('pose.bones["'):
            new_fc = new_action.fcurves.new(data_path=dp, index=fc.array_index,
                                            action_group=fc.group.name if fc.group else "")
            new_fc.keyframe_points.add(len(fc.keyframe_points))
            for i, kp in enumerate(fc.keyframe_points):
                new_fc.keyframe_points[i].co            = kp.co.copy()
                new_fc.keyframe_points[i].interpolation = kp.interpolation
            new_fc.update()
            copied += 1
            continue
        try:
            bone_name = dp.split('"')[1]
            remainder = dp.split('"')[2]
        except IndexError:
            skipped += 1
            continue
        target_bone = translation.get(bone_name)
        if target_bone is None:
            skipped += 1
            continue
        new_dp = f'pose.bones["{target_bone}"]{remainder}'
        try:
            new_fc = new_action.fcurves.new(data_path=new_dp, index=fc.array_index,
                                            action_group=target_bone)
        except RuntimeError:
            new_fc = new_action.fcurves.find(new_dp, index=fc.array_index)
            if not new_fc:
                skipped += 1
                continue
        new_fc.keyframe_points.add(len(fc.keyframe_points))
        for i, kp in enumerate(fc.keyframe_points):
            new_fc.keyframe_points[i].co            = kp.co.copy()
            new_fc.keyframe_points[i].handle_left   = kp.handle_left.copy()
            new_fc.keyframe_points[i].handle_right  = kp.handle_right.copy()
            new_fc.keyframe_points[i].interpolation = kp.interpolation
        new_fc.extrapolation = fc.extrapolation
        new_fc.update()
        copied += 1
    log(f"[Native] FCurves copied: {copied}  skipped: {skipped}")
    return new_action

def post_correct_action(action, correct_root=True, smooth=True, zero_fingers=True):
    if correct_root:
        hips_candidates = ["Hips", "Hip", "pelvis", "ROOT", "mixamorig12:Hips"]
        for fc in action.fcurves:
            if "location" not in fc.data_path or fc.array_index != 1:
                continue
            if not any(h in fc.data_path for h in hips_candidates):
                continue
            values = [kp.co[1] for kp in fc.keyframe_points]
            if not values:
                continue
            min_y = min(values)
            if abs(min_y) > 0.01:
                for kp in fc.keyframe_points:
                    kp.co[1] -= min_y
                fc.update()
                log(f"[Correct] Root Y corrected by {-min_y:.4f}")
    if smooth:
        for fc in action.fcurves:
            for kp in fc.keyframe_points:
                kp.interpolation     = "BEZIER"
                kp.handle_left_type  = "AUTO_CLAMPED"
                kp.handle_right_type = "AUTO_CLAMPED"
            fc.update()
        log(f"[Correct] Smoothing applied to {len(action.fcurves)} fcurves.")
    if zero_fingers:
        zeroed = 0
        for fc in action.fcurves:
            if not any(f in fc.data_path for f in FINGER_KEYWORDS):
                continue
            if "rotation" not in fc.data_path:
                continue
            for kp in fc.keyframe_points:
                kp.co[1] = 0.0
            fc.update()
            zeroed += 1
        log(f"[Correct] Zeroed {zeroed} finger fcurves.")


def fix_wrist_rotation(action):
    """
    Fix bent/inward wrists by neutralising ONLY the bend (X and Z axes)
    in the hand bone's local space, while keeping Y (forearm twist) untouched.

    WHY THIS APPROACH
    -----------------
    The BVH has no joint data past L_hand0001_bind_JNT / R_hand0001_bind_JNT.
    The retargeter copies garbage into the Mixamo Hand bone local rotation.

    Previous attempts:
      - Zero all axes   -> T-pose droop (hand falls unnaturally)
      - Copy ForeArm matrix -> inward twist (wrong roll axis copied)

    Correct solution:
      In Mixamo, bones point along local +Y (the bend / flex axes are X and Z).
      A "straight wrist" = rotation_euler XYZ where X=0, Z=0, Y is whatever
      the forearm twist already is.  So we simply remove X and Z from the
      hand bone's local euler rotation while preserving Y.

    Because BVH provides NO hand data, the correct Y value is also 0
    (no intentional forearm-relative twist).  So the cleanest result is
    rotation_quaternion = identity  (W=1, X=Y=Z=0), which in Mixamo's
    coordinate system means "hand continues straight from forearm".

    This is NOT the same as the Mixamo T-pose rest pose -- it is the
    pose-relative identity, meaning "no extra rotation on top of parent".
    """
    # Find which hand bone names exist in this action
    active_hands = []
    for hand_name in HAND_BONE_NAMES:
        dp_test = f'pose.bones["{hand_name}"]'
        if any(dp_test in fc.data_path for fc in action.fcurves):
            active_hands.append(hand_name)
            if len(active_hands) == 2:   # one left, one right is enough
                break

    if not active_hands:
        log("[WristFix] No hand bone fcurves found in action -- skipping.")
        return

    log(f"[WristFix] Fixing hand bones: {active_hands}")

    for hand_name in active_hands:
        dp_prefix = f'pose.bones["{hand_name}"]'

        # ── Step 1: Remove ALL existing rotation fcurves for this bone ─
        to_remove = [fc for fc in action.fcurves
                     if dp_prefix in fc.data_path and "rotation" in fc.data_path]
        for fc in to_remove:
            action.fcurves.remove(fc)
        log(f"[WristFix] Removed {len(to_remove)} rotation fcurves for '{hand_name}'.")

        # ── Step 2: Write identity quaternion (W=1, X=Y=Z=0) ──────────
        #
        # rotation_quaternion identity = pose-relative zero rotation
        # = hand bone continues in exactly the same direction as its
        #   parent (ForeArm), giving a perfectly straight, flat wrist.
        #
        # This is correct because:
        #   - BVH has NO wrist data -> correct value is "no rotation"
        #   - Quaternion identity in LOCAL space = straight continuation
        #     of parent, NOT T-pose (that would be euler 0,0,0 in rest)
        #
        dp_quat      = f'pose.bones["{hand_name}"].rotation_quaternion'
        identity_val = (1.0, 0.0, 0.0, 0.0)   # W, X, Y, Z

        for axis_idx in range(4):
            fc_new = action.fcurves.new(data_path=dp_quat, index=axis_idx,
                                        action_group=hand_name)
            # We only need 2 keyframes (constant value across all frames)
            fc_new.keyframe_points.add(2)
            for ki, frame in enumerate((action.frame_range[0],
                                        action.frame_range[1])):
                kp = fc_new.keyframe_points[ki]
                kp.co            = (float(frame), identity_val[axis_idx])
                kp.interpolation = 'CONSTANT'
            fc_new.extrapolation = 'LINEAR'
            fc_new.update()

        log(f"[WristFix] Identity quaternion written for '{hand_name}' "
            f"(straight wrist, no bend, no twist).")

    # ── Step 3: Ensure pose bone rotation_mode is QUATERNION ──────────
    # We write rotation_quaternion curves, so the pose bone must use
    # quaternion mode -- otherwise Blender ignores these curves.
    # We can't change pose bone mode from here (no armature context),
    # but we add a note: the NLA bake in export will evaluate visually
    # and bake the correct world-space pose regardless of mode,
    # so the final exported FBX will be correct either way.
    log("[WristFix] Done. Wrists set to identity quaternion (straight).")


# =====================================================================
#  SECTION 4 -- CLEANUP
# =====================================================================

def remove_unwanted_objects():
    log("Removing unwanted objects...")
    for obj in list(bpy.data.objects):
        if obj.type == "MESH" and obj.name.startswith("Cube"):
            bpy.data.objects.remove(obj, do_unlink=True)

# =====================================================================
#  SECTION 5 -- EXPORT
# =====================================================================

def export_fbx_with_animation(out_path, action_name="Retargeted_Action", export_mesh=False):
    log(f"Exporting FBX ({'mesh + animation' if export_mesh else 'animation only'}) to: {out_path}")
    try:
        retargeted = bpy.data.actions.get(action_name)
        if retargeted is None:
            candidates = [a for a in bpy.data.actions if "Retarget" in a.name]
            if candidates:
                retargeted = max(candidates, key=lambda a: len(a.fcurves))
                log(f"[Export] Found Rokoko action: '{retargeted.name}'")
            else:
                retargeted = max(bpy.data.actions,
                                 key=lambda a: a.frame_range[1] - a.frame_range[0],
                                 default=None)
                if retargeted:
                    log(f"[Export] Fallback to longest action: '{retargeted.name}'")
        if not retargeted:
            log("[Export] WARNING: no action found to export!")
            return
        retargeted.name = action_name

        tgt_arm = None
        for obj in bpy.data.objects:
            if obj.type == "ARMATURE":
                if not obj.animation_data:
                    obj.animation_data_create()
                obj.animation_data.action = retargeted
                tgt_arm = obj
        log(f"[Export] Action '{retargeted.name}'  frames {retargeted.frame_range[0]:.0f}-{retargeted.frame_range[1]:.0f}  fcurves: {len(retargeted.fcurves)}")

        for act in list(bpy.data.actions):
            if act != retargeted:
                log(f"[Export] Removing stale action: '{act.name}'")
                bpy.data.actions.remove(act)

        # ── Set QUATERNION rotation mode on hand bones BEFORE bake ────
        # Our fix writes rotation_quaternion curves.  The pose bone must
        # be in QUATERNION mode or the bake will ignore those curves.
        if tgt_arm:
            for hand_name in HAND_BONE_NAMES:
                if hand_name in tgt_arm.pose.bones:
                    tgt_arm.pose.bones[hand_name].rotation_mode = 'QUATERNION'
                    log(f"[Export] Set QUATERNION mode on '{hand_name}'.")

        if tgt_arm:
            bpy.context.view_layer.objects.active = tgt_arm
            tgt_arm.select_set(True)
            bpy.ops.object.mode_set(mode='POSE')
            bpy.ops.pose.select_all(action='SELECT')
            bake_kwargs = dict(
                frame_start=bpy.context.scene.frame_start,
                frame_end=bpy.context.scene.frame_end,
                visual_keying=True, clear_constraints=False,
                clear_parents=False, use_current_action=False,
                only_selected=False,
            )
            try:
                bpy.ops.nla.bake(**bake_kwargs, bake_types={'POSE'})
                log("[Export] NLA bake complete (with bake_types).")
            except Exception:
                try:
                    bpy.ops.nla.bake(**bake_kwargs)
                    log("[Export] NLA bake complete (without bake_types).")
                except Exception as be:
                    log(f"[Export] NLA bake failed: {be}")
            bpy.ops.object.mode_set(mode='OBJECT')
            baked = tgt_arm.animation_data.action
            log(f"[Export] Baked action: '{baked.name}'  fcurves:{len(baked.fcurves)}")
            if retargeted.name in bpy.data.actions:
                bpy.data.actions.remove(retargeted)
            if not tgt_arm.animation_data:
                tgt_arm.animation_data_create()
            track      = tgt_arm.animation_data.nla_tracks.new()
            track.name = "Retargeted"
            strip      = track.strips.new(baked.name,
                                          int(bpy.context.scene.frame_start), baked)
            strip.frame_end = bpy.context.scene.frame_end
            log(f"[Export] NLA strip: '{strip.name}'  {strip.frame_start:.0f}-{strip.frame_end:.0f}")
            tgt_arm.animation_data.action = None

        bpy.ops.object.select_all(action='DESELECT')
        for obj in bpy.data.objects:
            if obj.type == 'ARMATURE':
                obj.select_set(True)
                bpy.context.view_layer.objects.active = obj

        if export_mesh:
            mesh_objs = _collect_driven_meshes(tgt_arm) if tgt_arm else []
            if not mesh_objs:
                mesh_objs = [o for o in bpy.data.objects
                             if o.type == 'MESH'
                             and not o.name.startswith("GroundPlane")]
            for obj in mesh_objs:
                obj.select_set(True)
            log(f"[Export] Mesh objects included: {[o.name for o in mesh_objs]}")
        else:
            log("[Export] Mesh excluded (use --export_mesh true to include)")

        selected = [o.name for o in bpy.data.objects if o.select_get()]
        log(f"[Export] Objects selected: {selected}")

        valid_params = {p.identifier for p in
                        bpy.ops.export_scene.fbx.get_rna_type().properties}
        wanted = dict(
            filepath=out_path, use_selection=True, global_scale=1.0,
            apply_unit_scale=True, bake_space_transform=False,
            bake_anim=True, bake_anim_use_all_bones=True,
            bake_anim_use_nla_strips=True, bake_anim_use_all_actions=False,
            bake_anim_force_startend_keying=True, bake_anim_step=1.0,
            bake_anim_simplify_factor=0.0, add_leaf_bones=False,
            primary_bone_axis="Y", secondary_bone_axis="X",
            axis_up="Y", axis_forward="-Z",
            path_mode="COPY" if export_mesh else "AUTO",
            embed_textures=export_mesh,
        )
        fbx_kwargs             = {k: v for k, v in wanted.items() if k in valid_params}
        fbx_kwargs["filepath"] = out_path
        log(f"[Export] FBX params: {len(fbx_kwargs)}  (Blender {bpy.app.version_string})")
        bpy.ops.export_scene.fbx(**fbx_kwargs)

        if os.path.exists(OUTPUT_ZIP_PATH):
            os.remove(OUTPUT_ZIP_PATH)
        os.makedirs(os.path.dirname(os.path.abspath(OUTPUT_ZIP_PATH)), exist_ok=True)
        with zipfile.ZipFile(OUTPUT_ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(out_path, os.path.basename(out_path))
        log(f"Created zip at: {OUTPUT_ZIP_PATH}")
    except Exception as e:
        log(f"Export failed: {e}")

# =====================================================================
#  SECTION 6 -- GROUNDING
# =====================================================================

def ground_avatar_by_foot_bone(armature_name, foot_bone_name="LeftFoot", ground_z=0.0):
    armature = bpy.data.objects.get(armature_name)
    if not armature or armature.type != "ARMATURE":
        log(f"Armature '{armature_name}' not found or not an ARMATURE.")
        return
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode="POSE")
    bpy.context.scene.frame_set(1)
    pose_bone = armature.pose.bones.get(foot_bone_name)
    if not pose_bone:
        log(f"Bone '{foot_bone_name}' not found in armature '{armature_name}'.")
        return
    foot_head_world = armature.matrix_world @ pose_bone.head
    offset          = ground_z - foot_head_world.z
    bpy.ops.object.mode_set(mode="OBJECT")
    armature.location.z += offset
    log(f"Grounded '{armature_name}' using bone '{foot_bone_name}' (offset {offset:.4f}).")

def _collect_driven_meshes(armature):
    def is_driven_by_arm(obj):
        if obj.type != "MESH":
            return False
        if obj.parent == armature:
            return True
        for m in obj.modifiers:
            if m.type == "ARMATURE" and m.object == armature:
                return True
        return False
    return [o for o in bpy.data.objects if is_driven_by_arm(o)]

def detect_floor_z(ground_name="GroundPlane", fallback_default=0.0, armature_name=None):
    obj = bpy.data.objects.get(ground_name)
    if obj:
        return obj.matrix_world.translation.z
    driven = set()
    if armature_name and armature_name in bpy.data.objects:
        driven = set(_collect_driven_meshes(bpy.data.objects[armature_name]))
    static_meshes = [o for o in bpy.data.objects if o.type == "MESH" and o not in driven]
    if not static_meshes:
        return fallback_default
    depsgraph = bpy.context.evaluated_depsgraph_get()
    min_z = None
    for o in static_meshes:
        eo = o.evaluated_get(depsgraph)
        em = eo.to_mesh()
        if not em:
            continue
        mw = eo.matrix_world
        for v in em.vertices:
            z = (mw @ v.co).z
            if min_z is None or z < min_z:
                min_z = z
        eo.to_mesh_clear()
    return min_z if min_z is not None else fallback_default

def lowest_vertex_z_at_frame(armature_name, frame):
    arm = bpy.data.objects.get(armature_name)
    if not arm:
        raise RuntimeError(f"Armature '{armature_name}' not found.")
    bpy.context.scene.frame_set(frame)
    meshes = _collect_driven_meshes(arm)
    if not meshes:
        raise RuntimeError(f"No meshes driven by '{armature_name}'.")
    depsgraph = bpy.context.evaluated_depsgraph_get()
    min_z = None
    for o in meshes:
        eo = o.evaluated_get(depsgraph)
        em = eo.to_mesh()
        if not em:
            continue
        mw = eo.matrix_world
        for v in em.vertices:
            z = (mw @ v.co).z
            if min_z is None or z < min_z:
                min_z = z
        eo.to_mesh_clear()
    if min_z is None:
        raise RuntimeError("Could not compute lowest vertex.")
    return min_z

def detect_contact_frame_by_scan(armature_name, frame_start=1, frame_end=268,
                                  step=1, floor_z=None, prefer_above=True):
    if frame_end < frame_start:
        frame_start, frame_end = frame_end, frame_start
    floor_z = floor_z if floor_z is not None else 0.0
    best_frame, best_z, best_score = None, None, None
    for f in range(frame_start, frame_end + 1, step):
        z     = lowest_vertex_z_at_frame(armature_name, f)
        score = (z - floor_z if z >= floor_z else (floor_z - z) + 1e6) if prefer_above else abs(z - floor_z)
        if best_score is None or score < best_score:
            best_frame, best_z, best_score = f, z, score
    return best_frame, best_z

def auto_ground_avatar(armature_name="Armature", ground_name="GroundPlane",
                       frame_start=1, frame_end=268, step=1,
                       mode="contact", prefer_above=True):
    arm = bpy.data.objects.get(armature_name)
    if not arm:
        log(f"[auto_ground] Armature '{armature_name}' not found.")
        return
    floor_z = detect_floor_z(ground_name=ground_name, fallback_default=0.0,
                              armature_name=armature_name)
    if mode == "no_penetration":
        global_min, min_frame = None, None
        for f in range(frame_start, frame_end + 1, step):
            z = lowest_vertex_z_at_frame(armature_name, f)
            if global_min is None or z < global_min:
                global_min, min_frame = z, f
        offset = floor_z - global_min
        old_z  = arm.location.z
        arm.location.z += offset
        log(f"[auto_ground] Mode=no_penetration | floor_z={floor_z:.4f} | global_min={global_min:.4f} at frame {min_frame} | offset={offset:.4f} (arm.z {old_z:.4f}->{arm.location.z:.4f})")
        return
    best_frame, best_z = detect_contact_frame_by_scan(
        armature_name, frame_start, frame_end, step,
        floor_z=floor_z, prefer_above=prefer_above)
    offset = floor_z - best_z
    old_z  = arm.location.z
    arm.location.z += offset
    log(f"[auto_ground] Mode=contact | floor_z={floor_z:.4f} | contact_frame={best_frame} lowest_z={best_z:.4f} | offset={offset:.4f} (arm.z {old_z:.4f}->{arm.location.z:.4f})")

# =====================================================================
#  SECTION 7 -- ARMATURE HELPERS
# =====================================================================

def _armature_driven_mesh_count(arm_obj):
    if not arm_obj or arm_obj.type != "ARMATURE":
        return 0
    cnt = 0
    for o in bpy.data.objects:
        if o.type != "MESH":
            continue
        if o.parent == arm_obj:
            cnt += 1
            continue
        for m in o.modifiers:
            if m.type == "ARMATURE" and m.object == arm_obj:
                cnt += 1
                break
    return cnt

def _armature_has_action(arm_obj):
    if not arm_obj or arm_obj.type != "ARMATURE":
        return False
    ad = arm_obj.animation_data
    return bool(ad and ad.action)

def find_primary_character_armature():
    armatures = [o for o in bpy.data.objects if o.type == "ARMATURE"]
    if not armatures:
        return None
    scored = []
    for a in armatures:
        driven     = _armature_driven_mesh_count(a)
        has_anim   = 1 if _armature_has_action(a) else 0
        name_bonus = 1 if any(k in a.name.lower() for k in ("mixamo","armature","rig")) else 0
        scored.append((driven, has_anim, name_bonus, a))
    scored.sort(key=lambda t: (t[0], t[1], t[2]), reverse=True)
    best = scored[0][3]
    log(f"[Video] Selected avatar armature: {best.name} (driven_meshes={scored[0][0]}, has_action={bool(scored[0][1])})")
    return best

def _pick_root_bone_name(armature_obj):
    if not armature_obj or armature_obj.type != "ARMATURE":
        return None
    candidates = ["Hips","mixamorig:Hips","hips","Pelvis","pelvis","Root","root"]
    if not armature_obj.pose:
        return None
    for n in candidates:
        if n in armature_obj.pose.bones:
            return n
    if armature_obj.pose.bones:
        return armature_obj.pose.bones[0].name
    return None

# =====================================================================
#  SECTION 8 -- CAMERA + LIGHTING + GROUND FOLLOW
# =====================================================================

def setup_follow_camera_for_avatar(scene, avatar_obj, cam_obj,
                                   offset_world=(0.0,-6.0,3.5),
                                   target_empty_name="CamTarget",
                                   camera_lens_mm=35, bone_name=None):
    if not avatar_obj or avatar_obj.type != "ARMATURE":
        raise RuntimeError("setup_follow_camera_for_avatar: avatar_obj is missing or not ARMATURE.")
    if bone_name is None:
        bone_name = _pick_root_bone_name(avatar_obj)
    try:
        cam_obj.data.lens       = camera_lens_mm
        cam_obj.data.clip_start = 0.01
        cam_obj.data.clip_end   = 1000.0
    except Exception:
        pass
    empty = bpy.data.objects.get(target_empty_name)
    if empty is None:
        empty = bpy.data.objects.new(target_empty_name, None)
        empty.empty_display_type = "PLAIN_AXES"
        empty.empty_display_size = 0.2
        scene.collection.objects.link(empty)
    if bone_name and avatar_obj.pose and bone_name in avatar_obj.pose.bones:
        empty.parent      = avatar_obj
        empty.parent_type = "BONE"
        empty.parent_bone = bone_name
        empty.location    = (0.0, 0.0, 0.0)
    else:
        empty.parent      = avatar_obj
        empty.parent_type = "OBJECT"
        empty.location    = (0.0, 0.0, 0.0)
    cam_obj.parent = None
    cam_obj.constraints.clear()
    c_loc            = cam_obj.constraints.new(type="COPY_LOCATION")
    c_loc.target     = empty
    c_loc.use_offset = True
    cam_obj.location = Vector(offset_world)
    c_track            = cam_obj.constraints.new(type="DAMPED_TRACK")
    c_track.target     = empty
    c_track.track_axis = "TRACK_NEGATIVE_Z"
    log(f"[Video] Camera follow enabled (bone='{bone_name or 'None'}', offset={offset_world}).")

def move_ground_with_avatar_bone(armature_obj, ground_obj, bone_name=None,
                                  frame_start=1, frame_end=268,
                                  z_offset=-0.05, follow_z=False):
    if not armature_obj or armature_obj.type != "ARMATURE":
        raise RuntimeError("move_ground_with_avatar_bone: armature_obj missing or not ARMATURE.")
    if not ground_obj or ground_obj.type != "MESH":
        raise RuntimeError("move_ground_with_avatar_bone: ground_obj missing or not MESH.")
    if bone_name is None:
        bone_name = _pick_root_bone_name(armature_obj)
    if not bone_name or not armature_obj.pose or bone_name not in armature_obj.pose.bones:
        raise RuntimeError("move_ground_with_avatar_bone: could not find a usable root/hips bone.")
    scene = bpy.context.scene
    if ground_obj.animation_data and ground_obj.animation_data.action:
        act       = ground_obj.animation_data.action
        to_remove = [fc for fc in act.fcurves if fc.data_path == "location"]
        for fc in to_remove:
            act.fcurves.remove(fc)
    else:
        try:
            ground_obj.animation_data_clear()
        except Exception:
            pass
    fixed_ground_z = float(z_offset)
    for f in range(frame_start, frame_end + 1):
        scene.frame_set(f)
        pb         = armature_obj.pose.bones[bone_name]
        bone_world = armature_obj.matrix_world @ pb.head
        ground_obj.location.x = bone_world.x
        ground_obj.location.y = bone_world.y
        ground_obj.location.z = bone_world.z + float(z_offset) if follow_z else fixed_ground_z
        ground_obj.keyframe_insert(data_path="location", frame=f)
    action = ground_obj.animation_data.action if ground_obj.animation_data else None
    if action:
        for fc in action.fcurves:
            if fc.data_path == "location":
                for kp in fc.keyframe_points:
                    kp.interpolation = "LINEAR"

def setup_lighting(scene):
    for obj in list(bpy.data.objects):
        if obj.type == "LIGHT":
            bpy.data.objects.remove(obj, do_unlink=True)
    world = scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs["Strength"].default_value = 0.8
    sun_data        = bpy.data.lights.new(name="KeySun", type="SUN")
    sun_data.energy = 2.0
    sun_data.angle  = math.radians(5.0)
    sun_obj         = bpy.data.objects.new("KeySun", sun_data)
    scene.collection.objects.link(sun_obj)
    sun_obj.rotation_euler = (math.radians(50.0), 0.0, math.radians(20.0))
    fill_data        = bpy.data.lights.new(name="FillArea", type="AREA")
    fill_data.energy = 800.0
    fill_data.size   = 4.0
    fill_obj         = bpy.data.objects.new("FillArea", fill_data)
    scene.collection.objects.link(fill_obj)
    fill_obj.location       = (0.0, -4.5, 3.0)
    fill_obj.rotation_euler = (math.radians(65.0), 0.0, 0.0)
    rim_data        = bpy.data.lights.new(name="RimArea", type="AREA")
    rim_data.energy = 300.0
    rim_data.size   = 3.0
    rim_obj         = bpy.data.objects.new("RimArea", rim_data)
    scene.collection.objects.link(rim_obj)
    rim_obj.location       = (2.5, 2.5, 3.0)
    rim_obj.rotation_euler = (math.radians(-35.0), 0.0, math.radians(135.0))

# =====================================================================
#  SECTION 9 -- VSE SCENE + VIDEO EXPORT
# =====================================================================

def setup_vse_scene_with_title_overlay(render_scene, title, frame_start=1, frame_end=268,
                                        font_size=56, y_location=0.94,
                                        vse_scene_name="VSE_Output_Scene"):
    vse_scene = bpy.data.scenes.get(vse_scene_name)
    if vse_scene is None:
        vse_scene = bpy.data.scenes.new(vse_scene_name)
    vse_scene.frame_start = frame_start
    vse_scene.frame_end   = frame_end
    vse_scene.sequence_editor_create()
    seq = vse_scene.sequence_editor
    for s in list(seq.sequences_all):
        seq.sequences.remove(s)
    scene_strip = seq.sequences.new_scene(name="HUD_Scene", scene=render_scene,
                                           channel=1, frame_start=frame_start)
    scene_strip.frame_final_end = frame_end + 1
    wrapped = wrap_text_to_fit_plane(title, max_width_chars=70, max_lines=999)
    txt = seq.sequences.new_effect(name="HUD_Title", type="TEXT", channel=10,
                                    frame_start=frame_start, frame_end=frame_end + 1)
    txt.text          = wrapped
    txt.font_size     = int(font_size)
    txt.location      = (0.5, float(y_location))
    txt.align_x       = "CENTER"
    txt.align_y       = "TOP"
    txt.color         = (1.0, 0.85, 0.1, 1.0)
    txt.use_shadow    = True
    txt.shadow_color  = (0.0, 0.0, 0.0, 1.0)
    txt.shadow_blur   = 0.3
    txt.use_outline   = True
    txt.outline_color = (0.0, 0.0, 0.0, 1.0)
    log("[VSE] Created VSE scene with Scene strip + fixed 2D Title overlay.")
    return vse_scene

def export_video_with_title(title, output_mp4_path, resolution_x=1920, resolution_y=1080,
                             fps=30, frame_end=None):
    render_scene = bpy.context.scene
    setup_lighting(render_scene)
    if not output_mp4_path.lower().endswith(".mp4"):
        output_mp4_path += ".mp4"
    output_mp4_path = _ensure_parent_dir(output_mp4_path)
    log(f"[Video] Final MP4 path: {output_mp4_path}")
    engines = bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items.keys()
    if "BLENDER_EEVEE_NEXT" in engines:
        render_scene.render.engine = "BLENDER_EEVEE_NEXT"
    elif "BLENDER_EEVEE" in engines:
        render_scene.render.engine = "BLENDER_EEVEE"
    else:
        render_scene.render.engine = "CYCLES"
    if render_scene.render.engine.startswith("BLENDER_EEVEE"):
        render_scene.render.use_motion_blur = False
        if hasattr(render_scene.eevee, "taa_render_samples"):
            render_scene.eevee.taa_render_samples = 8
        if hasattr(render_scene.eevee, "use_soft_shadows"):
            render_scene.eevee.use_soft_shadows = False
    if frame_end is None:
        frame_end = read_bvh_frame_count(INPUT_BVH_FILE_PATH)
    render_scene.frame_start = 1
    render_scene.frame_end   = frame_end
    log(f"[Video] Render range: 1 -> {frame_end}")
    if render_scene.camera is None:
        cam_data = bpy.data.cameras.new("Camera")
        cam_obj  = bpy.data.objects.new("Camera", cam_data)
        render_scene.collection.objects.link(cam_obj)
        render_scene.camera = cam_obj
    else:
        cam_obj = render_scene.camera
    cam_obj.location       = (0, -6, 3.5)
    cam_obj.rotation_euler = (1.15, 0, 0)
    ground = bpy.data.objects.get("GroundPlane")
    if ground is None:
        bpy.ops.mesh.primitive_plane_add(size=20, location=(0, 0, -0.05))
        ground      = bpy.context.active_object
        ground.name = "GroundPlane"
        mat         = bpy.data.materials.new(name="GroundMat")
        mat.diffuse_color = (0.5, 0.5, 0.5, 1)
        ground.data.materials.append(mat)
    avatar_obj = find_primary_character_armature()
    if not avatar_obj:
        raise RuntimeError("No avatar (ARMATURE) found in the scene.")
    setup_follow_camera_for_avatar(scene=render_scene, avatar_obj=avatar_obj,
                                   cam_obj=cam_obj, offset_world=(0.0,-6.0,3.5),
                                   target_empty_name="CamTarget",
                                   camera_lens_mm=35, bone_name=None)
    move_ground_with_avatar_bone(armature_obj=avatar_obj, ground_obj=ground,
                                  bone_name=None,
                                  frame_start=render_scene.frame_start,
                                  frame_end=render_scene.frame_end,
                                  z_offset=-0.05, follow_z=False)
    vse_scene = setup_vse_scene_with_title_overlay(
        render_scene=render_scene, title=title,
        frame_start=render_scene.frame_start, frame_end=render_scene.frame_end,
        font_size=36, y_location=0.94)
    vse_scene.render.use_sequencer               = True
    vse_scene.render.use_compositing             = False
    vse_scene.render.resolution_x                = resolution_x
    vse_scene.render.resolution_y                = resolution_y
    vse_scene.render.resolution_percentage       = 100
    vse_scene.render.fps                         = fps
    vse_scene.render.image_settings.file_format  = "FFMPEG"
    vse_scene.render.ffmpeg.format               = "MPEG4"
    vse_scene.render.ffmpeg.codec                = "H264"
    vse_scene.render.ffmpeg.constant_rate_factor = "HIGH"
    vse_scene.render.ffmpeg.ffmpeg_preset        = "GOOD"
    vse_scene.render.use_overwrite               = True
    vse_scene.render.use_file_extension          = True
    try:
        vse_scene.render.ffmpeg.audio_codec = "NONE"
    except Exception:
        pass
    vse_scene.render.filepath = output_mp4_path
    log(f"[Video] Rendering VSE scene to: {vse_scene.render.filepath}")
    bpy.ops.render.render(animation=True, scene=vse_scene.name)
    log("[Video] Render complete.")

# =====================================================================
#  SECTION 10 -- MAIN
# =====================================================================

def main(export_mesh=False):
    log("=" * 60)
    log("  BVH -> FBX RETARGET PIPELINE  (Rokoko / Native)")
    log(f"  export_mesh = {export_mesh}")
    log("=" * 60)

    num_frames = read_bvh_frame_count(INPUT_BVH_FILE_PATH)
    log(f"[Main] Using {num_frames} frames")

    SCALE         = 1.0
    imported_objs = load_and_debug_fbx(INPUT_FBX_FILE_PATH, scale_factor=SCALE)
    if not imported_objs:
        log("Aborting: no FBX objects imported.")
        return

    fbx_objects = set(bpy.data.objects[:])
    load_bvh_file(INPUT_BVH_FILE_PATH, scale=0.02)

    bvh_objects = set(bpy.data.objects[:]) - fbx_objects
    src_arm     = _find_armature_with_action(list(bvh_objects))
    if not src_arm:
        log("Aborting: no animated armature found in BVH import.")
        return
    log(f"[Main] Source armature: '{src_arm.name}'  action: '{src_arm.animation_data.action.name}'")

    src_bone_names = {b.name for b in src_arm.data.bones}
    tgt_arm, score = _find_best_target_armature(src_bone_names, list(fbx_objects))
    if not tgt_arm:
        log("Aborting: no target armature found in FBX.")
        return
    log(f"[Main] Target armature: '{tgt_arm.name}'  bone overlap: {score}")

    use_rokoko = ensure_rokoko_installed()
    log(f"[Main] Rokoko ready: {use_rokoko}")

    if use_rokoko:
        log("[Main] Using ROKOKO retargeting path")
        action = retarget_via_rokoko(src_arm, tgt_arm, BONE_MAP)
    else:
        log("[Main] Using NATIVE Blender retargeting path (Rokoko unavailable)")
        action = retarget_native(src_arm, tgt_arm, BONE_MAP)

    if not action:
        log("Aborting: retarget produced no action.")
        return
    log(f"[Main] Action '{action.name}'  frames {action.frame_range[0]:.0f}-{action.frame_range[1]:.0f}  fcurves: {len(action.fcurves)}")

    # ── 7. Post-corrections ───────────────────────────────────────────
    post_correct_action(action, correct_root=True, smooth=True, zero_fingers=True)

    # ── 7b. Straight wrist fix ────────────────────────────────────────
    # BVH has no data past the wrist joint, so the retargeter leaves
    # garbage rotation on the Mixamo Hand bones causing bent/twisted wrists.
    #
    # Fix: write quaternion identity (W=1, X=Y=Z=0) on the hand bone.
    # In LOCAL space, identity = "no rotation on top of parent" = hand
    # continues straight from the forearm. This is correct because BVH
    # has zero intentional wrist data.
    #
    # rotation_mode is set to QUATERNION before the NLA bake in export
    # so Blender evaluates these curves correctly.
    fix_wrist_rotation(action)

    # ── 8. Remove BVH source armature ─────────────────────────────────
    log("[Main] Removing BVH source skeleton...")
    for obj in list(bvh_objects):
        try:
            bpy.data.objects.remove(obj, do_unlink=True)
        except Exception:
            pass

    remove_unwanted_objects()

    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end   = num_frames
    log(f"[Main] Scene frame range: 1 -> {num_frames}")

    if "GroundPlane" not in bpy.data.objects:
        bpy.ops.mesh.primitive_plane_add(size=5, location=(0, 0, -0.05))
        ground      = bpy.context.active_object
        ground.name = "GroundPlane"
        log("[Main] GroundPlane created.")

    auto_ground_avatar(armature_name=tgt_arm.name, ground_name="GroundPlane",
                       frame_start=1, frame_end=num_frames,
                       step=2, mode="contact", prefer_above=True)

    export_fbx_with_animation(OUTPUT_FBX_PATH, action_name="Retargeted_Action",
                               export_mesh=export_mesh)
    log("[Main] Pipeline complete.")

# =====================================================================
#  ENTRY POINT
# =====================================================================

if __name__ == "__main__":
    def parse_args():
        if "--" in sys.argv:
            script_args = sys.argv[sys.argv.index("--") + 1:]
        else:
            script_args = []
        parser = argparse.ArgumentParser(description="BVH->FBX Retarget (Rokoko/Native)")
        parser.add_argument("--video_render", type=str, default="false",
                            help="Render MP4 video (true/false)")
        parser.add_argument("--export_mesh", type=str, default="false",
                            help="Include mesh in exported FBX (true/false).")
        return parser.parse_args(script_args)

    args     = parse_args()
    do_mesh  = args.export_mesh.lower() == "true"
    do_video = args.video_render.lower() == "true"

    main(export_mesh=do_mesh)

    if do_video:
        title_text = read_video_title()
        export_video_with_title(
            title=title_text,
            output_mp4_path="./video_result/Final_Fbx_Mesh_Animation.mp4",
            resolution_x=1920, resolution_y=1080, fps=30, frame_end=None,
        )