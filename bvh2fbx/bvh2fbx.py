import bpy
import zipfile
import os
import addon_utils
import sys
import textwrap
import argparse
from mathutils import Vector
import math

from bpy.props import PointerProperty, IntProperty, StringProperty
from bpy.types import PropertyGroup

# ---------------------------------------------------------------------
# Configuration paths
# ---------------------------------------------------------------------
ADDON_NAME = "KeeMapAnimRetarget"

LOG_FILE_PATH = "./bvh2fbx/log.txt"
INPUT_BVH_FILE_PATH = "./bvh_folder/bvh_0_out.bvh"
# INPUT_FBX_FILE_PATH = "./bone_mapping_asset/Brian_model.fbx"
INPUT_FBX_FILE_PATH = "./avatar_model/Brian_model.fbx"

OUTPUT_FBX_PATH = "./fbx_folder/bvh_0_out.fbx"
OUTPUT_ZIP_PATH = "./fbx_zip_folder/bvh_0_out.zip"
BONE_MAPPING_FILE_PATH = "./bone_mapping_asset/mapping_mixamo_brianmodel.json"



# ---------------------------------------------------------------------
# Utility: logging to file + stdout
# ---------------------------------------------------------------------
def log(msg):
    os.makedirs(os.path.dirname(LOG_FILE_PATH), exist_ok=True)
    try:
        with open(LOG_FILE_PATH, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
    except Exception:
        pass
    print(msg)


# ---------------------------------------------------------------------
# Step 1: Enable your add-on via Blender's --addons flag
# ---------------------------------------------------------------------
def enable_keemap_addon():
    try:
        addon_utils.enable(ADDON_NAME, default_set=True, persistent=True)
        log(f"KeeMap add-on '{ADDON_NAME}' enabled.")
    except Exception as e:
        log(f"Failed to enable KeeMap add-on: {e}")


# ---------------------------------------------------------------------
# Fallback stub so scene.keemap_settings never blows up
# ---------------------------------------------------------------------
class KeemapSettingsStub(PropertyGroup):
    start_frame_to_apply: IntProperty(default=1)
    number_of_frames_to_apply: IntProperty(default=268)
    keyframe_every_n_frames: IntProperty(default=1)
    source_rig_name: StringProperty(default="")
    destination_rig_name: StringProperty(default="")
    bone_mapping_file: StringProperty(default="")


def ensure_keemap_settings():
    if not hasattr(bpy.types.Scene, "keemap_settings"):
        bpy.utils.register_class(KeemapSettingsStub)
        bpy.types.Scene.keemap_settings = PointerProperty(type=KeemapSettingsStub)


# ---------------------------------------------------------------------
# Step 2: Load models
# ---------------------------------------------------------------------
def load_and_debug_fbx(path, scale_factor=1.0, ground_to_z=0.0):
    log(f"Importing FBX: {path} @ scale={scale_factor}")
    before = set(bpy.context.scene.objects)
    bpy.ops.import_scene.fbx(filepath=path, global_scale=scale_factor)
    bpy.ops.file.pack_all()
    after = set(bpy.context.scene.objects)
    imported = list(after - before)

    if not imported:
        log("No objects were imported!")
        return []

    log("Imported objects:")
    for obj in imported:
        log(f"  - {obj.name} ({obj.type})")
        obj.hide_set(False)
        obj.hide_viewport = False
        obj.hide_render = False

    for obj in imported:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = imported[0]

    for area in bpy.context.screen.areas:
        if area.type == "VIEW_3D":
            for region in area.regions:
                if region.type == "WINDOW":
                    override = {"area": area, "region": region}
                    try:
                        bpy.ops.view3d.view_selected(override)
                    except Exception:
                        pass

    return imported


def load_bvh_file(
    path,
    scale=0.02,
    rotate_mode="NATIVE",
    axis_forward="-Z",
    axis_up="Y",
    start_frame=1,
    use_fps_scale=False,
    update_scene_fps=False,
    update_scene_duration=False,
):
    log(f"Loading BVH file: {path} (scale={scale})")
    bpy.ops.import_anim.bvh(
        filepath=path,
        target="ARMATURE",
        global_scale=scale,
        frame_start=start_frame,
        use_fps_scale=use_fps_scale,
        update_scene_fps=update_scene_fps,
        update_scene_duration=update_scene_duration,
        rotate_mode=rotate_mode,
        axis_forward=axis_forward,
        axis_up=axis_up,
    )


# ---------------------------------------------------------------------
# Step 3: Configure read-mapping settings
# ---------------------------------------------------------------------
def readBone_setup_scene_settings():
    ensure_keemap_settings()
    ks = bpy.context.scene.keemap_settings
    ks.start_frame_to_apply = 1
    ks.number_of_frames_to_apply = 268
    ks.keyframe_every_n_frames = 1
    ks.bone_mapping_file = BONE_MAPPING_FILE_PATH


# ---------------------------------------------------------------------
# Step 4: Call the add-on operators
# ---------------------------------------------------------------------
def keemap_read_file():
    log("Running keemap_read_file()")
    bpy.ops.wm.keemap_read_file()


def perform_animation_transfer():
    log("Running perform_animation_transfer()")
    bpy.ops.wm.perform_animation_transfer()


# ---------------------------------------------------------------------
# Step 5: Clean up extras
# ---------------------------------------------------------------------
def remove_unwanted_objects():
    log("Removing unwanted objects...")
    for obj in list(bpy.data.objects):
        if obj.type == "MESH" and obj.name.startswith("Cube"):
            bpy.data.objects.remove(obj, do_unlink=True)


# ---------------------------------------------------------------------
# Step 6: Pose bone selection
# ---------------------------------------------------------------------
def select_rigs_and_bones():
    scene = bpy.context.scene
    ks = scene.keemap_settings
    log(f"Selecting rigs '{ks.source_rig_name}' -> '{ks.destination_rig_name}'")
    for name in (ks.source_rig_name, ks.destination_rig_name):
        obj = bpy.data.objects.get(name)
        if not obj:
            log(f"Could not find rig: {name}")
            sys.exit(1)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode="POSE")


# ---------------------------------------------------------------------
# Step 7: Export + zip
# ---------------------------------------------------------------------
def export_fbx_with_animation(out_path):
    log(f"Exporting FBX Animation Only to: {out_path}")
    try:
        bpy.ops.export_scene.fbx(
            filepath=out_path,
            use_selection=False,
            apply_unit_scale=True,
            bake_space_transform=True,
            object_types={"ARMATURE"},
            bake_anim=True,
            bake_anim_use_all_bones=True,
            bake_anim_force_startend_keying=True,
            bake_anim_simplify_factor=1.0,
            add_leaf_bones=False,
            primary_bone_axis="Y",
            secondary_bone_axis="X",
            axis_up="Y",
            axis_forward="Z",
            path_mode="COPY",
            embed_textures=False,
        )

        if os.path.exists(OUTPUT_ZIP_PATH):
            os.remove(OUTPUT_ZIP_PATH)

        with zipfile.ZipFile(OUTPUT_ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(out_path, os.path.basename(out_path))

        log(f"Created zip at: {OUTPUT_ZIP_PATH}")
    except Exception as e:
        log(f"Export failed: {e}")


# ---------------------------------------------------------------------
# Step 8: Ground the armature by its foot bone (kept intact)
# ---------------------------------------------------------------------
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
    offset = ground_z - foot_head_world.z

    bpy.ops.object.mode_set(mode="OBJECT")
    armature.location.z += offset
    log(f"Grounded '{armature_name}' using bone '{foot_bone_name}' (offset {offset:.4f}).")


# ---------------------------------------------------------------------
# Step 9: (Optional) wrap text for video titles
# ---------------------------------------------------------------------
# def wrap_text_to_fit_plane(original_text, max_width_chars=70, max_lines=4):
#     wrapped = textwrap.wrap(original_text, width=max_width_chars)
#     if len(wrapped) > max_lines:
#         wrapped = wrapped[:max_lines]
#         wrapped[-1] += "..."
#     return "\n".join(wrapped)
def wrap_text_to_fit_plane(original_text, max_width_chars=30, max_lines=4):
    # Detect Japanese / CJK
    is_cjk = any(
        '\u3040' <= c <= '\u30ff' or '\u4e00' <= c <= '\u9fff'
        for c in original_text
    )

    if is_cjk:
        # Character-based wrapping (Japanese-safe)
        lines = [
            original_text[i:i + max_width_chars]
            for i in range(0, len(original_text), max_width_chars)
        ]
    else:
        # Word-based wrapping (English)
        lines = textwrap.wrap(original_text, width=max_width_chars)

    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] += "..."

    return "\n".join(lines)



def _ensure_parent_dir(path: str) -> str:
    abs_path = os.path.abspath(os.path.expanduser(path))
    parent = os.path.dirname(abs_path)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent, exist_ok=True)
    return abs_path


# ---------------------------------------------------------------------
# Automatic grounding helpers (kept as you provided)
# ---------------------------------------------------------------------
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
            if (min_z is None) or (z < min_z):
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
            if (min_z is None) or (z < min_z):
                min_z = z
        eo.to_mesh_clear()

    if min_z is None:
        raise RuntimeError("Could not compute lowest vertex (no evaluable geometry).")
    return min_z


def detect_contact_frame_by_scan(armature_name, frame_start=1, frame_end=268, step=1, floor_z=None, prefer_above=True):
    if frame_end < frame_start:
        frame_start, frame_end = frame_end, frame_start

    floor_z = floor_z if floor_z is not None else 0.0

    best_frame, best_z, best_score = None, None, None
    for f in range(frame_start, frame_end + 1, step):
        z = lowest_vertex_z_at_frame(armature_name, f)
        if prefer_above:
            if z >= floor_z:
                score = z - floor_z
            else:
                score = (floor_z - z) + 1e6
        else:
            score = abs(z - floor_z)

        if (best_score is None) or (score < best_score):
            best_frame, best_z, best_score = f, z, score

    return best_frame, best_z


def auto_ground_avatar(
    armature_name="Armature",
    ground_name="GroundPlane",
    frame_start=1,
    frame_end=268,
    step=1,
    mode="contact",
    prefer_above=True,
):
    arm = bpy.data.objects.get(armature_name)
    if not arm:
        log(f"[auto_ground] Armature '{armature_name}' not found.")
        return

    floor_z = detect_floor_z(ground_name=ground_name, fallback_default=0.0, armature_name=armature_name)

    if mode == "no_penetration":
        global_min = None
        min_frame = None
        for f in range(frame_start, frame_end + 1, step):
            z = lowest_vertex_z_at_frame(armature_name, f)
            if (global_min is None) or (z < global_min):
                global_min, min_frame = z, f
        offset = floor_z - global_min
        old_z = arm.location.z
        arm.location.z += offset
        log(
            f"[auto_ground] Mode=no_penetration | floor_z={floor_z:.4f} | "
            f"global_min={global_min:.4f} at frame {min_frame} | "
            f"applied offset={offset:.4f} (arm.z {old_z:.4f}->{arm.location.z:.4f})"
        )
        return

    best_frame, best_z = detect_contact_frame_by_scan(
        armature_name, frame_start, frame_end, step, floor_z=floor_z, prefer_above=prefer_above
    )
    offset = floor_z - best_z
    old_z = arm.location.z
    arm.location.z += offset
    log(
        f"[auto_ground] Mode=contact | floor_z={floor_z:.4f} | "
        f"contact_frame={best_frame} lowest_z={best_z:.4f} | "
        f"applied offset={offset:.4f} (arm.z {old_z:.4f}->{arm.location.z:.4f})"
    )


# ---------------------------------------------------------------------
# Armature selection helpers
# ---------------------------------------------------------------------
def _armature_driven_mesh_count(arm_obj) -> int:
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


def _armature_has_action(arm_obj) -> bool:
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
        driven = _armature_driven_mesh_count(a)
        has_anim = 1 if _armature_has_action(a) else 0
        name_bonus = 0
        n = a.name.lower()
        if "mixamo" in n or "armature" in n or "rig" in n:
            name_bonus = 1
        scored.append((driven, has_anim, name_bonus, a))

    scored.sort(key=lambda t: (t[0], t[1], t[2]), reverse=True)
    best = scored[0][3]
    log(f"[Video] Selected avatar armature: {best.name} (driven_meshes={scored[0][0]}, has_action={bool(scored[0][1])})")
    return best


# ---------------------------------------------------------------------
# Shared helper: choose root/hips bone
# ---------------------------------------------------------------------
def _pick_root_bone_name(armature_obj):
    if not armature_obj or armature_obj.type != "ARMATURE":
        return None

    candidates = ["Hips", "mixamorig:Hips", "hips", "Pelvis", "pelvis", "Root", "root"]

    if not armature_obj.pose:
        return None

    for n in candidates:
        if n in armature_obj.pose.bones:
            return n

    if armature_obj.pose.bones:
        return armature_obj.pose.bones[0].name

    return None


# ---------------------------------------------------------------------
# Camera follow helper
# ---------------------------------------------------------------------
def setup_follow_camera_for_avatar(
    scene,
    avatar_obj,
    cam_obj,
    offset_world=(0.0, -6.0, 3.5),
    target_empty_name="CamTarget",
    camera_lens_mm=35,
    bone_name=None,
):
    if not avatar_obj or avatar_obj.type != "ARMATURE":
        raise RuntimeError("setup_follow_camera_for_avatar: avatar_obj is missing or not an ARMATURE.")

    if bone_name is None:
        bone_name = _pick_root_bone_name(avatar_obj)

    try:
        cam_obj.data.lens = camera_lens_mm
        cam_obj.data.clip_start = 0.01
        cam_obj.data.clip_end = 1000.0
    except Exception:
        pass

    empty = bpy.data.objects.get(target_empty_name)
    if empty is None:
        empty = bpy.data.objects.new(target_empty_name, None)
        empty.empty_display_type = "PLAIN_AXES"
        empty.empty_display_size = 0.2
        scene.collection.objects.link(empty)

    if bone_name and avatar_obj.pose and bone_name in avatar_obj.pose.bones:
        empty.parent = avatar_obj
        empty.parent_type = "BONE"
        empty.parent_bone = bone_name
        empty.location = (0.0, 0.0, 0.0)
    else:
        empty.parent = avatar_obj
        empty.parent_type = "OBJECT"
        empty.location = (0.0, 0.0, 0.0)

    cam_obj.parent = None
    cam_obj.constraints.clear()

    c_loc = cam_obj.constraints.new(type="COPY_LOCATION")
    c_loc.target = empty
    c_loc.use_offset = True
    cam_obj.location = Vector(offset_world)

    c_track = cam_obj.constraints.new(type="DAMPED_TRACK")
    c_track.target = empty
    c_track.track_axis = "TRACK_NEGATIVE_Z"

    log(f"[Video] Camera follow enabled (target bone='{bone_name or 'None'}', offset={offset_world}).")


# ---------------------------------------------------------------------
# Ground follow helper
# ---------------------------------------------------------------------
def move_ground_with_avatar_bone(
    armature_obj,
    ground_obj,
    bone_name=None,
    frame_start=1,
    frame_end=268,
    z_offset=-0.05,
    follow_z=False,
):
    if not armature_obj or armature_obj.type != "ARMATURE":
        raise RuntimeError("move_ground_with_avatar_bone: armature_obj is missing or not an ARMATURE.")
    if not ground_obj or ground_obj.type != "MESH":
        raise RuntimeError("move_ground_with_avatar_bone: ground_obj is missing or not a MESH.")

    if bone_name is None:
        bone_name = _pick_root_bone_name(armature_obj)

    if not bone_name or not armature_obj.pose or bone_name not in armature_obj.pose.bones:
        raise RuntimeError("move_ground_with_avatar_bone: could not find a usable root/hips bone.")

    scene = bpy.context.scene

    if ground_obj.animation_data and ground_obj.animation_data.action:
        act = ground_obj.animation_data.action
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
        pb = armature_obj.pose.bones[bone_name]
        bone_world = armature_obj.matrix_world @ pb.head

        ground_obj.location.x = bone_world.x
        ground_obj.location.y = bone_world.y

        if follow_z:
            ground_obj.location.z = bone_world.z + float(z_offset)
        else:
            ground_obj.location.z = fixed_ground_z

        ground_obj.keyframe_insert(data_path="location", frame=f)

    action = ground_obj.animation_data.action if ground_obj.animation_data else None
    if action:
        for fc in action.fcurves:
            if fc.data_path == "location":
                for kp in fc.keyframe_points:
                    kp.interpolation = "LINEAR"



#disable all scene lights
def setup_lighting(scene):
    # Remove existing lights
    for obj in list(bpy.data.objects):
        if obj.type == "LIGHT":
            bpy.data.objects.remove(obj, do_unlink=True)

    # World ambient (softens darkness)
    world = scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs["Strength"].default_value = 0.8  # increase to brighten overall scene

    # Key light (Sun) from front-top
    sun_data = bpy.data.lights.new(name="KeySun", type="SUN")
    sun_data.energy = 2.0
    sun_data.angle = math.radians(5.0)  # softer shadow edges
    sun_obj = bpy.data.objects.new("KeySun", sun_data)
    scene.collection.objects.link(sun_obj)
    sun_obj.rotation_euler = (math.radians(50.0), 0.0, math.radians(20.0))

    # Fill light (Area) near camera to brighten front
    fill_data = bpy.data.lights.new(name="FillArea", type="AREA")
    fill_data.energy = 800.0
    fill_data.size = 4.0
    fill_obj = bpy.data.objects.new("FillArea", fill_data)
    scene.collection.objects.link(fill_obj)
    fill_obj.location = (0.0, -4.5, 3.0)
    fill_obj.rotation_euler = (math.radians(65.0), 0.0, 0.0)

    # Rim light (optional)
    rim_data = bpy.data.lights.new(name="RimArea", type="AREA")
    rim_data.energy = 300.0
    rim_data.size = 3.0
    rim_obj = bpy.data.objects.new("RimArea", rim_data)
    scene.collection.objects.link(rim_obj)
    rim_obj.location = (2.5, 2.5, 3.0)
    rim_obj.rotation_euler = (math.radians(-35.0), 0.0, math.radians(135.0))
# ---------------------------------------------------------------------
# Title helper: FIXED HUD logic (does not move with avatar, stays on screen)
# ---------------------------------------------------------------------
# def setup_title_text(scene, cam_obj, title):
#     wrapped_title = wrap_text_to_fit_plane(title, max_width_chars=70, max_lines=4)
#
#     if "FootTitleText" in bpy.data.objects:
#         text_obj = bpy.data.objects["FootTitleText"]
#         text_obj.data.body = wrapped_title
#     else:
#         bpy.ops.object.text_add(location=(0.0, 0.0, 0.0))
#         text_obj = bpy.context.active_object
#         text_obj.name = "FootTitleText"
#         text_obj.data.body = wrapped_title
#         # text_obj.data.size = 0.14
#         if any('\u3040' <= c <= '\u30ff' or '\u4e00' <= c <= '\u9fff' for c in wrapped_title):
#             text_obj.data.size = 0.10
#         else:
#             text_obj.data.size = 0.14
#
#         text_obj.data.align_x = "CENTER"
#         text_obj.data.align_y = "CENTER"
#
#         text_mat = bpy.data.materials.new(name="TextMat")
#         text_mat.use_nodes = True
#         nodes = text_mat.node_tree.nodes
#         links = text_mat.node_tree.links
#         nodes.clear()
#         emission = nodes.new("ShaderNodeEmission")
#         emission.inputs["Color"].default_value = (1.0, 0.8, 0.0, 1.0)
#         emission.inputs["Strength"].default_value = 8.0
#         output = nodes.new("ShaderNodeOutputMaterial")
#         links.new(emission.outputs["Emission"], output.inputs["Surface"])
#         text_obj.data.materials.append(text_mat)
#
#     text_obj.hide_render = False
#     text_obj.hide_viewport = False
#     text_obj.hide_set(False)
#
#     # Anchor to camera so it stays fixed on screen
#     text_obj.constraints.clear()
#     text_obj.parent = cam_obj
#     text_obj.parent_type = "OBJECT"
#     text_obj.matrix_parent_inverse = cam_obj.matrix_world.inverted()
#
#     # Camera local axes: X right, Y up, -Z forward
#    #HUD_OFFSET = (-2.0, -2.0, 2.0)  # change this if not positioned well
#     #HUD_OFFSET = (0, 0, 0)  # change this if not positioned well
#     if any('\u3040' <= c <= '\u30ff' or '\u4e00' <= c <= '\u9fff' for c in wrapped_title):
#         HUD_OFFSET = (0.0, -2.0, 2.0)
#     else:
#         HUD_OFFSET = (0.0, 0.0, 0.0)
#         #HUD_OFFSET = (-1.0, -2.0, 2.0)
#
#     text_obj.location = HUD_OFFSET
#
#     # If text looks backwards, set this to math.pi
#     TEXT_FACE_FLIP = 0.0
#     text_obj.rotation_euler = (math.radians(90.0), TEXT_FACE_FLIP, 0.0)
#
#     log("[Video] Title ready (HUD camera-anchored).")
#
#
#
#
#
#
#
#
# # ---------------------------------------------------------------------
# # Video export
# # ---------------------------------------------------------------------
# def export_video_with_title(
#     title,
#     output_mp4_path,
#     resolution_x=1920,
#     resolution_y=1080,
#     fps=30,
#     test_mode=False,
# ):
#     scene = bpy.context.scene
#     setup_lighting(scene)
#     if not output_mp4_path.lower().endswith(".mp4"):
#         output_mp4_path += ".mp4"
#     output_mp4_path = _ensure_parent_dir(output_mp4_path)
#     log(f"[Video] Final MP4 path: {output_mp4_path}")
#
#     engines = bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items.keys()
#     if "BLENDER_EEVEE_NEXT" in engines:
#         scene.render.engine = "BLENDER_EEVEE_NEXT"
#     elif "BLENDER_EEVEE" in engines:
#         scene.render.engine = "BLENDER_EEVEE"
#     else:
#         scene.render.engine = "CYCLES"
#
#     scene.render.use_sequencer = False
#     scene.render.use_compositing = False
#
#     if scene.render.engine.startswith("BLENDER_EEVEE"):
#         scene.render.use_motion_blur = False
#         if hasattr(scene.eevee, "taa_render_samples"):
#             scene.eevee.taa_render_samples = 8
#         if hasattr(scene.eevee, "use_soft_shadows"):
#             scene.eevee.use_soft_shadows = False
#
#     if test_mode:
#         resolution_x, resolution_y = 960, 540
#     scene.render.resolution_x = resolution_x
#     scene.render.resolution_y = resolution_y
#     scene.render.resolution_percentage = 100
#     scene.render.fps = fps
#
#     scene.frame_start = 1
#     scene.frame_end = 268
#
#     if scene.camera is None:
#         cam_data = bpy.data.cameras.new("Camera")
#         cam_obj = bpy.data.objects.new("Camera", cam_data)
#         scene.collection.objects.link(cam_obj)
#         scene.camera = cam_obj
#     else:
#         cam_obj = scene.camera
#
#     cam_obj.location = (0, -6, 3.5)
#     cam_obj.rotation_euler = (1.15, 0, 0)
#
#     ground = bpy.data.objects.get("GroundPlane")
#     if ground is None:
#         bpy.ops.mesh.primitive_plane_add(size=20, location=(0, 0, -0.05))
#         ground = bpy.context.active_object
#         ground.name = "GroundPlane"
#         mat = bpy.data.materials.new(name="GroundMat")
#         mat.diffuse_color = (0.5, 0.5, 0.5, 1)
#         ground.data.materials.append(mat)
#
#     avatar_obj = find_primary_character_armature()
#     if not avatar_obj:
#         raise RuntimeError("No avatar (ARMATURE) found in the scene.")
#
#     if not hasattr(avatar_obj, "_title_nudged"):
#         avatar_obj.location.y += 1.5
#         avatar_obj["_title_nudged"] = True
#
#     setup_follow_camera_for_avatar(
#         scene=scene,
#         avatar_obj=avatar_obj,
#         cam_obj=cam_obj,
#         offset_world=(0.0, -6.0, 3.5),
#         target_empty_name="CamTarget",
#         camera_lens_mm=35,
#         bone_name=None,
#     )
#
#     move_ground_with_avatar_bone(
#         armature_obj=avatar_obj,
#         ground_obj=ground,
#         bone_name=None,
#         frame_start=scene.frame_start,
#         frame_end=scene.frame_end,
#         z_offset=-0.05,
#         follow_z=False,
#     )
#
#     # Correct call: pass camera, not avatar
#     setup_title_text(scene, cam_obj, title)
#
#     scene.render.image_settings.file_format = "FFMPEG"
#     scene.render.ffmpeg.format = "MPEG4"
#     scene.render.ffmpeg.codec = "H264"
#     scene.render.ffmpeg.constant_rate_factor = "HIGH"
#     scene.render.ffmpeg.ffmpeg_preset = "GOOD"
#     scene.render.use_overwrite = True
#     scene.render.use_file_extension = True
#
#     try:
#         scene.render.ffmpeg.audio_codec = "NONE"
#     except Exception:
#         pass
#
#     scene.render.filepath = output_mp4_path
#
#     log(f"[Video] Rendering to: {scene.render.filepath}")
#     bpy.ops.render.render(animation=True)
#     log("[Video] Render complete.")


def setup_vse_scene_with_title_overlay(
    render_scene,
    title,
    frame_start=1,
    frame_end=268,
    font_size=56,
    y_location=0.94,
    vse_scene_name="VSE_Output_Scene",
):
    """
    Creates/uses a dedicated VSE scene that contains:
      - a SCENE strip referencing render_scene (so avatar render appears)
      - a TEXT strip on top (fixed 2D title)

    Returns the VSE scene to render from.
    """
    # Create or reuse a separate VSE scene (more stable than referencing itself)
    vse_scene = bpy.data.scenes.get(vse_scene_name)
    if vse_scene is None:
        vse_scene = bpy.data.scenes.new(vse_scene_name)

    # Match timing
    vse_scene.frame_start = frame_start
    vse_scene.frame_end = frame_end

    # Ensure sequence editor
    vse_scene.sequence_editor_create()
    seq = vse_scene.sequence_editor

    # Clear previous strips (avoid stacking old strips)
    for s in list(seq.sequences_all):
        seq.sequences.remove(s)

    # 1) Add Scene strip (your actual 3D render)
    scene_strip = seq.sequences.new_scene(
        name="HUD_Scene",
        scene=render_scene,
        channel=1,
        frame_start=frame_start,
    )
    scene_strip.frame_final_end = frame_end + 1

    # 2) Add Text strip (fixed overlay)
    wrapped = wrap_text_to_fit_plane(title, max_width_chars=70, max_lines=999)

    txt = seq.sequences.new_effect(
        name="HUD_Title",
        type="TEXT",
        channel=10,
        frame_start=frame_start,
        frame_end=frame_end + 1,
    )

    txt.text = wrapped
    txt.font_size = int(font_size)

    # Normalized screen coords (0..1)
    txt.location = (0.5, float(y_location))
    txt.align_x = "CENTER"
    txt.align_y = "TOP"

    # Styling
    txt.color = (1.0, 0.85, 0.1, 1.0)
    txt.use_shadow = True
    txt.shadow_color = (0.0, 0.0, 0.0, 1.0)
    txt.shadow_blur = 0.3

    txt.use_outline = True
    txt.outline_color = (0.0, 0.0, 0.0, 1.0)

    log("[VSE] Created VSE scene with Scene strip + fixed 2D Title overlay.")
    return vse_scene



def export_video_with_title(
    title,
    output_mp4_path,
    resolution_x=1920,
    resolution_y=1080,
    fps=30,
):
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

    # Fixed render range
    render_scene.frame_start = 1
    render_scene.frame_end = 268

    # Ensure camera exists
    if render_scene.camera is None:
        cam_data = bpy.data.cameras.new("Camera")
        cam_obj = bpy.data.objects.new("Camera", cam_data)
        render_scene.collection.objects.link(cam_obj)
        render_scene.camera = cam_obj
    else:
        cam_obj = render_scene.camera

    cam_obj.location = (0, -6, 3.5)
    cam_obj.rotation_euler = (1.15, 0, 0)

    ground = bpy.data.objects.get("GroundPlane")
    if ground is None:
        bpy.ops.mesh.primitive_plane_add(size=20, location=(0, 0, -0.05))
        ground = bpy.context.active_object
        ground.name = "GroundPlane"
        mat = bpy.data.materials.new(name="GroundMat")
        mat.diffuse_color = (0.5, 0.5, 0.5, 1)
        ground.data.materials.append(mat)

    avatar_obj = find_primary_character_armature()
    if not avatar_obj:
        raise RuntimeError("No avatar (ARMATURE) found in the scene.")

    setup_follow_camera_for_avatar(
        scene=render_scene,
        avatar_obj=avatar_obj,
        cam_obj=cam_obj,
        offset_world=(0.0, -6.0, 3.5),
        target_empty_name="CamTarget",
        camera_lens_mm=35,
        bone_name=None,
    )

    move_ground_with_avatar_bone(
        armature_obj=avatar_obj,
        ground_obj=ground,
        bone_name=None,
        frame_start=render_scene.frame_start,
        frame_end=render_scene.frame_end,
        z_offset=-0.05,
        follow_z=False,
    )

    # Build a dedicated VSE scene to output the final video with overlay
    vse_scene = setup_vse_scene_with_title_overlay(
        render_scene=render_scene,
        title=title,
        frame_start=render_scene.frame_start,
        frame_end=render_scene.frame_end,
        font_size=36,       # fixed
        y_location=0.94,    # fixed
    )

    # Configure render settings on the VSE output scene
    vse_scene.render.use_sequencer = True
    vse_scene.render.use_compositing = False

    vse_scene.render.resolution_x = resolution_x
    vse_scene.render.resolution_y = resolution_y
    vse_scene.render.resolution_percentage = 100
    vse_scene.render.fps = fps

    vse_scene.render.image_settings.file_format = "FFMPEG"
    vse_scene.render.ffmpeg.format = "MPEG4"
    vse_scene.render.ffmpeg.codec = "H264"
    vse_scene.render.ffmpeg.constant_rate_factor = "HIGH"
    vse_scene.render.ffmpeg.ffmpeg_preset = "GOOD"
    vse_scene.render.use_overwrite = True
    vse_scene.render.use_file_extension = True

    try:
        vse_scene.render.ffmpeg.audio_codec = "NONE"
    except Exception:
        pass

    vse_scene.render.filepath = output_mp4_path

    log(f"[Video] Rendering VSE output scene to: {vse_scene.render.filepath}")
    bpy.ops.render.render(animation=True, scene=vse_scene.name)
    log("[Video] Render complete.")




# ---------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------
def main():
    log("Script started")

    SCALE = 1.0
    GROUND = 0.0
    imported_objs = load_and_debug_fbx(INPUT_FBX_FILE_PATH, scale_factor=SCALE, ground_to_z=GROUND)
    if not imported_objs:
        log("Aborting: no FBX loaded.")
        return

    enable_keemap_addon()
    load_bvh_file(INPUT_BVH_FILE_PATH, scale=0.02)

    readBone_setup_scene_settings()
    keemap_read_file()
    remove_unwanted_objects()
    select_rigs_and_bones()
    perform_animation_transfer()

    if "GroundPlane" not in bpy.data.objects:
        bpy.ops.mesh.primitive_plane_add(size=5, location=(0, 0, -0.05))
        ground = bpy.context.active_object
        ground.name = "GroundPlane"

    auto_ground_avatar(
        armature_name="Armature",
        ground_name="GroundPlane",
        frame_start=1,
        frame_end=268,
        step=2,
        mode="contact",
        prefer_above=True,
    )

    export_fbx_with_animation(OUTPUT_FBX_PATH)
    log("Script completed successfully.")


if __name__ == "__main__":
    def parse_args():
        if "--" in sys.argv:
            idx = sys.argv.index("--")
            script_args = sys.argv[idx + 1:]
        else:
            script_args = []
        parser = argparse.ArgumentParser(description="GaitSimPT Controller")
        parser.add_argument("--video_render", type=str, default="false", help="Render MP4 video (true/false)")
        # parser.add_argument("--high_res", type=str, default="false", help="Use high resolution output (true/false)")
        return parser.parse_args(script_args)

    args = parse_args()
    main()

    if args.video_render.lower() == "true":
        input_txt_path = "./input.txt"
        if not os.path.isfile(input_txt_path):
            raise FileNotFoundError(f"Could not find: {input_txt_path}")
        with open(input_txt_path, "r", encoding="utf-8") as f:
            title_text = f.read().strip()

        export_video_with_title(
            title=title_text,
            output_mp4_path="./video_result/Final_Fbx_Mesh_Animation.mp4",
            resolution_x=1920,
            resolution_y=1080,
            fps=30,
        )

