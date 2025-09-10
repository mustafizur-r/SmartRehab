import bpy
import zipfile
import os
import addon_utils
import sys
import textwrap
import argparse
from mathutils import Vector

from bpy.props import PointerProperty, IntProperty, StringProperty
from bpy.types import PropertyGroup

#––– Configuration paths –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
ADDON_NAME             = "KeeMapAnimRetarget"

LOG_FILE_PATH          = "./bvh2fbx/log.txt"
INPUT_BVH_FILE_PATH    = "./bvh_folder/bvh_0_out.bvh"

INPUT_FBX_FILE_PATH    = "./bone_mapping_asset/Brian_model.fbx"

OUTPUT_FBX_PATH        = "./fbx_folder/bvh_0_out.fbx"
OUTPUT_ZIP_PATH        = "./fbx_zip_folder/bvh_0_out.zip"
BONE_MAPPING_FILE_PATH = "./bone_mapping_asset/mapping_mixamo_brianmodel.json"


#––– Utility: logging to file + stdout –––––––––––––––––––––––––––––––––––––––––––––––––––––––––
def log(msg):
    os.makedirs(os.path.dirname(LOG_FILE_PATH), exist_ok=True)
    try:
        with open(LOG_FILE_PATH, "a") as f:
            f.write(msg + "\n")
    except Exception:
        pass
    print(msg)


#––– Step 1: Enable your add-on via Blender’s --addons flag ––––––––––––––––––––––––––––––––––––
def enable_keemap_addon():
    try:
        addon_utils.enable(ADDON_NAME, default_set=True, persistent=True)
        log(f"KeeMap add-on '{ADDON_NAME}' enabled.")
    except Exception as e:
        log(f"Failed to enable KeeMap add-on: {e}")


#––– Fallback stub so scene.keemap_settings never blows up –––––––––––––––––––––––––––––––––––––
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


#––– Step 2: Load models –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
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
        log(f"  • {obj.name} ({obj.type})")
        obj.hide_set(False)
        obj.hide_viewport = False
        obj.hide_render   = False

    # Instead of bounding‐box grounding, we do nothing here.
    # The dedicated grounding function runs later, once the armature is in place.
    for obj in imported:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = imported[0]

    for area in bpy.context.screen.areas:
        if area.type == 'VIEW_3D':
            for region in area.regions:
                if region.type == 'WINDOW':
                    override = {'area': area, 'region': region}
                    try:
                        bpy.ops.view3d.view_selected(override)
                    except Exception:
                        pass

    return imported


# def load_bvh_file(path):
#     log(f"Loading BVH file: {path}")
#     bpy.ops.import_anim.bvh(filepath=path)

def load_bvh_file(path, scale=0.02,
                  rotate_mode='NATIVE',
                  axis_forward='-Z', axis_up='Y',
                  start_frame=1,
                  use_fps_scale=False,
                  update_scene_fps=False,
                  update_scene_duration=False):
    """
    Imports a BVH with a specific scale (default 0.02).
    Parameters mirror Blender's BVH importer.
    """
    log(f"Loading BVH file: {path} (scale={scale})")
    bpy.ops.import_anim.bvh(
        filepath=path,
        target='ARMATURE',
        global_scale=scale,
        frame_start=start_frame,
        use_fps_scale=use_fps_scale,
        update_scene_fps=update_scene_fps,
        update_scene_duration=update_scene_duration,
        rotate_mode=rotate_mode,
        axis_forward=axis_forward,
        axis_up=axis_up
    )


#––– Step 3: Configure read‐mapping settings ––––––––––––––––––––––––––––––––––––––––––––––––––
def readBone_setup_scene_settings():
    ensure_keemap_settings()
    ks = bpy.context.scene.keemap_settings
    ks.start_frame_to_apply      = 1
    ks.number_of_frames_to_apply = 268
    ks.keyframe_every_n_frames   = 1
    ks.bone_mapping_file         = BONE_MAPPING_FILE_PATH


#––– Step 4: Call the add-on operators –––––––––––––––––––––––––––––––––––––––––––––––––––––––––
def keemap_read_file():
    log("Running keemap_read_file()")
    bpy.ops.wm.keemap_read_file()


def perform_animation_transfer():
    log("Running perform_animation_transfer()")
    bpy.ops.wm.perform_animation_transfer()


#––– Step 5: Clean up extras ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
def remove_unwanted_objects():
    log("Removing unwanted objects…")
    for obj in list(bpy.data.objects):
        if obj.type == "MESH" and obj.name.startswith("Cube"):
            bpy.data.objects.remove(obj, do_unlink=True)


#––– Step 6: Pose bone selection ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
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


#––– Step 7: Export + zip –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# def export_fbx_with_animation(out_path):
#     log(f"Exporting FBX to: {out_path}")
#     try:
#         bpy.ops.export_scene.fbx(
#             filepath=out_path,
#             use_selection=False,
#             apply_unit_scale=True,
#             bake_space_transform=True,
#             object_types={"ARMATURE", "MESH"},
#             bake_anim=True,
#             bake_anim_use_all_bones=True,
#             bake_anim_force_startend_keying=True,
#             bake_anim_simplify_factor=1.0,
#             add_leaf_bones=False,
#             primary_bone_axis="Y",
#             secondary_bone_axis="X",
#             axis_up="Y",
#             axis_forward="Z",
#             path_mode='COPY',
#             embed_textures=True
#         )
#         if os.path.exists(OUTPUT_ZIP_PATH):
#             os.remove(OUTPUT_ZIP_PATH)
#         with zipfile.ZipFile(OUTPUT_ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
#             zf.write(out_path, os.path.basename(out_path))
#         log(f"Created zip at: {OUTPUT_ZIP_PATH}")
#     except Exception as e:
#         log(f"Export failed: {e}")

def export_fbx_with_animation(out_path):
    log(f"Exporting FBX Animation Only to: {out_path}")
    try:
        bpy.ops.export_scene.fbx(
            filepath=out_path,
            use_selection=False,
            apply_unit_scale=True,
            bake_space_transform=True,
            object_types={"ARMATURE"},  # Only export armatures (no mesh)
            bake_anim=True,
            bake_anim_use_all_bones=True,
            bake_anim_force_startend_keying=True,
            bake_anim_simplify_factor=1.0,
            add_leaf_bones=False,
            primary_bone_axis="Y",
            secondary_bone_axis="X",
            axis_up="Y",
            axis_forward="Z",
            path_mode='COPY',
            embed_textures=False  # Irrelevant since no mesh/textures included
        )
        if os.path.exists(OUTPUT_ZIP_PATH):
            os.remove(OUTPUT_ZIP_PATH)
        with zipfile.ZipFile(OUTPUT_ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(out_path, os.path.basename(out_path))
        log(f"Created zip at: {OUTPUT_ZIP_PATH}")
    except Exception as e:
        log(f"Export failed: {e}")



#––– Step 8: Ground the armature by its foot bone –––––––––––––––––––––––––––––––––––––––––––––
def ground_avatar_by_foot_bone(armature_name, foot_bone_name="LeftFoot", ground_z=0.0):
    """
    Finds the specified armature, switches to POSE mode at frame 1,
    reads the world‐space position of the foot bone head, then moves
    the entire armature so that that bone’s Z == ground_z.
    """
    armature = bpy.data.objects.get(armature_name)
    if not armature or armature.type != "ARMATURE":
        log(f"Armature '{armature_name}' not found or not an ARMATURE.")
        return

    # Make sure we're in OBJECT mode before changing frames
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode='POSE')
    bpy.context.scene.frame_set(1)

    pose_bone = armature.pose.bones.get(foot_bone_name)
    if not pose_bone:
        log(f"Bone '{foot_bone_name}' not found in armature '{armature_name}'.")
        return

    # Compute foot bone head position in world coordinates
    foot_head_world = armature.matrix_world @ pose_bone.head
    offset = ground_z - foot_head_world.z

    # Exit POSE mode, apply offset
    bpy.ops.object.mode_set(mode='OBJECT')
    armature.location.z += offset
    log(f"Grounded '{armature_name}' using bone '{foot_bone_name}' (offset {offset:.4f}).")


#––– Step 9: (Optional) wrap text for video titles –––––––––––––––––––––––––––––––––––––––––––––
def wrap_text_to_fit_plane(original_text, max_width_chars=70, max_lines=4):
    wrapped = textwrap.wrap(original_text, width=max_width_chars)
    if len(wrapped) > max_lines:
        wrapped = wrapped[:max_lines]
        wrapped[-1] += "..."
    return "\n".join(wrapped)


def export_video_with_title(title, output_mp4_path,
                            resolution_x=1920, resolution_y=1080,
                            fps=30, test_mode=False):
    scene = bpy.context.scene

    engines = bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items.keys()
    if "BLENDER_EEVEE_NEXT" in engines:
        scene.render.engine = 'BLENDER_EEVEE_NEXT'
    elif "BLENDER_EEVEE" in engines:
        scene.render.engine = 'BLENDER_EEVEE'
    else:
        scene.render.engine = 'CYCLES'

    scene.render.use_sequencer = False
    scene.render.use_compositing = False

    if scene.render.engine.startswith("BLENDER_EEVEE"):
        scene.render.use_motion_blur = False
        if hasattr(scene.eevee, "taa_render_samples"):
            scene.eevee.taa_render_samples = 8
        if hasattr(scene.eevee, "use_soft_shadows"):
            scene.eevee.use_soft_shadows = False

    if test_mode:
        resolution_x = 960
        resolution_y = 540

    scene.render.resolution_x = resolution_x
    scene.render.resolution_y = resolution_y
    scene.render.resolution_percentage = 100
    scene.render.fps = fps
    scene.frame_start = 1
    scene.frame_end = 268
    scene.render.filepath = output_mp4_path

    if scene.camera is None:
        cam_data = bpy.data.cameras.new("Camera")
        cam_obj = bpy.data.objects.new("Camera", cam_data)
        scene.collection.objects.link(cam_obj)
        scene.camera = cam_obj
    else:
        cam_obj = scene.camera

    cam_obj.location = (0, -6, 3.5)
    cam_obj.rotation_euler = (1.15, 0, 0)

    bpy.ops.mesh.primitive_plane_add(size=5, location=(0, 0, -0.05))
    ground = bpy.context.active_object
    ground.name = "GroundPlane"
    mat = bpy.data.materials.new(name="GroundMat")
    mat.diffuse_color = (0.5, 0.5, 0.5, 1)
    ground.data.materials.append(mat)

    avatar_obj = next((obj for obj in bpy.data.objects if obj.type == 'ARMATURE'), None)
    if not avatar_obj:
        raise RuntimeError("No avatar (ARMATURE) found in the scene.")

    avatar_obj.location.y += 1.5

    wrapped_title = wrap_text_to_fit_plane(title, max_width_chars=70, max_lines=4)

    pos = avatar_obj.location
    bpy.ops.object.text_add(location=(pos.x, pos.y - 1.5, pos.z + 1.9))
    text_obj = bpy.context.active_object
    text_obj.name = "FootTitleText"
    text_obj.data.body = wrapped_title
    text_obj.data.size = 0.1
    text_obj.data.align_x = 'CENTER'
    text_obj.data.align_y = 'CENTER'
    text_obj.rotation_euler = (1.5708, 0, 0)

    text_mat = bpy.data.materials.new(name="TextMat")
    text_mat.use_nodes = True
    nodes = text_mat.node_tree.nodes
    links = text_mat.node_tree.links
    nodes.clear()

    emission = nodes.new("ShaderNodeEmission")
    emission.inputs["Color"].default_value = (1.0, 0.8, 0.0, 1.0)
    emission.inputs["Strength"].default_value = 5.0
    output = nodes.new("ShaderNodeOutputMaterial")
    links.new(emission.outputs["Emission"], output.inputs["Surface"])

    text_obj.data.materials.append(text_mat)
    text_obj.hide_render = False
    text_obj.hide_viewport = False

    scene.render.image_settings.file_format = 'FFMPEG'
    scene.render.ffmpeg.format = 'MPEG4'
    scene.render.ffmpeg.codec = 'H264'
    scene.render.ffmpeg.constant_rate_factor = 'HIGH'
    scene.render.ffmpeg.ffmpeg_preset = 'GOOD'

    print(f"Rendering to: {output_mp4_path}")
    bpy.ops.render.render(animation=True)
    print("Render complete.")


#––– Main workflow –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
def main():
    log("Script started")

    # -- Step A: Import the FBX
    SCALE = 1.0   # Tweak as needed so the retargeted mesh matches your original size
    GROUND = 0.0  # We will ground later via the foot bone
    imported_objs = load_and_debug_fbx(INPUT_FBX_FILE_PATH,
                                       scale_factor=SCALE,
                                       ground_to_z=GROUND)
    if not imported_objs:
        log("Aborting: no FBX loaded.")
        return

    # -- Step B: Enable KeeMap and load BVH
    enable_keemap_addon()
    load_bvh_file(INPUT_BVH_FILE_PATH, scale=0.02)

    # -- Step C: Configure and run retargeting
    readBone_setup_scene_settings()
    keemap_read_file()
    remove_unwanted_objects()
    select_rigs_and_bones()
    perform_animation_transfer()

    # -- Step D: Ground the final retargeted armature
    # Replace "Armature" with your actual armature name in the scene
    ground_avatar_by_foot_bone(armature_name="Armature", foot_bone_name="LeftFoot", ground_z=0.0)

    # -- Step E: Export to FBX (and create ZIP)
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
        parser.add_argument("--high_res", type=str, default="false", help="Use high resolution output (true/false)")
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
            test_mode=(args.high_res.lower() != "true")
        )