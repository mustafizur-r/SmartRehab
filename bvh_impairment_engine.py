"""
bvh_impairment_engine.py  v3 — SmartRehab
==========================================
Biomechanically accurate gait impairment engine.

KEY INSIGHT from v2 failure:
  The corrections were too subtle — severity=0.6 produced barely visible change.
  v3 applies much stronger angular corrections grounded in clinical biomechanics:

  Clinical reference ranges (healthy walking):
    Hip flexion peak:    ~30° (swing)
    Hip extension peak: ~10° (late stance)
    Knee flexion peak:  ~60° (swing)  ← most visible joint
    Ankle dorsiflexion:  ~15° (swing)
    Ankle plantarflexion: ~20° (push-off)

  For visible impairment, corrections must be:
    Foot drop:     push ankle toward 0° dorsiflexion (lose -15° → 0°)
    Knee stiff:    reduce peak knee flexion from 60° to ~10-15°
    Stride asym:   reduce hip flexion swing amplitude by 30-60%
    Trunk lean:    add 10-20° lateral tilt to spine Z
    Arm swing:     reduce humeral swing from ±25° to near 0°
    Cadence:       slow frame rate (stretch time)

AXIS CONVENTIONS (verified on real SnapMoGen BVH bvh_0_out.bvh):
  All rotations in degrees.
  legUpper X: positive = flexion (swing forward), negative = extension
  legLower X: negative = knee flexion (more negative = more bent)
  foot X:     negative = dorsiflexion, positive = plantarflexion
  spine Z:    positive = lean right, negative = lean left
  armUpper X: oscillates ± during walk (arm swing)
  ROOT Y:     forward translation (meters, scaled)

NEW FEATURES in v3:
  - Pelvis drop (Trendelenburg):  pelvis dips toward swing side
  - Hip hike (circumduction):     pelvis hikes up to clear foot during swing
  - Vaulting:                     rise on tiptoe of stance leg to clear swing foot
  - Antalgic gait:                shortened stance time on painful side
  - Scissor gait:                 legs cross midline (adductor spasticity)
  - Crouch gait:                  increased knee and hip flexion throughout
  - Parkinsonian shuffle:         reduced step length + flexed posture + reduced arm swing
"""

import os
import math

# ─────────────────────────────────────────────────────────────────────────────
# Joint channel map  (verified against real SnapMoGen BVH, 75 channels)
# ─────────────────────────────────────────────────────────────────────────────
JOINT_CHANNEL_MAP = {
    "ROOT":                      {"start": 0,  "channels": 6},   # Tx,Ty,Tz,Rz,Rx,Ry
    "C_spine0001_bind_JNT":      {"start": 6,  "channels": 3},
    "C_spine0002_bind_JNT":      {"start": 9,  "channels": 3},
    "C_spine0003_bind_JNT":      {"start": 12, "channels": 3},
    "C_neck0001_bind_JNT":       {"start": 15, "channels": 3},
    "C_neck0002_bind_JNT":       {"start": 18, "channels": 3},
    "C_head_bind_JNT":           {"start": 21, "channels": 3},
    "L_clavicle_bind_JNT":       {"start": 24, "channels": 3},
    "L_armUpper0001_bind_JNT":   {"start": 27, "channels": 3},
    "L_armLower0001_bind_JNT":   {"start": 30, "channels": 3},
    "L_hand0001_bind_JNT":       {"start": 33, "channels": 3},
    "R_clavicle_bind_JNT":       {"start": 36, "channels": 3},
    "R_armUpper0001_bind_JNT":   {"start": 39, "channels": 3},
    "R_armLower0001_bind_JNT":   {"start": 42, "channels": 3},
    "R_hand0001_bind_JNT":       {"start": 45, "channels": 3},
    "C_pelvis0001_bind_JNT":     {"start": 48, "channels": 3},
    "L_legUpper0001_bind_JNT":   {"start": 51, "channels": 3},
    "L_legLower0001_bind_JNT":   {"start": 54, "channels": 3},
    "L_foot0001_bind_JNT":       {"start": 57, "channels": 3},
    "L_foot0002_bind_JNT":       {"start": 60, "channels": 3},
    "R_legUpper0001_bind_JNT":   {"start": 63, "channels": 3},
    "R_legLower0001_bind_JNT":   {"start": 66, "channels": 3},
    "R_foot0001_bind_JNT":       {"start": 69, "channels": 3},
    "R_foot0002_bind_JNT":       {"start": 72, "channels": 3},
}

P = {k: v["start"] for k, v in JOINT_CHANNEL_MAP.items()}

# ─────────────────────────────────────────────────────────────────────────────
# BVH I/O
# ─────────────────────────────────────────────────────────────────────────────

def parse_bvh(filepath):
    with open(filepath, "r") as f:
        content = f.read()
    idx        = content.find("MOTION")
    header     = content[:idx + len("MOTION") + 1]
    body       = content[idx + len("MOTION"):].strip().splitlines()
    frame_time = float(body[1].split(":")[1].strip())
    frames     = []
    for line in body[2:]:
        line = line.strip()
        if line:
            frames.append([float(v) for v in line.split()])
    return header, frame_time, frames


def write_bvh(filepath, header, frame_time, frames):
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "w") as f:
        f.write(header.rstrip() + "\n")
        f.write(f"Frames: {len(frames)}\n")
        f.write(f"Frame Time: {frame_time:.6f}\n")
        for frame in frames:
            f.write(" ".join(f"{v:.6f}" for v in frame) + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Low-level getters / setters
# ─────────────────────────────────────────────────────────────────────────────

def _get(frame, joint, offset=0):
    s = P[joint]
    o = 3 if joint == "ROOT" else 0   # ROOT: first 3 channels are translation
    return frame[s + o + offset]


def _set(frame, joint, offset, value):
    s = P[joint]
    o = 3 if joint == "ROOT" else 0
    frame[s + o + offset] = value


def get_rot(frame, joint):
    """Return (x, y, z) rotation of joint in degrees."""
    return _get(frame, joint, 0), _get(frame, joint, 1), _get(frame, joint, 2)


def set_rot(frame, joint, x, y, z):
    _set(frame, joint, 0, x)
    _set(frame, joint, 1, y)
    _set(frame, joint, 2, z)


def get_root_pos(frame):
    return frame[0], frame[1], frame[2]   # Tx, Ty, Tz


def set_root_pos(frame, tx, ty, tz):
    frame[0] = tx; frame[1] = ty; frame[2] = tz


# ─────────────────────────────────────────────────────────────────────────────
# Gait phase detection
# ─────────────────────────────────────────────────────────────────────────────

def _side_prefix(side):
    return "R" if side == "right" else "L"


def _hip_x(frame, side):
    return get_rot(frame, f"{_side_prefix(side)}_legUpper0001_bind_JNT")[0]


def _knee_x(frame, side):
    return get_rot(frame, f"{_side_prefix(side)}_legLower0001_bind_JNT")[0]


def _foot_x(frame, side):
    return get_rot(frame, f"{_side_prefix(side)}_foot0001_bind_JNT")[0]


def _is_swing(frame, side):
    """Hip flexion (positive X) indicates swing phase."""
    return _hip_x(frame, side) > 2.0


def _is_stance(frame, side):
    return not _is_swing(frame, side)


def _swing_intensity(frame, side):
    """0→1 ramp indicating how deep into swing phase (0=just started, 1=peak)."""
    hx = max(0.0, _hip_x(frame, side))
    # Typical peak hip flexion ~30°
    return min(1.0, hx / 25.0)


def _compute_stats(frames, joint, axis=0):
    """Return (mean, std, min, max) of a joint channel across all frames."""
    vals = [get_rot(f, joint)[axis] for f in frames]
    mean = sum(vals) / len(vals)
    variance = sum((v - mean) ** 2 for v in vals) / len(vals)
    return mean, math.sqrt(variance), min(vals), max(vals)


# ─────────────────────────────────────────────────────────────────────────────
# ── IMPAIRMENT 1: FOOT DROP / ANKLE DROP ──────────────────────────────────────
#
# Biomechanics:
#   Healthy: tibialis anterior pulls foot up (dorsiflexion, foot X → negative)
#   during swing to clear ground. Peak dorsiflexion ~15°.
#   Foot drop: tibialis anterior weak → cannot dorsiflex → foot hangs (X → 0°)
#   or even plantar-flexed (X → positive).
#   Compensation: hip hiking, circumduction, or steppage gait.
# ─────────────────────────────────────────────────────────────────────────────

def apply_ankle_drop(frames, side, severity):
    """
    Reduce dorsiflexion during swing. severity=1.0 → foot neutral/plantarflexed.
    Also adds toe drag artifact at initial swing (foot drags briefly).
    """
    px   = _side_prefix(side)
    foot = f"{px}_foot0001_bind_JNT"
    toe  = f"{px}_foot0002_bind_JNT"

    # Measure peak dorsiflexion during swing (most negative foot X)
    swing_xs = [get_rot(f, foot)[0] for f in frames if _is_swing(f, side)]
    if not swing_xs:
        return
    peak_dorsi = min(swing_xs)   # most negative = max dorsiflexion (e.g. -20°)
    mean_dorsi  = sum(swing_xs) / len(swing_xs)

    # Target: push toward 0° (neutral) or slightly positive (plantarflexed)
    # at severity=1.0, foot X → max(0, current) i.e. no dorsiflexion at all
    for i, frame in enumerate(frames):
        if _is_swing(frame, side):
            intensity = _swing_intensity(frame, side)
            xr, yr, zr = get_rot(frame, foot)

            # How much dorsiflexion to remove
            # At peak swing intensity: remove up to (peak_dorsi * severity) degrees
            correction = -peak_dorsi * severity * intensity * 0.90
            new_x = xr + correction
            set_rot(frame, foot, new_x, yr, zr)

            # Toe: same correction, reduced
            xr2, yr2, zr2 = get_rot(frame, toe)
            set_rot(frame, toe, xr2 + correction * 0.5, yr2, zr2)

        else:
            # Stance phase: slight plantarflexion loss (weak push-off)
            xr, yr, zr = get_rot(frame, foot)
            # Reduce push-off amplitude by severity
            if xr > 5.0:   # plantarflexion during push-off
                reduction = (xr - 5.0) * severity * 0.5
                set_rot(frame, foot, xr - reduction, yr, zr)


# ─────────────────────────────────────────────────────────────────────────────
# ── IMPAIRMENT 2: STIFF KNEE GAIT ────────────────────────────────────────────
#
# Biomechanics:
#   Healthy: knee flexes to ~60° during swing for foot clearance.
#   Stiff knee: rectus femoris spasticity → peak knee flexion reduced to 30-10°.
#   Key signature: leg swings forward with straight knee → circumduction or hip hike.
#   This is one of the MOST VISIBLE impairments.
# ─────────────────────────────────────────────────────────────────────────────

def apply_knee_stiffness(frames, side, severity):
    """
    Reduce knee flexion during swing. severity=1.0 → nearly rigid knee (~5° flex).
    Also adds compensatory hip circumduction (slight outward arc).
    """
    px   = _side_prefix(side)
    knee = f"{px}_legLower0001_bind_JNT"
    hip  = f"{px}_legUpper0001_bind_JNT"

    mean_k, std_k, min_k, max_k = _compute_stats(frames, knee, axis=0)
    # During swing, knee X goes very negative (flexion). min_k = peak flex.
    # During stance, knee is near 0 or slightly negative.

    swing_knee = [get_rot(f, knee)[0] for f in frames if _is_swing(f, side)]
    if not swing_knee:
        return
    peak_flex = min(swing_knee)   # most negative = peak flexion e.g. -55°
    healthy_range = abs(peak_flex - mean_k)

    for frame in frames:
        xk, yk, zk = get_rot(frame, knee)
        if _is_swing(frame, side):
            intensity = _swing_intensity(frame, side)

            # Push knee toward mean (less flexion) proportional to severity
            # At severity=1: knee stays near mean (nearly straight)
            target     = mean_k + (xk - mean_k) * (1.0 - severity * 0.85)
            set_rot(frame, knee, target, yk, zk)

            # Compensatory: add slight hip abduction (Z rotation) for circumduction
            xh, yh, zh = get_rot(frame, hip)
            circum = severity * 8.0 * intensity   # up to 8° abduction
            set_rot(frame, hip, xh, yh, zh + circum)
        else:
            # Stance: mild extension lag (knee slightly flexed throughout)
            crouch_add = severity * 5.0
            set_rot(frame, knee, xk - crouch_add, yk, zk)


# ─────────────────────────────────────────────────────────────────────────────
# ── IMPAIRMENT 3: STRIDE ASYMMETRY ───────────────────────────────────────────
#
# Biomechanics:
#   Healthy: left and right steps equal length, equal timing.
#   Antalgic/weak side: shorter step (reduced hip flexion in swing),
#   faster swing-through (spend less time on painful leg in stance).
#   Visible as: one leg clearly steps shorter than the other.
# ─────────────────────────────────────────────────────────────────────────────

def apply_stride_asymmetry(frames, side, severity):
    """
    Shorten step on affected side. severity=1.0 → ~50% shorter stride.
    Reduces hip flexion peak and knee flexion amplitude during swing.
    """
    px    = _side_prefix(side)
    hip   = f"{px}_legUpper0001_bind_JNT"
    knee  = f"{px}_legLower0001_bind_JNT"

    hip_swing = [get_rot(f, hip)[0] for f in frames if _is_swing(f, side)]
    if not hip_swing:
        return
    peak_flex = max(hip_swing)   # max hip flexion
    mean_hip  = sum(get_rot(f, hip)[0] for f in frames) / len(frames)

    for frame in frames:
        if _is_swing(frame, side):
            xh, yh, zh = get_rot(frame, hip)
            # Compress swing amplitude — push hip flex toward mean
            reduction = (xh - mean_hip) * severity * 0.50
            set_rot(frame, hip, xh - reduction, yh, zh)

            # Also reduce knee flexion proportionally
            xk, yk, zk = get_rot(frame, knee)
            mean_k     = sum(get_rot(f, knee)[0] for f in frames) / len(frames)
            set_rot(frame, knee, mean_k + (xk - mean_k) * (1.0 - severity * 0.40), yk, zk)


# ─────────────────────────────────────────────────────────────────────────────
# ── IMPAIRMENT 4: TRUNK LATERAL LEAN (TRENDELENBURG) ─────────────────────────
#
# Biomechanics:
#   Healthy: gluteus medius keeps pelvis level during single-leg stance.
#   Weak glute med (stroke, hip OA): pelvis drops to opposite side (stance phase).
#   Compensation: trunk leans toward weak side to shift CoM.
#   Visible: body tilts sharply left/right with each step.
#
#   Two subtypes:
#   a) Static lean: constant lean throughout (e.g. stroke, pain avoidance)
#   b) Dynamic Trendelenburg: lean occurs only during single-leg stance
# ─────────────────────────────────────────────────────────────────────────────

def apply_trunk_lean(frames, direction, severity, dynamic=True):
    """
    Add lateral trunk lean.
    direction: 'right' or 'left'
    dynamic=True: lean occurs during stance on same side (Trendelenburg pattern)
    dynamic=False: constant lean throughout (antalgic, stroke)
    """
    sign   = 1.0 if direction == "right" else -1.0
    spine_joints = [
        "C_spine0001_bind_JNT",
        "C_spine0002_bind_JNT",
        "C_spine0003_bind_JNT",
    ]
    # Distribute lean across 3 spine segments
    # Total lean at severity=1.0: ~20° (clinically significant, clearly visible)
    total_lean = 20.0 * severity
    per_joint  = total_lean / len(spine_joints)

    # Pelvis tilt contribution
    pelvis_lean = 8.0 * severity * sign

    for frame in frames:
        # Dynamic: lean increases during single-leg stance on that side
        if dynamic:
            # On stance of the leaning side, glute med fires → lean toward it
            stance_side   = direction
            is_on_stance  = _is_stance(frame, stance_side)
            lean_factor   = 1.0 if is_on_stance else 0.3
        else:
            lean_factor = 1.0

        for j in spine_joints:
            xr, yr, zr = get_rot(frame, j)
            set_rot(frame, j, xr, yr, zr + sign * per_joint * lean_factor)

        # Pelvis drop on opposite side (classic Trendelenburg)
        px, py, pz = get_rot(frame, "C_pelvis0001_bind_JNT")
        set_rot(frame, "C_pelvis0001_bind_JNT",
                px, py, pz + pelvis_lean * lean_factor)


# ─────────────────────────────────────────────────────────────────────────────
# ── IMPAIRMENT 5: REDUCED ARM SWING ──────────────────────────────────────────
#
# Biomechanics:
#   Healthy: arms swing opposite to legs, ~±20-25° humerus flexion/extension.
#   Reduced swing: common in hemiplegia (spastic arm held flexed),
#   Parkinson's (rigidity), pain.
#   Hemiplegia: affected arm held at elbow flexion ~90°, shoulder internally rotated.
# ─────────────────────────────────────────────────────────────────────────────

def apply_arm_swing_reduction(frames, side, severity):
    """
    Reduce arm swing on affected side. severity=1.0 → arm nearly static.
    At high severity, adds elbow flexion (hemiplegic posture).
    """
    px    = _side_prefix(side)
    upper = f"{px}_armUpper0001_bind_JNT"
    lower = f"{px}_armLower0001_bind_JNT"

    mean_u, _, min_u, max_u = _compute_stats(frames, upper, axis=0)
    mean_l, _, min_l, max_l = _compute_stats(frames, lower, axis=0)

    for frame in frames:
        # Upper arm: compress swing around mean
        xu, yu, zu = get_rot(frame, upper)
        new_xu     = mean_u + (xu - mean_u) * (1.0 - severity * 0.90)
        # At high severity: add internal rotation (hemiplegic posture)
        int_rot    = severity * 15.0 if severity > 0.5 else 0.0
        set_rot(frame, upper, new_xu, yu + int_rot, zu)

        # Lower arm: add elbow flexion at high severity (held flexed)
        xl, yl, zl = get_rot(frame, lower)
        new_xl     = mean_l + (xl - mean_l) * (1.0 - severity * 0.85)
        # Elbow flexion bias: move toward more flexion
        elbow_flex = severity * 20.0
        set_rot(frame, lower, new_xl - elbow_flex * 0.5, yl, zl)


# ─────────────────────────────────────────────────────────────────────────────
# ── IMPAIRMENT 6: CADENCE REDUCTION (SLOW WALKING) ───────────────────────────
#
# Biomechanics:
#   Reduced cadence → stretched frame time → avatar moves slower.
#   Also reduces ROOT translation speed to match slower cadence.
# ─────────────────────────────────────────────────────────────────────────────

def apply_cadence_reduction(frames, frame_time, severity):
    """
    Slow down the animation by stretching frame time and interpolating frames.
    severity=1.0 → 50% slower.
    """
    if severity <= 0.0:
        return frames, frame_time

    stretch    = 1.0 + (0.60 * severity)   # up to 60% slower
    new_ft     = frame_time * stretch
    n          = len(frames)
    new_n      = int(n * stretch)
    new_frames = []

    for i in range(new_n):
        pos = i / stretch
        lo  = int(pos)
        hi  = min(lo + 1, n - 1)
        t   = pos - lo
        interp = [frames[lo][j] * (1 - t) + frames[hi][j] * t
                  for j in range(len(frames[0]))]
        new_frames.append(interp)

    return new_frames, new_ft


# ─────────────────────────────────────────────────────────────────────────────
# ── IMPAIRMENT 7: HIP HIKE / CIRCUMDUCTION ───────────────────────────────────
#
# Biomechanics:
#   Compensation for foot drop or stiff knee: patient hikes pelvis up on
#   affected side during swing to gain ground clearance.
#   Or swings leg outward in a wide arc (circumduction).
#   Highly visible — pelvis tilts up sharply on swing side.
# ─────────────────────────────────────────────────────────────────────────────

def apply_hip_hike(frames, side, severity):
    """
    Add pelvic hike on affected side during swing phase.
    severity=1.0 → ~12° pelvic tilt + 10° hip abduction arc.
    """
    sign = 1.0 if side == "right" else -1.0
    px   = _side_prefix(side)
    hip  = f"{px}_legUpper0001_bind_JNT"

    for frame in frames:
        if _is_swing(frame, side):
            intensity = _swing_intensity(frame, side)

            # Pelvis hike: tilt up on swing side (Z rotation of pelvis)
            px_rot, py_rot, pz_rot = get_rot(frame, "C_pelvis0001_bind_JNT")
            hike = sign * severity * 12.0 * intensity
            set_rot(frame, "C_pelvis0001_bind_JNT", px_rot, py_rot, pz_rot + hike)

            # Circumduction: hip abducts outward during swing
            xh, yh, zh = get_rot(frame, hip)
            circ = sign * severity * 10.0 * intensity
            set_rot(frame, hip, xh, yh, zh + circ)


# ─────────────────────────────────────────────────────────────────────────────
# ── IMPAIRMENT 8: CROUCH GAIT ────────────────────────────────────────────────
#
# Biomechanics:
#   Persistent knee and hip flexion throughout stance.
#   Common in: cerebral palsy, spastic diplegia, hamstring contracture.
#   Visible as: crouched posture, knees never fully extend, forward trunk lean.
# ─────────────────────────────────────────────────────────────────────────────

def apply_crouch_gait(frames, severity):
    """
    Add persistent knee and hip flexion throughout gait cycle (both sides).
    Also adds forward trunk lean.
    severity=1.0 → ~25° sustained knee flexion, 15° hip flexion, 10° trunk lean.
    """
    knee_flex = 25.0 * severity   # degrees of constant knee flexion
    hip_flex  = 15.0 * severity   # degrees of constant hip flexion
    trunk_fwd = 10.0 * severity   # degrees of forward trunk lean

    for frame in frames:
        for side in ("right", "left"):
            px = _side_prefix(side)

            # Knee: add persistent flexion (more negative X)
            knee = f"{px}_legLower0001_bind_JNT"
            xk, yk, zk = get_rot(frame, knee)
            set_rot(frame, knee, xk - knee_flex, yk, zk)

            # Hip: add persistent flexion (more positive X in stance)
            hip = f"{px}_legUpper0001_bind_JNT"
            xh, yh, zh = get_rot(frame, hip)
            set_rot(frame, hip, xh + hip_flex * 0.5, yh, zh)

        # Forward trunk lean (spine X — forward tilt)
        for spine in ["C_spine0001_bind_JNT", "C_spine0002_bind_JNT", "C_spine0003_bind_JNT"]:
            xs, ys, zs = get_rot(frame, spine)
            set_rot(frame, spine, xs + trunk_fwd / 3.0, ys, zs)


# ─────────────────────────────────────────────────────────────────────────────
# ── IMPAIRMENT 9: PARKINSONIAN SHUFFLE ───────────────────────────────────────
#
# Biomechanics:
#   Short shuffling steps, reduced arm swing, flexed posture, festination.
#   Festination: steps get faster and smaller as if chasing center of gravity.
#   Visible: very small steps, arms barely swing, forward-flexed trunk.
# ─────────────────────────────────────────────────────────────────────────────

def apply_parkinsonian_shuffle(frames, frame_time, severity):
    """
    Full Parkinsonian gait pattern:
    - Reduced stride length (both sides)
    - Near-zero arm swing (both sides)
    - Forward trunk flexion
    - Reduced cadence
    Returns (new_frames, new_frame_time).
    """
    # Step 1: slow cadence slightly
    frames, frame_time = apply_cadence_reduction(frames, frame_time, severity * 0.4)

    # Step 2: reduce stride amplitude (both sides)
    for side in ("right", "left"):
        apply_stride_asymmetry(frames, side, severity * 0.7)

    # Step 3: reduce arm swing (both sides)
    for side in ("right", "left"):
        apply_arm_swing_reduction(frames, side, severity * 0.90)

    # Step 4: forward flexed posture
    trunk_fwd = 15.0 * severity
    for frame in frames:
        for j in ["C_spine0001_bind_JNT", "C_spine0002_bind_JNT",
                  "C_spine0003_bind_JNT"]:
            xs, ys, zs = get_rot(frame, j)
            set_rot(frame, j, xs + trunk_fwd / 3.0, ys, zs)

    return frames, frame_time


# ─────────────────────────────────────────────────────────────────────────────
# ── IMPAIRMENT 10: SCISSOR GAIT ──────────────────────────────────────────────
#
# Biomechanics:
#   Adductor spasticity (CP, spastic diplegia): legs cross midline in swing.
#   Thighs adduct and internally rotate → legs cross like scissors.
#   Visible: legs clearly cross each other during swing.
# ─────────────────────────────────────────────────────────────────────────────

def apply_scissor_gait(frames, severity):
    """
    Add hip adduction and internal rotation during swing (both sides).
    severity=1.0 → legs visibly cross midline.
    """
    for frame in frames:
        for side in ("right", "left"):
            if _is_swing(frame, side):
                intensity = _swing_intensity(frame, side)
                sign      = -1.0 if side == "right" else 1.0   # cross toward midline
                px        = _side_prefix(side)
                hip       = f"{px}_legUpper0001_bind_JNT"

                xh, yh, zh = get_rot(frame, hip)
                adduction  = sign * severity * 12.0 * intensity   # adduct toward midline
                int_rot    = sign * severity * 8.0 * intensity     # internal rotation
                set_rot(frame, hip, xh, yh + int_rot, zh + adduction)


# ─────────────────────────────────────────────────────────────────────────────
# ── IMPAIRMENT 11: ANTALGIC GAIT (PAIN AVOIDANCE) ────────────────────────────
#
# Biomechanics:
#   Patient hurries through stance on painful leg (shorter, faster stance).
#   Trunk lurches toward painful side to reduce joint loading.
#   Healthy cadence but very asymmetric timing.
# ─────────────────────────────────────────────────────────────────────────────

def apply_antalgic_gait(frames, side, severity):
    """
    Simulate pain avoidance: shortened stance on affected side + trunk lurch.
    Combined: stride asymmetry + trunk lean toward affected side.
    """
    # Shorter step on affected side
    apply_stride_asymmetry(frames, side, severity * 0.7)

    # Trunk lurch toward affected side (offloading painful joint)
    apply_trunk_lean(frames, side, severity * 0.5, dynamic=True)

    # Slight knee flexion on affected side (pain protective)
    px   = _side_prefix(side)
    knee = f"{px}_legLower0001_bind_JNT"
    for frame in frames:
        if _is_stance(frame, side):
            xk, yk, zk = get_rot(frame, knee)
            set_rot(frame, knee, xk - severity * 8.0, yk, zk)


# ─────────────────────────────────────────────────────────────────────────────
# ── IMPAIRMENT 12: HEMIPLEGIC GAIT (FULL PACKAGE) ────────────────────────────
#
# Biomechanics:
#   Post-stroke hemiplegia: all ipsilateral limb systems affected.
#   Leg: spastic equinovarus foot, stiff knee, reduced hip flexion, circumduction.
#   Arm: held in flexed/pronated posture, no swing.
#   Trunk: leans toward affected side.
#   Pelvis: hikes to clear spastic foot.
# ─────────────────────────────────────────────────────────────────────────────

def apply_hemiplegic_gait(frames, frame_time, side, severity):
    """
    Complete hemiplegic gait pattern — single call applies all components.
    """
    # Foot drop (equinus/varus foot)
    apply_ankle_drop(frames, side, severity * 0.85)

    # Stiff knee (rectus femoris spasticity)
    apply_knee_stiffness(frames, side, severity * 0.75)

    # Reduced hip flexion (shorter step)
    apply_stride_asymmetry(frames, side, severity * 0.70)

    # Hip hike to clear the foot
    apply_hip_hike(frames, side, severity * 0.65)

    # Trunk lean toward affected side
    apply_trunk_lean(frames, side, severity * 0.50, dynamic=True)

    # Arm held in spastic flexion posture
    apply_arm_swing_reduction(frames, side, severity * 0.90)

    # Slight cadence reduction
    frames, frame_time = apply_cadence_reduction(frames, frame_time, severity * 0.25)

    return frames, frame_time


# ─────────────────────────────────────────────────────────────────────────────
# Main apply_impairment entry point
# ─────────────────────────────────────────────────────────────────────────────

def apply_impairment(input_bvh: str, state: dict, output_bvh: str) -> str:
    """
    Apply clinical gait impairments to a BVH file.

    Parameters
    ----------
    input_bvh  : path to clean base BVH (from SnapMoGen)
    state      : dict of impairment parameters (see below)
    output_bvh : path to write impaired BVH

    State keys and value ranges (all 0.0 → 1.0):
    ─────────────────────────────────────────────
    Basic parameters:
      ankle_drop_right/left        : foot drop, dragging foot
      knee_stiffness_right/left    : stiff-legged gait, rectus femoris spasticity
      stride_asymmetry_right/left  : shorter step on affected side
      trunk_lean_right/left        : lateral trunk lean (Trendelenburg)
      arm_swing_reduction_right/left : spastic or rigid arm
      cadence_reduction            : overall slow walking

    Compound patterns (one key activates a full clinical syndrome):
      hemiplegic_right/left        : full hemiplegic gait (post-stroke)
      parkinsonian_shuffle         : Parkinson's gait pattern
      crouch_gait                  : persistent knee/hip flexion (CP)
      scissor_gait                 : adductor spasticity (CP)
      antalgic_right/left          : pain-avoidance gait (hip/knee OA)
      hip_hike_right/left          : compensatory pelvic hike
    """
    header, frame_time, frames = parse_bvh(input_bvh)
    frames = [list(f) for f in frames]
    n_orig = len(frames)

    print(f"[BVH Engine v3] Input: {n_orig} frames, {frame_time*1000:.1f}ms/frame")
    print(f"[BVH Engine v3] Parameters: {state}")

    # ── Compound patterns (applied first, they call primitives internally) ────

    for side in ("right", "left"):
        v = float(state.get(f"hemiplegic_{side}", 0.0))
        if v > 0:
            print(f"[BVH Engine v3] Applying hemiplegic gait ({side}) severity={v:.2f}")
            frames, frame_time = apply_hemiplegic_gait(frames, frame_time, side, v)

    v = float(state.get("parkinsonian_shuffle", 0.0))
    if v > 0:
        print(f"[BVH Engine v3] Applying parkinsonian shuffle severity={v:.2f}")
        frames, frame_time = apply_parkinsonian_shuffle(frames, frame_time, v)

    v = float(state.get("crouch_gait", 0.0))
    if v > 0:
        print(f"[BVH Engine v3] Applying crouch gait severity={v:.2f}")
        apply_crouch_gait(frames, v)

    v = float(state.get("scissor_gait", 0.0))
    if v > 0:
        print(f"[BVH Engine v3] Applying scissor gait severity={v:.2f}")
        apply_scissor_gait(frames, v)

    for side in ("right", "left"):
        v = float(state.get(f"antalgic_{side}", 0.0))
        if v > 0:
            print(f"[BVH Engine v3] Applying antalgic gait ({side}) severity={v:.2f}")
            apply_antalgic_gait(frames, side, v)

    # ── Cadence (before stride operations, affects frame count) ──────────────
    v = float(state.get("cadence_reduction", 0.0))
    if v > 0:
        print(f"[BVH Engine v3] Cadence reduction severity={v:.2f}")
        frames, frame_time = apply_cadence_reduction(frames, frame_time, v)

    # ── Hip hike ──────────────────────────────────────────────────────────────
    for side in ("right", "left"):
        v = float(state.get(f"hip_hike_{side}", 0.0))
        if v > 0:
            print(f"[BVH Engine v3] Hip hike ({side}) severity={v:.2f}")
            apply_hip_hike(frames, side, v)

    # ── Primitive parameters ──────────────────────────────────────────────────
    for side in ("right", "left"):
        v = float(state.get(f"ankle_drop_{side}", 0.0))
        if v > 0:
            print(f"[BVH Engine v3] Ankle drop ({side}) severity={v:.2f}")
            apply_ankle_drop(frames, side, v)

        v = float(state.get(f"knee_stiffness_{side}", 0.0))
        if v > 0:
            print(f"[BVH Engine v3] Knee stiffness ({side}) severity={v:.2f}")
            apply_knee_stiffness(frames, side, v)

        v = float(state.get(f"stride_asymmetry_{side}", 0.0))
        if v > 0:
            print(f"[BVH Engine v3] Stride asymmetry ({side}) severity={v:.2f}")
            apply_stride_asymmetry(frames, side, v)

        v = float(state.get(f"arm_swing_reduction_{side}", 0.0))
        if v > 0:
            print(f"[BVH Engine v3] Arm swing reduction ({side}) severity={v:.2f}")
            apply_arm_swing_reduction(frames, side, v)

    for side in ("right", "left"):
        v = float(state.get(f"trunk_lean_{side}", 0.0))
        if v > 0:
            print(f"[BVH Engine v3] Trunk lean ({side}) severity={v:.2f}")
            apply_trunk_lean(frames, side, v, dynamic=True)

    write_bvh(output_bvh, header, frame_time, frames)
    print(f"[BVH Engine v3] → {output_bvh}  ({len(frames)} frames, {frame_time*1000:.1f}ms/frame)")
    return output_bvh


# ─────────────────────────────────────────────────────────────────────────────
# Standalone test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    inp = sys.argv[1] if len(sys.argv) > 1 else "bvh_folder/bvh_0_out.bvh"

    tests = [
        ("hemiplegic_right_severe", {"hemiplegic_right": 0.8}),
        ("parkinson",               {"parkinsonian_shuffle": 0.7}),
        ("crouch",                  {"crouch_gait": 0.7}),
        ("stiff_knee_right",        {"knee_stiffness_right": 0.8, "hip_hike_right": 0.6}),
        ("antalgic_left_hip",       {"antalgic_left": 0.7}),
        ("scissor",                 {"scissor_gait": 0.6}),
    ]

    for name, state in tests:
        out = f"impaired_bvh_folder/test_{name}.bvh"
        apply_impairment(inp, state, out)
        print()