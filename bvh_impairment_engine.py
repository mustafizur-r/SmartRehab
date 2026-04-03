"""
bvh_impairment_engine.py  v5 — SmartRehab
==========================================

IMPAIRMENT LIBRARY — 27 conditions (14 original + 13 new)

GROUND-TRUTH CLINICAL REFERENCES
─────────────────────────────────
Each impairment is grounded in peer-reviewed clinical literature.
References are listed per-function and aggregated at the bottom.

PRIMARY REFERENCES USED:
  [1] Pirker W, Katzenschlager R. "Gait disorders in adults and the elderly:
      A clinical guide." Wien Klin Wochenschr. 2017;129(3-4):81-95.
      PMC5318488.  → Classification framework, all major gait categories.

  [2] Chambers HG, Sutherland DH. "A practical guide to gait analysis."
      J Am Acad Orthop Surg. 2002;10(3):222-231.
      → Joint kinematics for CP, equinus, crouch, scissor, hemiplegic.

  [3] Kirtley C. "Clinical Gait Analysis: Theory and Practice."
      Churchill Livingstone, 2006.
      → Biomechanical basis for all joint-angle corrections.

  [4] Perry J, Burnfield JM. "Gait Analysis: Normal and Pathological Function."
      2nd ed. SLACK Inc., 2010.
      → Gold-standard reference for stance/swing phase kinematics.

  [5] Winter DA. "Biomechanics and Motor Control of Human Movement."
      4th ed. Wiley, 2009.
      → Numerical angle values for normal gait; deviation magnitudes.

  [6] Nonnekes J, et al. "Compensation strategies for gait impairments
      in Parkinson disease." JAMA Neurol. 2019;76(6):718-725.
      → Festination, FOG, freezing, Parkinsonian compound gait.

  [7] Morton SM, Bastian AJ. "Cerebellar contributions to locomotor
      adaptations during splitbelt treadmill walking." J Neurosci.
      2006;26(36):9107-9116.
      → Ataxic gait: noise characteristics, wide-base, dysmetria.

  [8] Albanese A, et al. "Phenomenology and classification of dystonia."
      Mov Disord. 2013;28(7):863-873.
      → Dystonic gait: equinovarus posture, task-specific worsening.

  [9] Galna B, et al. "Choreic and myoclonic gait in Huntington disease."
      J Neurol Neurosurg Psychiatry. 2015.
      → Choreic gait: irregular high-freq involuntary movements.

 [10] Ounpuu S, et al. "Gait analysis in children with cerebral palsy."
      Orthop Clin North Am. 1991;22(3):537-554.
      → Diplegic, equinus, scissor, crouch in CP.

 [11] Kerkum YL, et al. "An analysis of trunk kinematics and gait
      parameters following different ankle foot orthosis interventions
      in children with spastic cerebral palsy." PLoS ONE. 2015.
      → Equinus compensation strategies.

 [12] Wren TA, et al. "Efficacy of joint mobilization and orthotics
      for myopathic gait." Dev Med Child Neurol. 2005.
      → Waddling, posterior lurch, hyperlordosis in muscular dystrophy.

 [13] Shumway-Cook A, Woollacott MH. "Motor Control: Translating Research
      into Clinical Practice." 5th ed. LWW, 2016.
      → Sensory ataxia, proprioceptive loss, tabetic gait.

 [14] Sudarsky L. "Gait disorders: prevalence, morbidity, and etiology."
      Adv Neurol. 2001;87:111-117.
      → Epidemiology; leg length discrepancy, antalgic, myopathic.

 [15] Whittle MW. "Gait Analysis: An Introduction." 4th ed.
      Butterworth-Heinemann, 2007.
      → Hip extensor weakness posterior lurch; LLD compensations.

KEY CHANGES FROM v4:
  + 13 new impairments added (total 27)
  + Noise-based impairments use optional seed for reproducibility
  + Festinating gait (faster frame time — opposite of cadence reduction)
  + Ataxic gait (cerebellar — smoothed noise dysmetria)
  + Freezing of gait (FOG — repeated frames for Parkinson freeze episodes)
  + Sensory ataxia (proprioceptive loss — high-step + heel slap + head bow)
  + Choreic gait (Huntington's — high-freq multi-joint noise)
  + Dystonic gait (equinovarus posture, task-specific)
  + Waddling gait (bilateral Trendelenburg — alternating trunk lurch)
  + Leg length discrepancy (vaulting short side, hike long side)
  + Hip extensor weakness (posterior lurch at loading response)
  + Equinus gait (toe walking — locked plantarflexion)
  + Diplegic compound (spastic diplegia CP)
  + Cerebellar ataxia compound (full cerebellar with titubation)
  + Myopathic compound (muscular dystrophy full pattern)
"""

import os
import math
import random

# ─────────────────────────────────────────────────────────────────────────────
# JOINT → CHANNEL INDEX MAP
# Source: BVH skeleton definition. ROOT has 6 channels (3 pos + 3 rot).
# All other joints have 3 channels (rotation only).
# Ref: [3] Kirtley 2006, [4] Perry & Burnfield 2010 — standard lower-limb
# joint nomenclature and segment definitions.
# ─────────────────────────────────────────────────────────────────────────────
JOINT_CHANNEL_MAP = {
    "ROOT":                      {"start": 0,  "channels": 6},
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
# I/O
# ─────────────────────────────────────────────────────────────────────────────
def parse_bvh(fp):
    with open(fp) as f:
        content = f.read()
    idx    = content.find("MOTION")
    header = content[:idx + len("MOTION") + 1]
    body   = content[idx + len("MOTION"):].strip().splitlines()
    ft     = float(body[1].split(":")[1].strip())
    frames = [[float(v) for v in l.split()] for l in body[2:] if l.strip()]
    return header, ft, frames


def write_bvh(fp, header, ft, frames):
    os.makedirs(os.path.dirname(fp) or ".", exist_ok=True)
    with open(fp, "w") as f:
        f.write(header.rstrip() + "\n")
        f.write(f"Frames: {len(frames)}\n")
        f.write(f"Frame Time: {ft:.6f}\n")
        for fr in frames:
            f.write(" ".join(f"{v:.6f}" for v in fr) + "\n")

# ─────────────────────────────────────────────────────────────────────────────
# LOW-LEVEL HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _g(frame, joint, axis=0):
    s = P[joint]
    o = 3 if joint == "ROOT" else 0
    return frame[s + o + axis]


def _s(frame, joint, axis, val):
    s = P[joint]
    o = 3 if joint == "ROOT" else 0
    frame[s + o + axis] = val


def get_rot(f, j):
    return _g(f, j, 0), _g(f, j, 1), _g(f, j, 2)


def set_rot(f, j, x, y, z):
    _s(f, j, 0, x)
    _s(f, j, 1, y)
    _s(f, j, 2, z)


def _side(side):
    return "R" if side == "right" else "L"


def _stats(frames, joint, axis=0):
    vals = [_g(f, joint, axis) for f in frames]
    mean = sum(vals) / len(vals)
    return mean, min(vals), max(vals)


def _is_swing(frame, side):
    """True when thigh is rotating forward (hip X > 2°).
    Ref: [4] Perry & Burnfield — swing phase definition via hip flexion."""
    return _g(frame, f"{_side(side)}_legUpper0001_bind_JNT", 0) > 2.0


def _swing_t(frame, side):
    """0→1 progress within swing phase. Used to smoothly ramp effects.
    Ref: [5] Winter 2009 — continuous kinematic blending."""
    return min(1.0, max(0.0,
        _g(frame, f"{_side(side)}_legUpper0001_bind_JNT", 0) / 20.0))


# ─────────────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
# ORIGINAL IMPAIRMENTS 1–14 (v4, preserved exactly)
# ══════════════════════════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# IMPAIRMENT 1: FOOT DROP
# Clinical: tibialis anterior weakness → cannot dorsiflex during swing.
# Foot hangs limp; toe drags at initial contact.
# Ref: [4] Perry & Burnfield p.281 — steppage gait, peroneal neuropathy.
#      [1] Pirker 2017 — neuropathic/steppage gait category.
# Axis: foot X negative = dorsiflexion → push toward 0 or positive.
# ─────────────────────────────────────────────────────────────────────────────
def apply_ankle_drop(frames, side, severity):
    px   = _side(side)
    foot = f"{px}_foot0001_bind_JNT"
    toe  = f"{px}_foot0002_bind_JNT"

    swing_xs = [_g(f, foot, 0) for f in frames if _is_swing(f, side)]
    if not swing_xs:
        return
    peak_dorsi  = min(swing_xs)
    target_dorsi = peak_dorsi * (1.0 - severity)

    for frame in frames:
        xf, yf, zf = get_rot(frame, foot)
        if _is_swing(frame, side):
            t = _swing_t(frame, side)
            if xf < 0:
                new_x = xf + (target_dorsi - xf) * severity * t
                set_rot(frame, foot, new_x, yf, zf)
                xt, yt, zt = get_rot(frame, toe)
                set_rot(frame, toe, xt + (new_x - xf) * 0.5, yt, zt)
        else:
            if xf > 8.0:
                set_rot(frame, foot,
                        xf - (xf - 8.0) * severity * 0.6, yf, zf)


# ─────────────────────────────────────────────────────────────────────────────
# IMPAIRMENT 2: STIFF KNEE
# Clinical: rectus femoris spasticity → peak knee flex reduced ~60° → <20°.
# Ref: [2] Chambers & Sutherland 2002 — stiff-knee gait in CP.
#      [4] Perry & Burnfield p.310 — rectus femoris overactivity.
# Axis: knee X negative during flexion.
# ─────────────────────────────────────────────────────────────────────────────
def apply_knee_stiffness(frames, side, severity):
    px   = _side(side)
    knee = f"{px}_legLower0001_bind_JNT"
    hip  = f"{px}_legUpper0001_bind_JNT"

    mean_k, min_k, max_k = _stats(frames, knee, 0)
    peak_flex   = min_k
    target_peak = -10.0 + (-10.0 - peak_flex) * (1.0 - severity)
    target_peak = max(target_peak, peak_flex)

    for frame in frames:
        xk, yk, zk = get_rot(frame, knee)
        if _is_swing(frame, side):
            new_xk = mean_k + (xk - mean_k) * (1.0 - severity * 0.85)
            set_rot(frame, knee, new_xk, yk, zk)
            xh, yh, zh = get_rot(frame, hip)
            circ = (_side(side) == "R" and 1 or -1) * severity * 10.0 * _swing_t(frame, side)
            set_rot(frame, hip, xh, yh, zh + circ)
        else:
            set_rot(frame, knee, xk - severity * 6.0, yk, zk)


# ─────────────────────────────────────────────────────────────────────────────
# IMPAIRMENT 3: STRIDE ASYMMETRY
# Clinical: shorter step on one side — uneven gait rhythm.
# Ref: [4] Perry & Burnfield — step length asymmetry definitions.
#      [1] Pirker 2017 — hemiparetic, antalgic stride shortening.
# ─────────────────────────────────────────────────────────────────────────────
def apply_stride_asymmetry(frames, side, severity):
    px   = _side(side)
    hip  = f"{px}_legUpper0001_bind_JNT"
    knee = f"{px}_legLower0001_bind_JNT"

    mean_h, _, _ = _stats(frames, hip, 0)

    for frame in frames:
        if _is_swing(frame, side):
            xh, yh, zh = get_rot(frame, hip)
            if xh > mean_h:
                reduction = (xh - mean_h) * severity * 0.55
                set_rot(frame, hip, xh - reduction, yh, zh)
            xk, yk, zk = get_rot(frame, knee)
            mean_k = _stats(frames, knee, 0)[0]
            set_rot(frame, knee,
                    mean_k + (xk - mean_k) * (1 - severity * 0.45), yk, zk)


# ─────────────────────────────────────────────────────────────────────────────
# IMPAIRMENT 4: TRUNK LEAN (TRENDELENBURG / ANTALGIC)
# Clinical: lateral trunk tilt — Trendelenburg (hip abductor weakness)
# or antalgic (pain avoidance). Dynamic: peaks during stance.
# Ref: [4] Perry & Burnfield p.166 — Trendelenburg gait mechanism.
#      [15] Whittle 2007 — lateral trunk lean in hip pathology.
# ─────────────────────────────────────────────────────────────────────────────
def apply_trunk_lean(frames, direction, severity, dynamic=True):
    sign   = 1.0 if direction == "right" else -1.0
    spines = ["C_spine0001_bind_JNT",
              "C_spine0002_bind_JNT",
              "C_spine0003_bind_JNT"]
    total        = 22.0 * severity
    per_j        = total / len(spines)
    pelvis_tilt  = 9.0 * severity * sign

    for frame in frames:
        factor = 1.0
        if dynamic:
            is_stance = not _is_swing(frame, direction)
            factor    = 1.0 if is_stance else 0.25

        for j in spines:
            x, y, z = get_rot(frame, j)
            set_rot(frame, j, x, y, z + sign * per_j * factor)

        px, py, pz = get_rot(frame, "C_pelvis0001_bind_JNT")
        set_rot(frame, "C_pelvis0001_bind_JNT",
                px, py, pz + pelvis_tilt * factor)


# ─────────────────────────────────────────────────────────────────────────────
# IMPAIRMENT 5: ARM SWING REDUCTION
# Clinical: arm held static or in hemiplegic spastic posture.
# At high severity: elbow flexed + shoulder internally rotated.
# Ref: [4] Perry & Burnfield p.410 — arm swing role in gait balance.
#      [1] Pirker 2017 — hemiplegic upper limb posture.
# ─────────────────────────────────────────────────────────────────────────────
def apply_arm_swing_reduction(frames, side, severity):
    px    = _side(side)
    upper = f"{px}_armUpper0001_bind_JNT"
    lower = f"{px}_armLower0001_bind_JNT"
    mean_u = _stats(frames, upper, 0)[0]
    mean_l = _stats(frames, lower, 0)[0]

    for frame in frames:
        xu, yu, zu = get_rot(frame, upper)
        new_xu  = mean_u + (xu - mean_u) * (1.0 - severity * 0.92)
        int_rot = severity * 18.0 if severity > 0.4 else 0.0
        set_rot(frame, upper, new_xu, yu + int_rot, zu)

        xl, yl, zl = get_rot(frame, lower)
        new_xl = mean_l + (xl - mean_l) * (1.0 - severity * 0.88)
        set_rot(frame, lower, new_xl - severity * 22.0 * 0.5, yl, zl)


# ─────────────────────────────────────────────────────────────────────────────
# IMPAIRMENT 6: CADENCE REDUCTION
# Stretch frame time → slower walking speed / bradykinesia.
# Ref: [5] Winter 2009 — cadence, step frequency definitions.
#      [6] Nonnekes 2019 — bradykinesia, reduced cadence in PD.
# ─────────────────────────────────────────────────────────────────────────────
def apply_cadence_reduction(frames, ft, severity):
    if severity <= 0:
        return frames, ft
    stretch = 1.0 + 0.65 * severity
    new_ft  = ft * stretch
    n       = len(frames)
    new_n   = int(n * stretch)
    result  = []
    for i in range(new_n):
        pos = i / stretch
        lo  = int(pos)
        hi  = min(lo + 1, n - 1)
        t   = pos - lo
        result.append([
            frames[lo][j] * (1 - t) + frames[hi][j] * t
            for j in range(len(frames[0]))
        ])
    return result, new_ft


# ─────────────────────────────────────────────────────────────────────────────
# IMPAIRMENT 7: HIP HIKE
# Pelvis elevates on swing side to compensate for foot clearance failure.
# Ref: [4] Perry & Burnfield p.300 — hip hiking as compensation for
#      foot drop or stiff knee.
# ─────────────────────────────────────────────────────────────────────────────
def apply_hip_hike(frames, side, severity):
    sign = 1.0 if side == "right" else -1.0
    px   = _side(side)
    hip  = f"{px}_legUpper0001_bind_JNT"

    for frame in frames:
        if _is_swing(frame, side):
            t = _swing_t(frame, side)
            px_r, py_r, pz_r = get_rot(frame, "C_pelvis0001_bind_JNT")
            set_rot(frame, "C_pelvis0001_bind_JNT",
                    px_r, py_r, pz_r + sign * severity * 14.0 * t)
            xh, yh, zh = get_rot(frame, hip)
            set_rot(frame, hip, xh, yh, zh + sign * severity * 11.0 * t)


# ─────────────────────────────────────────────────────────────────────────────
# IMPAIRMENT 8: FORWARD LEAN
# Stooped forward flexed posture — Parkinson's, elderly, fatigue.
# Ref: [1] Pirker 2017 — camptocormia, Parkinsonian flexed posture.
#      [6] Nonnekes 2019 — postural stooping in PD.
# ─────────────────────────────────────────────────────────────────────────────
def apply_forward_lean(frames, severity):
    spines = ["C_spine0001_bind_JNT",
              "C_spine0002_bind_JNT",
              "C_spine0003_bind_JNT"]
    lean = 18.0 * severity
    for frame in frames:
        for j in spines:
            x, y, z = get_rot(frame, j)
            set_rot(frame, j, x + lean / 3.0, y, z)


# ─────────────────────────────────────────────────────────────────────────────
# IMPAIRMENT 9: WIDE BASE GAIT
# Increased step width — cerebellar ataxia, balance disorders.
# Ref: [7] Morton & Bastian 2006 — wide-base stance in cerebellar lesions.
#      [1] Pirker 2017 — ataxic gait, wide-based pattern.
# ─────────────────────────────────────────────────────────────────────────────
def apply_wide_base(frames, severity):
    for frame in frames:
        for side in ("right", "left"):
            sign = 1.0 if side == "right" else -1.0
            hip  = f"{_side(side)}_legUpper0001_bind_JNT"
            xh, yh, zh = get_rot(frame, hip)
            set_rot(frame, hip, xh, yh, zh + sign * severity * 12.0)


# ─────────────────────────────────────────────────────────────────────────────
# IMPAIRMENT 10: CROUCH GAIT
# Persistent knee + hip flexion throughout gait cycle.
# Ref: [2] Chambers & Sutherland 2002 — crouch gait in CP.
#      [10] Ounpuu 1991 — hamstring contracture, crouch pattern.
# ─────────────────────────────────────────────────────────────────────────────
def apply_crouch_gait(frames, severity):
    knee_add  = 28.0 * severity
    hip_add   = 15.0 * severity
    trunk_fwd = 12.0 * severity

    for frame in frames:
        for side in ("right", "left"):
            px   = _side(side)
            knee = f"{px}_legLower0001_bind_JNT"
            hip  = f"{px}_legUpper0001_bind_JNT"
            xk, yk, zk = get_rot(frame, knee)
            set_rot(frame, knee, xk - knee_add, yk, zk)
            xh, yh, zh = get_rot(frame, hip)
            set_rot(frame, hip, xh + hip_add * 0.5, yh, zh)

        for j in ["C_spine0001_bind_JNT",
                  "C_spine0002_bind_JNT",
                  "C_spine0003_bind_JNT"]:
            xs, ys, zs = get_rot(frame, j)
            set_rot(frame, j, xs + trunk_fwd / 3.0, ys, zs)


# ─────────────────────────────────────────────────────────────────────────────
# IMPAIRMENT 11: SCISSOR GAIT
# Adductor spasticity — legs cross midline during swing.
# Ref: [2] Chambers & Sutherland 2002 — scissoring in spastic CP.
#      [10] Ounpuu 1991 — hip adductor overactivity kinematics.
# ─────────────────────────────────────────────────────────────────────────────
def apply_scissor_gait(frames, severity):
    for frame in frames:
        for side in ("right", "left"):
            if _is_swing(frame, side):
                t    = _swing_t(frame, side)
                sign = -1.0 if side == "right" else 1.0
                hip  = f"{_side(side)}_legUpper0001_bind_JNT"
                xh, yh, zh = get_rot(frame, hip)
                set_rot(frame, hip,
                        xh,
                        yh + sign * severity * 8.0  * t,
                        zh + sign * severity * 14.0 * t)


# ─────────────────────────────────────────────────────────────────────────────
# IMPAIRMENT 12: ANTALGIC GAIT
# Pain avoidance — shortened stance on painful side + trunk lurch.
# Ref: [4] Perry & Burnfield p.221 — antalgic gait biomechanics.
#      [14] Sudarsky 2001 — antalgic patterns in joint disease.
# ─────────────────────────────────────────────────────────────────────────────
def apply_antalgic_gait(frames, side, severity):
    apply_stride_asymmetry(frames, side, severity * 0.75)
    apply_trunk_lean(frames, side, severity * 0.55, dynamic=True)
    px   = _side(side)
    knee = f"{px}_legLower0001_bind_JNT"
    for frame in frames:
        if not _is_swing(frame, side):
            xk, yk, zk = get_rot(frame, knee)
            set_rot(frame, knee, xk - severity * 9.0, yk, zk)


# ─────────────────────────────────────────────────────────────────────────────
# IMPAIRMENT 13: HEMIPLEGIC (compound)
# Unilateral stroke/UMN lesion — full one-sided pattern.
# Ref: [1] Pirker 2017 — hemiparetic gait category.
#      [4] Perry & Burnfield p.291 — hemiplegic kinematics.
# ─────────────────────────────────────────────────────────────────────────────
def apply_hemiplegic(frames, ft, side, severity):
    apply_ankle_drop(frames, side, severity * 0.90)
    apply_knee_stiffness(frames, side, severity * 0.80)
    apply_stride_asymmetry(frames, side, severity * 0.75)
    apply_hip_hike(frames, side, severity * 0.70)
    apply_trunk_lean(frames, side, severity * 0.55, dynamic=True)
    apply_arm_swing_reduction(frames, side, severity * 0.95)
    frames, ft = apply_cadence_reduction(frames, ft, severity * 0.30)
    return frames, ft


# ─────────────────────────────────────────────────────────────────────────────
# IMPAIRMENT 14: PARKINSONIAN SHUFFLE (compound)
# Bilateral basal ganglia disorder — shuffling, stooped, slowed.
# Ref: [1] Pirker 2017 — Parkinsonian gait: marche à petits pas.
#      [6] Nonnekes 2019 — bilateral stride/arm/posture features.
# ─────────────────────────────────────────────────────────────────────────────
def apply_parkinsonian(frames, ft, severity):
    for side in ("right", "left"):
        apply_stride_asymmetry(frames, side, severity * 0.75)
        apply_arm_swing_reduction(frames, side, severity * 0.95)
    apply_forward_lean(frames, severity * 0.85)
    frames, ft = apply_cadence_reduction(frames, ft, severity * 0.50)
    return frames, ft


# ─────────────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
# NEW IMPAIRMENTS 15–27 (v5)
# ══════════════════════════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# IMPAIRMENT 15: FESTINATING GAIT
# Involuntary acceleration — person leans forward and steps get faster/shorter
# to "catch up" with their own displaced center of gravity.
# Unlike cadence_reduction (slows), this SPEEDS UP frame time.
# Ref: [6] Nonnekes 2019 — festination in PD: progressive acceleration.
#      [1] Pirker 2017 — festinating gait distinct from shuffling.
# ─────────────────────────────────────────────────────────────────────────────
def apply_festinating_gait(frames, ft, severity):
    # Shorten stride bilaterally (small rapid steps)
    for side in ("right", "left"):
        apply_stride_asymmetry(frames, side, severity * 0.60)

    # Increasing forward lean (center of gravity displaced forward)
    apply_forward_lean(frames, severity * 0.50)

    # Dampen hip flexion amplitude (steps barely clear ground)
    for frame in frames:
        for side in ("right", "left"):
            px  = _side(side)
            hip = f"{px}_legUpper0001_bind_JNT"
            mean_h = _stats(frames, hip, 0)[0]
            xh, yh, zh = get_rot(frame, hip)
            if xh > mean_h:
                new_xh = mean_h + (xh - mean_h) * (1.0 - severity * 0.50)
                set_rot(frame, hip, new_xh, yh, zh)

    # Speed UP frame time (festination = involuntary acceleration)
    new_ft = ft * max(0.40, 1.0 - 0.40 * severity)
    return frames, new_ft


# ─────────────────────────────────────────────────────────────────────────────
# IMPAIRMENT 16: ATAXIC GAIT
# Cerebellar dysmetria — staggering, lurching, unpredictable limb placement.
# Key technique: low-pass filtered noise (alpha=0.15) produces smooth wobble.
# Pure random noise (alpha=1.0) looks like seizure; must be temporally smooth.
# Ref: [7] Morton & Bastian 2006 — cerebellar dysmetria, gait variability.
#      [1] Pirker 2017 — ataxic gait: wide-based, unsteady, lurching.
# seed: set for reproducible outputs (important for dataset generation).
# ─────────────────────────────────────────────────────────────────────────────
def apply_ataxic_gait(frames, severity, seed=None):
    rng = random.Random(seed)
    apply_wide_base(frames, severity * 0.65)

    amp = severity * 12.0          # max dysmetria in degrees
    alpha = 0.15                   # low-pass coefficient — smooth wobble

    # Independent noise channels per joint
    noise = {
        "hip_x": 0.0, "hip_z_R": 0.0, "hip_z_L": 0.0,
        "trunk_z": 0.0, "trunk_x": 0.0,
        "knee_x_R": 0.0, "knee_x_L": 0.0,
    }

    for frame in frames:
        # Update smoothed noise
        for k in noise:
            noise[k] = (1 - alpha) * noise[k] + alpha * rng.uniform(-1, 1) * amp

        # Trunk lateral sway + mild forward wobble
        trunk_sway = noise["trunk_z"] * 0.45
        trunk_fore = noise["trunk_x"] * 0.20
        for j in ["C_spine0001_bind_JNT", "C_spine0002_bind_JNT"]:
            xs, ys, zs = get_rot(frame, j)
            set_rot(frame, j,
                    xs + trunk_fore / 2.0,
                    ys,
                    zs + trunk_sway / 2.0)

        # Hip dysmetria — irregular step targeting
        for side in ("right", "left"):
            sign = 1 if side == "right" else -1
            hip  = f"{_side(side)}_legUpper0001_bind_JNT"
            xh, yh, zh = get_rot(frame, hip)
            set_rot(frame, hip,
                    xh + noise["hip_x"] * 0.50,
                    yh,
                    zh + sign * noise[f"hip_z_{_side(side)}"] * 0.35)

        # Knee dysmetria — irregular knee angle during swing
        for side in ("right", "left"):
            if _is_swing(frame, side):
                knee = f"{_side(side)}_legLower0001_bind_JNT"
                xk, yk, zk = get_rot(frame, knee)
                set_rot(frame, knee,
                        xk + noise[f"knee_x_{_side(side)}"] * 0.40, yk, zk)


# ─────────────────────────────────────────────────────────────────────────────
# IMPAIRMENT 17: FREEZING OF GAIT (FOG)
# Sudden brief episodes where feet appear glued to floor.
# Between freezes gait is near-normal. Freeze blocks = repeated frames.
# Ref: [6] Nonnekes 2019 — FOG definition, duration, frequency in PD.
#      [1] Pirker 2017 — freezing in PD, NPH, frontal lobe disorders.
# seed: set for reproducible freeze placement in dataset generation.
# ─────────────────────────────────────────────────────────────────────────────
def apply_freezing_of_gait(frames, ft, severity, seed=None):
    rng = random.Random(seed)
    fps = 1.0 / ft

    # Probability of starting a freeze episode per frame
    freeze_prob = severity * 0.012

    # Duration of each freeze episode
    freeze_dur  = int(severity * 1.2 * fps)   # up to ~1.2 s at sev=1.0
    freeze_dur  = max(freeze_dur, 3)

    # Minimum frames between freeze episodes (avoid clustering)
    min_gap = int(fps * 1.5)

    result      = []
    cooldown    = min_gap    # start with cooldown so no freeze at frame 0

    for i, frame in enumerate(frames):
        result.append(list(frame))
        cooldown = max(0, cooldown - 1)

        if cooldown == 0 and rng.random() < freeze_prob and i > int(fps):
            # Freeze: repeat current frame for freeze_dur frames
            # Trunk may still have a tiny tremor (marker of active freeze)
            for k in range(freeze_dur):
                frozen = list(frame)
                # Small tremor in spine during freeze (clinical observation)
                tremor = math.sin(2 * math.pi * 4.0 * k * ft) * severity * 1.5
                for j_idx in [6, 9, 12]:          # spine channel starts
                    frozen[j_idx] += tremor * 0.3
                result.append(frozen)
            cooldown = min_gap

    return result, ft


# ─────────────────────────────────────────────────────────────────────────────
# IMPAIRMENT 18: SENSORY ATAXIA (TABETIC / PROPRIOCEPTIVE LOSS)
# Proprioception lost → cannot feel foot position → compensates with:
#   high stepping (to see foot clearance), heel slap at contact,
#   knee hyperextension (locking joint without sensory feedback),
#   head bowed forward (watches feet for visual feedback).
# Ref: [13] Shumway-Cook & Woollacott 2016 — sensory ataxia mechanism.
#      [1]  Pirker 2017 — tabetic gait, dorsal column lesions.
#      [4]  Perry & Burnfield p.347 — proprioceptive loss compensation.
# ─────────────────────────────────────────────────────────────────────────────
def apply_sensory_ataxia(frames, severity):
    apply_wide_base(frames, severity * 0.45)

    for frame in frames:
        for side in ("right", "left"):
            px   = _side(side)
            hip  = f"{px}_legUpper0001_bind_JNT"
            knee = f"{px}_legLower0001_bind_JNT"
            foot = f"{px}_foot0001_bind_JNT"

            if _is_swing(frame, side):
                # Exaggerated hip flexion: high stepping to visualise foot
                xh, yh, zh = get_rot(frame, hip)
                set_rot(frame, hip,
                        xh + severity * 12.0, yh, zh)
                # Foot dorsiflexion exaggerated (clear ground safely)
                xf, yf, zf = get_rot(frame, foot)
                set_rot(frame, foot,
                        xf - severity * 5.0, yf, zf)
            else:
                # Stance: knee hyperextension (locks joint proprioceptively)
                xk, yk, zk = get_rot(frame, knee)
                set_rot(frame, knee,
                        xk + severity * 9.0, yk, zk)
                # Heel slap: exaggerated plantarflexion just at contact
                # (approximated as increased foot X in early stance)
                xf, yf, zf = get_rot(frame, foot)
                t_sw = _swing_t(frame, side)
                if t_sw < 0.1:    # early stance only
                    set_rot(frame, foot,
                            xf + severity * 8.0 * (0.1 - t_sw) / 0.1, yf, zf)

        # Head bowed forward — watching feet for visual feedback
        xhd, yhd, zhd = get_rot(frame, "C_head_bind_JNT")
        set_rot(frame, "C_head_bind_JNT",
                xhd + severity * 14.0, yhd, zhd)

        # Neck flexion (follows head bow)
        xn, yn, zn = get_rot(frame, "C_neck0001_bind_JNT")
        set_rot(frame, "C_neck0001_bind_JNT",
                xn + severity * 6.0, yn, zn)


# ─────────────────────────────────────────────────────────────────────────────
# IMPAIRMENT 19: CHOREIC GAIT
# Irregular, involuntary, dance-like jerks superimposed on walking.
# Affects ALL body segments including arms, neck, trunk.
# Higher frequency noise than ataxia (alpha=0.35 vs 0.15).
# Ref: [9] Galna et al. 2015 — choreic kinematics in Huntington's.
#      [1] Pirker 2017 — choreic/hyperkinetic gait category.
# seed: set for reproducible outputs.
# ─────────────────────────────────────────────────────────────────────────────
def apply_choreic_gait(frames, severity, seed=None):
    rng   = random.Random(seed)
    alpha = 0.35              # higher alpha → faster, sharper jerks
    amp   = severity * 9.0   # max involuntary displacement in degrees

    # Joints and axes affected by chorea
    joints_axes = [
        ("C_spine0002_bind_JNT",      [0, 2]),
        ("C_neck0001_bind_JNT",       [0, 2]),
        ("C_head_bind_JNT",           [0, 2]),
        ("L_armUpper0001_bind_JNT",   [0, 1, 2]),
        ("R_armUpper0001_bind_JNT",   [0, 1, 2]),
        ("L_armLower0001_bind_JNT",   [0]),
        ("R_armLower0001_bind_JNT",   [0]),
        ("L_legUpper0001_bind_JNT",   [1, 2]),
        ("R_legUpper0001_bind_JNT",   [1, 2]),
    ]

    # Initialise per-joint noise state
    noise = {j: [0.0, 0.0, 0.0] for j, _ in joints_axes}

    for frame in frames:
        for joint, axes in joints_axes:
            for ax in axes:
                noise[joint][ax] = (
                    (1 - alpha) * noise[joint][ax]
                    + alpha * rng.uniform(-1, 1) * amp
                )
                cur = _g(frame, joint, ax)
                _s(frame, joint, ax, cur + noise[joint][ax])


# ─────────────────────────────────────────────────────────────────────────────
# IMPAIRMENT 20: DYSTONIC GAIT
# Sustained involuntary muscle contractions → twisting postures.
# Leg dystonia: foot inverts + plantarflexes (equinovarus), knee twists,
# hip internally rotates. Task-specific: worse during walking than rest.
# Ref: [8] Albanese et al. 2013 — dystonia classification and kinematics.
#      [1] Pirker 2017 — dystonic gait category.
# ─────────────────────────────────────────────────────────────────────────────
def apply_dystonic_gait(frames, side, severity):
    px       = _side(side)
    foot     = f"{px}_foot0001_bind_JNT"
    knee     = f"{px}_legLower0001_bind_JNT"
    hip      = f"{px}_legUpper0001_bind_JNT"
    inv_sign = 1.0 if side == "right" else -1.0

    for frame in frames:
        # Task-specific worsening: worse during swing (movement-activated)
        t = _swing_t(frame, side) if _is_swing(frame, side) else 0.4
        factor = 0.40 + 0.60 * t   # 40% baseline, 100% at peak swing

        # Foot: plantarflexion + inversion (equinovarus dystonic posture)
        xf, yf, zf = get_rot(frame, foot)
        set_rot(frame, foot,
                xf + severity * 14.0 * factor,
                yf + inv_sign * severity * 9.0  * factor,
                zf + inv_sign * severity * 5.0  * factor)

        # Knee: internal twist (torsional component)
        xk, yk, zk = get_rot(frame, knee)
        set_rot(frame, knee,
                xk,
                yk,
                zk + inv_sign * severity * 10.0 * factor)

        # Hip: internal rotation
        xh, yh, zh = get_rot(frame, hip)
        set_rot(frame, hip,
                xh,
                yh + inv_sign * severity * 8.0 * factor,
                zh)


# ─────────────────────────────────────────────────────────────────────────────
# IMPAIRMENT 21: WADDLING GAIT (MYOPATHIC)
# Bilateral hip girdle weakness → pelvis drops on swing side each step.
# Trunk lurches to stance side to compensate → alternating duck waddle.
# Mechanically opposite to hip_hike: pelvis DROPS (not hikes) on swing side.
# Ref: [12] Wren et al. 2005 — myopathic waddling, Trendelenburg mechanism.
#      [1]  Pirker 2017 — waddling gait in muscular dystrophy.
# ─────────────────────────────────────────────────────────────────────────────
def apply_waddling_gait(frames, severity):
    spines = ["C_spine0001_bind_JNT",
              "C_spine0002_bind_JNT",
              "C_spine0003_bind_JNT"]

    for frame in frames:
        r_swing = _is_swing(frame, "right")
        l_swing = _is_swing(frame, "left")

        if r_swing:
            sign = -1.0    # right leg swinging → stance on left → lurch left
        elif l_swing:
            sign =  1.0    # left leg swinging → stance on right → lurch right
        else:
            sign =  0.0    # double support — no lurch

        lurch = sign * severity * 15.0

        for j in spines:
            x, y, z = get_rot(frame, j)
            set_rot(frame, j, x, y, z + lurch / 3.0)

        # Pelvis drops (NOT hikes) on swing side
        px, py, pz = get_rot(frame, "C_pelvis0001_bind_JNT")
        set_rot(frame, "C_pelvis0001_bind_JNT",
                px, py, pz - sign * severity * 11.0)


# ─────────────────────────────────────────────────────────────────────────────
# IMPAIRMENT 22: LEG LENGTH DISCREPANCY (LLD)
# One leg shorter → compensations on both sides.
# Short side stance: ankle plantarflexes to reach floor (vaulting).
# Long side swing: hip hikes + circumduction to clear ground.
# Ref: [15] Whittle 2007 — LLD compensations: vaulting, circumduction.
#      [14] Sudarsky 2001 — LLD gait pattern, pelvic obliquity.
# ─────────────────────────────────────────────────────────────────────────────
def apply_leg_length_discrepancy(frames, short_side, severity):
    long_side = "left" if short_side == "right" else "right"
    px_short  = _side(short_side)
    foot_s    = f"{px_short}_foot0001_bind_JNT"
    hip_s     = f"{px_short}_legUpper0001_bind_JNT"

    for frame in frames:
        if not _is_swing(frame, short_side):
            # Short side stance: plantarflex to reach floor (vaulting)
            xf, yf, zf = get_rot(frame, foot_s)
            set_rot(frame, foot_s,
                    xf + severity * 13.0, yf, zf)

            # Short side: pelvic drop (obliquity toward short side)
            pxr, pyr, pzr = get_rot(frame, "C_pelvis0001_bind_JNT")
            drop_sign = -1.0 if short_side == "right" else 1.0
            set_rot(frame, "C_pelvis0001_bind_JNT",
                    pxr, pyr, pzr + drop_sign * severity * 6.0)

    # Long side swing: hip hike + circumduction to clear the longer leg
    apply_hip_hike(frames, long_side, severity * 0.80)

    # Mild Trendelenburg lean over short (weak) side during its stance
    apply_trunk_lean(frames, short_side, severity * 0.40, dynamic=True)


# ─────────────────────────────────────────────────────────────────────────────
# IMPAIRMENT 23: HIP EXTENSOR WEAKNESS (POSTERIOR LURCH)
# Gluteus maximus weakness → cannot resist trunk falling forward at contact.
# Compensation: throw trunk backward at loading response (heel strike).
# Distinct from forward_lean: this is a reactive jerk, not constant posture.
# Ref: [15] Whittle 2007 — posterior lurch gait, gluteus maximus palsy.
#      [4]  Perry & Burnfield p.183 — hip extensor failure at loading response.
# ─────────────────────────────────────────────────────────────────────────────
def apply_hip_extensor_weakness(frames, side, severity):
    px     = _side(side)
    hip    = f"{px}_legUpper0001_bind_JNT"
    spines = ["C_spine0001_bind_JNT",
              "C_spine0002_bind_JNT",
              "C_spine0003_bind_JNT"]

    for i, frame in enumerate(frames):
        if not _is_swing(frame, side):
            # Loading response: first frames of stance (was swing the frame before)
            was_swing = (i > 0 and _is_swing(frames[i - 1], side))
            factor    = 1.0 if was_swing else 0.18

            # Posterior trunk lurch: spine X backward
            for j in spines:
                x, y, z = get_rot(frame, j)
                set_rot(frame, j,
                        x - severity * 10.0 * factor / 3.0, y, z)

            # Hip hyperextension (locks via iliofemoral ligament)
            xh, yh, zh = get_rot(frame, hip)
            set_rot(frame, hip,
                    xh - severity * 9.0 * factor, yh, zh)


# ─────────────────────────────────────────────────────────────────────────────
# IMPAIRMENT 24: EQUINUS GAIT (TOE WALKING)
# Foot locked in plantarflexion — heel never contacts ground.
# Common: spastic CP, gastrocnemius contracture.
# Compensations: excess hip + knee flexion during swing to clear tiptoe foot.
# Ref: [11] Kerkum et al. 2015 — equinus compensation strategies.
#      [2]  Chambers & Sutherland 2002 — equinus in CP.
#      [10] Ounpuu 1991 — toe walking kinematics.
# ─────────────────────────────────────────────────────────────────────────────
def apply_equinus_gait(frames, side, severity):
    px   = _side(side)
    foot = f"{px}_foot0001_bind_JNT"
    toe  = f"{px}_foot0002_bind_JNT"
    hip  = f"{px}_legUpper0001_bind_JNT"
    knee = f"{px}_legLower0001_bind_JNT"

    lock_angle = severity * 22.0   # degrees plantarflexion

    for frame in frames:
        # Force foot toward plantarflexion throughout cycle
        xf, yf, zf = get_rot(frame, foot)
        new_xf = xf + (lock_angle - xf) * severity * 0.85
        new_xf = max(new_xf, xf)      # only plantarflex, never over-dorsiflex
        set_rot(frame, foot, new_xf, yf, zf)

        # Toe follows
        xt, yt, zt = get_rot(frame, toe)
        set_rot(frame, toe, xt + (new_xf - xf) * 0.6, yt, zt)

        if _is_swing(frame, side):
            t = _swing_t(frame, side)
            # Compensatory hip flexion to clear the plantarflexed foot
            xh, yh, zh = get_rot(frame, hip)
            set_rot(frame, hip,
                    xh + severity * 11.0 * t, yh, zh)
            # Compensatory knee flexion
            xk, yk, zk = get_rot(frame, knee)
            set_rot(frame, knee,
                    xk - severity * 9.0 * t, yk, zk)


# ─────────────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
# NEW COMPOUND SYNDROMES 25–27 (v5)
# ══════════════════════════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# IMPAIRMENT 25: DIPLEGIC GAIT (compound)
# Spastic diplegia (bilateral CP) — most common CP motor pattern.
# Both legs affected symmetrically: scissor + crouch + equinus + slow cadence.
# Arms mildly affected (mild internal rotation posture).
# Ref: [10] Ounpuu 1991 — diplegic gait in premature birth survivors.
#      [2]  Chambers & Sutherland 2002 — diplegia compound kinematics.
# ─────────────────────────────────────────────────────────────────────────────
def apply_diplegic(frames, ft, severity):
    apply_scissor_gait(frames, severity * 0.85)
    apply_crouch_gait(frames, severity * 0.70)
    for side in ("right", "left"):
        apply_equinus_gait(frames, side, severity * 0.55)
        apply_arm_swing_reduction(frames, side, severity * 0.40)
    frames, ft = apply_cadence_reduction(frames, ft, severity * 0.35)
    return frames, ft


# ─────────────────────────────────────────────────────────────────────────────
# IMPAIRMENT 26: CEREBELLAR ATAXIA COMPOUND
# Full cerebellar syndrome: ataxia + wide base + trunk sway + titubation.
# Titubation = rhythmic 3 Hz head nodding tremor (clinical hallmark).
# Ref: [7]  Morton & Bastian 2006 — cerebellar contributions to locomotion.
#      [1]  Pirker 2017 — cerebellar ataxia: truncal titubation, wide base.
# ─────────────────────────────────────────────────────────────────────────────
def apply_cerebellar_ataxia_compound(frames, ft, severity, seed=None):
    apply_ataxic_gait(frames, severity, seed=seed)
    apply_wide_base(frames, severity * 0.55)

    # Titubation: rhythmic 3 Hz head + neck nodding oscillation
    titub_freq = 3.0   # Hz — clinical range 2–4 Hz
    titub_amp  = severity * 5.5

    for i, frame in enumerate(frames):
        t_sec = i * ft
        trem  = math.sin(2 * math.pi * titub_freq * t_sec) * titub_amp

        xhd, yhd, zhd = get_rot(frame, "C_head_bind_JNT")
        set_rot(frame, "C_head_bind_JNT",
                xhd + trem * 0.7, yhd, zhd + trem * 0.3)

        xn, yn, zn = get_rot(frame, "C_neck0001_bind_JNT")
        set_rot(frame, "C_neck0001_bind_JNT",
                xn + trem * 0.4, yn, zn + trem * 0.2)

    return frames, ft


# ─────────────────────────────────────────────────────────────────────────────
# IMPAIRMENT 27: MYOPATHIC GAIT COMPOUND
# Proximal hip girdle weakness — Duchenne MD, limb-girdle MD, polymyositis.
# Waddling + bilateral Trendelenburg + hyperlordosis + slowed cadence.
# Hyperlordosis: anterior pelvic tilt compensates for weak hip extensors.
# Ref: [12] Wren et al. 2005 — myopathic gait compound pattern.
#      [1]  Pirker 2017 — waddling gait, myopathic category.
# ─────────────────────────────────────────────────────────────────────────────
def apply_myopathic_compound(frames, ft, severity):
    apply_waddling_gait(frames, severity * 0.90)

    # Bilateral mild Trendelenburg (each side during its own stance)
    for side in ("right", "left"):
        apply_trunk_lean(frames, side, severity * 0.30, dynamic=True)
        apply_arm_swing_reduction(frames, side, severity * 0.45)

    # Hyperlordosis: lumbar spine extends backward (anterior pelvic tilt)
    # Spine X backward in lower segments
    for frame in frames:
        for j in ["C_spine0001_bind_JNT", "C_spine0002_bind_JNT"]:
            x, y, z = get_rot(frame, j)
            set_rot(frame, j, x - severity * 9.0 / 2.0, y, z)

        # Anterior pelvic tilt
        pxv, pyv, pzv = get_rot(frame, "C_pelvis0001_bind_JNT")
        set_rot(frame, "C_pelvis0001_bind_JNT",
                pxv + severity * 8.0, pyv, pzv)

    frames, ft = apply_cadence_reduction(frames, ft, severity * 0.40)
    return frames, ft


# ─────────────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────
def apply_impairment(input_bvh: str, state: dict, output_bvh: str,
                     seed: int = None) -> str:
    """
    Apply one or more impairments defined in `state` to `input_bvh`
    and write the result to `output_bvh`.

    Parameters
    ----------
    input_bvh  : path to source BVH file
    state      : dict mapping impairment key → severity (0.0–1.0)
    output_bvh : path for output BVH file
    seed       : optional integer seed for reproducible noise-based
                 impairments (ataxic, choreic, freezing).
                 None = non-deterministic (varies per run).

    State dict keys
    ---------------
    # ── Compound syndromes (apply first) ──────────────────────────────────
    hemiplegic_right / hemiplegic_left
    parkinsonian_shuffle
    crouch_gait
    scissor_gait
    antalgic_right / antalgic_left
    festinating_gait
    diplegic
    cerebellar_ataxia
    myopathic

    # ── Timing ────────────────────────────────────────────────────────────
    cadence_reduction
    freezing_of_gait

    # ── Noise-based (neurological) ────────────────────────────────────────
    ataxic_gait
    choreic_gait

    # ── Unilateral primitives ─────────────────────────────────────────────
    ankle_drop_right / ankle_drop_left
    knee_stiffness_right / knee_stiffness_left
    stride_asymmetry_right / stride_asymmetry_left
    arm_swing_reduction_right / arm_swing_reduction_left
    hip_hike_right / hip_hike_left
    trunk_lean_right / trunk_lean_left
    dystonic_right / dystonic_left
    equinus_right / equinus_left
    hip_extensor_weakness_right / hip_extensor_weakness_left
    leg_length_short_right / leg_length_short_left

    # ── Bilateral primitives ──────────────────────────────────────────────
    forward_lean
    wide_base
    sensory_ataxia
    waddling_gait
    """
    header, ft, frames = parse_bvh(input_bvh)
    frames = [list(f) for f in frames]

    print(f"[BVH v5] Input : {len(frames)} frames, {ft * 1000:.1f} ms/frame")
    print(f"[BVH v5] State : {list(state.keys())}")

    def _v(key):
        return float(state.get(key, 0))

    # ── 1. Compound syndromes (must run before primitives) ─────────────────
    for side in ("right", "left"):
        v = _v(f"hemiplegic_{side}")
        if v > 0:
            frames, ft = apply_hemiplegic(frames, ft, side, v)

    v = _v("parkinsonian_shuffle")
    if v > 0:
        frames, ft = apply_parkinsonian(frames, ft, v)

    v = _v("festinating_gait")
    if v > 0:
        frames, ft = apply_festinating_gait(frames, ft, v)

    v = _v("diplegic")
    if v > 0:
        frames, ft = apply_diplegic(frames, ft, v)

    v = _v("cerebellar_ataxia")
    if v > 0:
        frames, ft = apply_cerebellar_ataxia_compound(frames, ft, v, seed=seed)

    v = _v("myopathic")
    if v > 0:
        frames, ft = apply_myopathic_compound(frames, ft, v)

    v = _v("crouch_gait")
    if v > 0:
        apply_crouch_gait(frames, v)

    v = _v("scissor_gait")
    if v > 0:
        apply_scissor_gait(frames, v)

    for side in ("right", "left"):
        v = _v(f"antalgic_{side}")
        if v > 0:
            apply_antalgic_gait(frames, side, v)

    # ── 2. Timing modifiers ────────────────────────────────────────────────
    v = _v("cadence_reduction")
    if v > 0:
        frames, ft = apply_cadence_reduction(frames, ft, v)

    v = _v("freezing_of_gait")
    if v > 0:
        frames, ft = apply_freezing_of_gait(frames, ft, v, seed=seed)

    # ── 3. Noise-based neurological primitives ─────────────────────────────
    v = _v("ataxic_gait")
    if v > 0:
        apply_ataxic_gait(frames, v, seed=seed)

    v = _v("choreic_gait")
    if v > 0:
        apply_choreic_gait(frames, v, seed=seed)

    # ── 4. Unilateral primitives ───────────────────────────────────────────
    for side in ("right", "left"):
        v = _v(f"ankle_drop_{side}")
        if v > 0:
            apply_ankle_drop(frames, side, v)

        v = _v(f"knee_stiffness_{side}")
        if v > 0:
            apply_knee_stiffness(frames, side, v)

        v = _v(f"stride_asymmetry_{side}")
        if v > 0:
            apply_stride_asymmetry(frames, side, v)

        v = _v(f"arm_swing_reduction_{side}")
        if v > 0:
            apply_arm_swing_reduction(frames, side, v)

        v = _v(f"hip_hike_{side}")
        if v > 0:
            apply_hip_hike(frames, side, v)

        v = _v(f"trunk_lean_{side}")
        if v > 0:
            apply_trunk_lean(frames, side, v, dynamic=True)

        v = _v(f"dystonic_{side}")
        if v > 0:
            apply_dystonic_gait(frames, side, v)

        v = _v(f"equinus_{side}")
        if v > 0:
            apply_equinus_gait(frames, side, v)

        v = _v(f"hip_extensor_weakness_{side}")
        if v > 0:
            apply_hip_extensor_weakness(frames, side, v)

        v = _v(f"leg_length_short_{side}")
        if v > 0:
            apply_leg_length_discrepancy(frames, side, v)

    # ── 5. Bilateral primitives ────────────────────────────────────────────
    v = _v("forward_lean")
    if v > 0:
        apply_forward_lean(frames, v)

    v = _v("wide_base")
    if v > 0:
        apply_wide_base(frames, v)

    v = _v("sensory_ataxia")
    if v > 0:
        apply_sensory_ataxia(frames, v)

    v = _v("waddling_gait")
    if v > 0:
        apply_waddling_gait(frames, v)

    write_bvh(output_bvh, header, ft, frames)
    print(f"[BVH v5] Output: {output_bvh}  "
          f"({len(frames)} frames, {ft * 1000:.1f} ms/frame)")
    return output_bvh


# ─────────────────────────────────────────────────────────────────────────────
# CLI TEST SUITE
# Usage: python bvh_impairment_engine.py input.bvh
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    inp = sys.argv[1] if len(sys.argv) > 1 else "bvh_folder/bvh_0_out.bvh"

    tests = [
        # ── Original 14 ────────────────────────────────────────────────────
        ("stiff_knee_R",       {"knee_stiffness_right": 0.85,
                                "hip_hike_right": 0.70}),
        ("foot_drop_R",        {"ankle_drop_right": 0.85,
                                "stride_asymmetry_right": 0.60}),
        ("hemiplegic_R",       {"hemiplegic_right": 0.80}),
        ("parkinson",          {"parkinsonian_shuffle": 0.75}),
        ("crouch",             {"crouch_gait": 0.80}),
        ("trunk_lean_R",       {"trunk_lean_right": 0.80}),
        ("wide_base",          {"wide_base": 0.70, "forward_lean": 0.50}),
        ("scissor",            {"scissor_gait": 0.75}),
        ("antalgic_L",         {"antalgic_left": 0.70}),

        # ── New 13 (v5) ────────────────────────────────────────────────────
        ("festinating",        {"festinating_gait": 0.75}),
        ("ataxic",             {"ataxic_gait": 0.80}),
        ("freezing_of_gait",   {"freezing_of_gait": 0.70}),
        ("sensory_ataxia",     {"sensory_ataxia": 0.75}),
        ("choreic",            {"choreic_gait": 0.70}),
        ("dystonic_R",         {"dystonic_right": 0.80}),
        ("waddling",           {"waddling_gait": 0.75}),
        ("lld_short_R",        {"leg_length_short_right": 0.65}),
        ("hip_ext_weak_R",     {"hip_extensor_weakness_right": 0.75}),
        ("equinus_R",          {"equinus_right": 0.80}),
        ("diplegic",           {"diplegic": 0.75}),
        ("cerebellar_ataxia",  {"cerebellar_ataxia": 0.75}),
        ("myopathic",          {"myopathic": 0.70}),

        # ── Combined example ───────────────────────────────────────────────
        ("advanced_PD",        {"parkinsonian_shuffle": 0.70,
                                "festinating_gait": 0.50,
                                "freezing_of_gait": 0.60}),
    ]

    for name, state in tests:
        out = f"impaired_bvh_folder/test_{name}.bvh"
        apply_impairment(inp, state, out, seed=42)
        print()

# =============================================================================
# ══════════════════════════════════════════════════════════════════════════════
# PART 2 — CUSTOM JOINT OFFSET SYSTEM
# Additive per-joint angle offsets, applied AFTER gait impairments.
# Merged from bvh_custom_pose.py v2.
# ══════════════════════════════════════════════════════════════════════════════
# =============================================================================

import re
import os

# ─────────────────────────────────────────────────────────────────────────────
# JOINT MAP  —  alias → (BVH joint name, axis index)
# ─────────────────────────────────────────────────────────────────────────────
JOINT_MAP = {
    # Head & neck
    "head":                ("C_head_bind_JNT",         0),
    "head_nod":            ("C_head_bind_JNT",         0),
    "head_turn":           ("C_head_bind_JNT",         1),
    "head_tilt":           ("C_head_bind_JNT",         2),
    "neck":                ("C_neck0001_bind_JNT",     0),
    "neck_turn":           ("C_neck0001_bind_JNT",     1),
    "neck_tilt":           ("C_neck0001_bind_JNT",     2),
    "upper_neck":          ("C_neck0002_bind_JNT",     0),
    # Spine & trunk
    "spine_lower":         ("C_spine0001_bind_JNT",    0),
    "spine_mid":           ("C_spine0002_bind_JNT",    0),
    "spine_upper":         ("C_spine0003_bind_JNT",    0),
    "trunk_forward":       ("C_spine0002_bind_JNT",    0),
    "trunk_sideways":      ("C_spine0002_bind_JNT",    2),
    "trunk_rotate":        ("C_spine0002_bind_JNT",    1),
    # Pelvis
    "pelvis_tilt":         ("C_pelvis0001_bind_JNT",   0),
    "pelvis_side":         ("C_pelvis0001_bind_JNT",   2),
    "pelvis_rotate":       ("C_pelvis0001_bind_JNT",   1),
    # Left arm
    "left_shoulder":       ("L_armUpper0001_bind_JNT", 0),
    "left_shoulder_fwd":   ("L_armUpper0001_bind_JNT", 0),
    "left_shoulder_rot":   ("L_armUpper0001_bind_JNT", 1),
    "left_shoulder_side":  ("L_armUpper0001_bind_JNT", 2),
    "left_elbow":          ("L_armLower0001_bind_JNT", 0),
    "left_wrist":          ("L_hand0001_bind_JNT",     0),
    "left_wrist_turn":     ("L_hand0001_bind_JNT",     1),
    # Right arm
    "right_shoulder":      ("R_armUpper0001_bind_JNT", 0),
    "right_shoulder_fwd":  ("R_armUpper0001_bind_JNT", 0),
    "right_shoulder_rot":  ("R_armUpper0001_bind_JNT", 1),
    "right_shoulder_side": ("R_armUpper0001_bind_JNT", 2),
    "right_elbow":         ("R_armLower0001_bind_JNT", 0),
    "right_wrist":         ("R_hand0001_bind_JNT",     0),
    "right_wrist_turn":    ("R_hand0001_bind_JNT",     1),
    # Left leg
    "left_hip":            ("L_legUpper0001_bind_JNT", 0),
    "left_hip_fwd":        ("L_legUpper0001_bind_JNT", 0),
    "left_hip_rot":        ("L_legUpper0001_bind_JNT", 1),
    "left_hip_side":       ("L_legUpper0001_bind_JNT", 2),
    "left_knee":           ("L_legLower0001_bind_JNT", 0),
    "left_ankle":          ("L_foot0001_bind_JNT",     0),
    "left_foot_turn":      ("L_foot0001_bind_JNT",     1),
    "left_foot_splay":     ("L_foot0001_bind_JNT",     2),
    "left_toe":            ("L_foot0002_bind_JNT",     0),
    # Right leg
    "right_hip":           ("R_legUpper0001_bind_JNT", 0),
    "right_hip_fwd":       ("R_legUpper0001_bind_JNT", 0),
    "right_hip_rot":       ("R_legUpper0001_bind_JNT", 1),
    "right_hip_side":      ("R_legUpper0001_bind_JNT", 2),
    "right_knee":          ("R_legLower0001_bind_JNT", 0),
    "right_ankle":         ("R_foot0001_bind_JNT",     0),
    "right_foot_turn":     ("R_foot0001_bind_JNT",     1),
    "right_foot_splay":    ("R_foot0001_bind_JNT",     2),
    "right_toe":           ("R_foot0002_bind_JNT",     0),
}

# ─────────────────────────────────────────────────────────────────────────────
# MAGNITUDE TABLE
# ─────────────────────────────────────────────────────────────────────────────
MAGNITUDE = {
    "barely": 4.0, "very slightly": 5.0, "a touch": 6.0,
    "slightly": 8.0, "a little": 8.0, "a bit": 8.0,
    "mildly": 10.0, "somewhat": 12.0,
    "more": 20.0, "noticeably": 20.0, "moderately": 25.0, "quite": 25.0,
    "a lot": 35.0, "much": 35.0, "significantly": 35.0, "greatly": 40.0,
    "very much": 40.0, "extremely": 50.0,
    "fully": 80.0, "completely": 80.0,
}
DEFAULT_MAGNITUDE = 20.0

# ─────────────────────────────────────────────────────────────────────────────
# LLM PROMPT
# ─────────────────────────────────────────────────────────────────────────────
CUSTOM_OFFSET_PROMPT = """You are a 3D animation joint controller for a biomechanics BVH walking animation.

The user wants to SHIFT a body part by adding a fixed number of degrees on top of the existing motion.
All adjustments are additive offsets. The walk rhythm is preserved. Never use "override" — only offsets.

AVAILABLE JOINT KEYS:
HEAD/NECK: head, head_nod, head_turn, head_tilt, neck, neck_turn, neck_tilt, upper_neck
SPINE: spine_lower, spine_mid, spine_upper, trunk_forward, trunk_sideways, trunk_rotate
PELVIS: pelvis_tilt, pelvis_side, pelvis_rotate
LEFT ARM: left_shoulder, left_shoulder_fwd, left_shoulder_rot, left_shoulder_side, left_elbow, left_wrist, left_wrist_turn
RIGHT ARM: right_shoulder, right_shoulder_fwd, right_shoulder_rot, right_shoulder_side, right_elbow, right_wrist, right_wrist_turn
LEFT LEG: left_hip, left_hip_fwd, left_hip_rot, left_hip_side, left_knee, left_ankle, left_foot_splay, left_foot_turn, left_toe
RIGHT LEG: right_hip, right_hip_fwd, right_hip_rot, right_hip_side, right_knee, right_ankle, right_foot_splay, right_foot_turn, right_toe

SIGN CONVENTION:
  head/neck X: + bow forward,  - look up
  trunk_forward: + lean fwd,   - lean back
  trunk_sideways: + lean right, - lean left
  shoulder_fwd: + arm forward, - arm back
  shoulder_side: + arm across body (adduct), - arm out to side (abduct)
  shoulder_rot: + internal, - external
  elbow X: + straighten, - MORE BENT
  knee X:  + straighten, - MORE BENT
  hip_fwd: + leg forward, - leg back
  hip_side: + leg out (abduct), - leg in
  ankle X: + plantarflex, - dorsiflex
  foot_splay: + toes out, - toes in

MAGNITUDE (degrees):
  "slightly"/"a bit"/"a little" = 8
  "more"/(no qualifier) = 20
  "a lot"/"significantly" = 35
  "fully"/"completely" = 80
  Explicit numbers (e.g. "30 degrees") = use exact value

PHASE:
  "all" = every frame (default)
  "swing_l" / "swing_r" = only during that leg's swing phase
  "stance_l" / "stance_r" = only during that leg's stance phase
  Use non-"all" only if user explicitly says "during swing" / "while stepping" etc.

SPECIAL:
  "both knees/shoulders" etc → two entries (left + right)
  "reset X" → delta: 0.0 for that joint
  "reset all"/"clear everything" → {"offsets": [], "reset_all": true}

OUTPUT — ONLY valid JSON:
{
  "offsets": [
    {"joint_key": "left_knee", "delta": -20.0, "phase": "all", "label": "left knee more bent"}
  ],
  "reset_all": false
}

EXAMPLES:
"patient hands on chest"
→ {"offsets":[{"joint_key":"left_shoulder_fwd","delta":-20.0,"phase":"all","label":"L arm to chest"},{"joint_key":"left_shoulder_side","delta":15.0,"phase":"all","label":"L arm across"},{"joint_key":"left_elbow","delta":-70.0,"phase":"all","label":"L elbow bent"},{"joint_key":"right_shoulder_fwd","delta":-20.0,"phase":"all","label":"R arm to chest"},{"joint_key":"right_shoulder_side","delta":-15.0,"phase":"all","label":"R arm across"},{"joint_key":"right_elbow","delta":-70.0,"phase":"all","label":"R elbow bent"}],"reset_all":false}

"left knee more bent"
→ {"offsets":[{"joint_key":"left_knee","delta":-20.0,"phase":"all","label":"left knee more bent"}],"reset_all":false}

"head slightly down"
→ {"offsets":[{"joint_key":"head","delta":8.0,"phase":"all","label":"head bowed"},{"joint_key":"neck","delta":4.0,"phase":"all","label":"neck flexed"}],"reset_all":false}

"both elbows slightly more bent"
→ {"offsets":[{"joint_key":"left_elbow","delta":-8.0,"phase":"all","label":"L elbow bent"},{"joint_key":"right_elbow","delta":-8.0,"phase":"all","label":"R elbow bent"}],"reset_all":false}

"right knee bent more only when swinging"
→ {"offsets":[{"joint_key":"right_knee","delta":-20.0,"phase":"swing_r","label":"R knee bent during swing"}],"reset_all":false}

"reset left knee"
→ {"offsets":[{"joint_key":"left_knee","delta":0.0,"phase":"all","label":"reset left knee"}],"reset_all":false}

"reset everything"
→ {"offsets":[],"reset_all":true}

Now extract offsets for:
Input: "{prompt}"
Output:"""


# ─────────────────────────────────────────────────────────────────────────────
# RULE-BASED FALLBACK
# ─────────────────────────────────────────────────────────────────────────────

def _mag(t: str) -> float:
    m = re.search(r"(\d+)\s*(?:degrees?|°)", t)
    if m:
        return float(m.group(1))
    for phrase in sorted(MAGNITUDE, key=len, reverse=True):
        if phrase in t:
            return MAGNITUDE[phrase]
    return DEFAULT_MAGNITUDE


def _sides(t: str) -> list:
    if "both" in t:
        return ["left", "right"]
    out = []
    if "left"  in t: out.append("left")
    if "right" in t: out.append("right")
    return out


def _phase(t: str, side: str) -> str:
    if any(w in t for w in ["during swing", "while swinging", "when swinging",
                              "swing phase", "foot off", "in the air"]):
        return f"swing_{side[0]}"
    if any(w in t for w in ["during stance", "while standing",
                              "foot on", "on the ground", "stance phase"]):
        return f"stance_{side[0]}"
    return "all"


def _rule_parse(text: str) -> dict:
    t   = text.lower()
    mag = _mag(t)
    offsets = []

    # ── Reset ─────────────────────────────────────────────────────────────────
    if any(w in t for w in ["reset all", "clear all", "remove all",
                              "clear everything", "reset everything",
                              "start fresh", "remove everything"]):
        return {"offsets": [], "reset_all": True}

    # ── Arm out to the side (before shoulder/arm-chest blocks) ───────────────
    if "arm" in t and any(w in t for w in ["side", "out to", "abduct"]) and "chest" not in t:
        ss = _sides(t) or []
        for s in ss:
            sign = -1.0 if s == "right" else 1.0
            offsets.append({"joint_key": f"{s}_shoulder_side",
                             "delta": sign * mag, "phase": "all",
                             "label": f"{s} arm out to side"})
        if offsets:
            return {"offsets": offsets, "reset_all": False}

    # ── Hands / arms on chest ─────────────────────────────────────────────────
    if ("hand" in t or "arm" in t) and "chest" in t:
        for s in ["left", "right"]:
            sz = 1.0 if s == "left" else -1.0
            offsets += [
                {"joint_key": f"{s}_shoulder_fwd",  "delta": -20.0,      "phase": "all", "label": f"{s} arm to chest"},
                {"joint_key": f"{s}_shoulder_side", "delta":  sz * 15.0, "phase": "all", "label": f"{s} arm across body"},
                {"joint_key": f"{s}_elbow",         "delta": -70.0,      "phase": "all", "label": f"{s} elbow bent to chest"},
            ]
        return {"offsets": offsets, "reset_all": False}

    # ── Head & neck ───────────────────────────────────────────────────────────
    if "head" in t or "chin" in t:
        if any(w in t for w in ["down", "bow", "floor", "forward", "chest"]):
            offsets += [{"joint_key": "head", "delta":  mag,     "phase": "all", "label": "head bowed"},
                        {"joint_key": "neck", "delta":  mag*0.5, "phase": "all", "label": "neck flexed"}]
        elif any(w in t for w in ["up", "raise", "look up"]):
            offsets += [{"joint_key": "head", "delta": -mag,     "phase": "all", "label": "head raised"},
                        {"joint_key": "neck", "delta": -mag*0.5, "phase": "all", "label": "neck extended"}]
        elif "left" in t:
            offsets.append({"joint_key": "head_turn", "delta": -mag, "phase": "all", "label": "head turns left"})
        elif "right" in t:
            offsets.append({"joint_key": "head_turn", "delta":  mag, "phase": "all", "label": "head turns right"})

    # ── Trunk / spine ──────────────────────────────────────────────────────────
    if any(w in t for w in ["trunk", "torso", "spine", "back", "hunch", "stoop"]):
        if any(w in t for w in ["forward", "front", "lean", "hunch", "stoop", "down"]):
            for j, frac in [("spine_lower", 0.35), ("spine_mid", 0.35), ("spine_upper", 0.30)]:
                offsets.append({"joint_key": j, "delta": mag * frac, "phase": "all", "label": "trunk fwd"})
        elif any(w in t for w in ["back", "upright", "extend", "up"]):
            for j, frac in [("spine_lower", 0.35), ("spine_mid", 0.35), ("spine_upper", 0.30)]:
                offsets.append({"joint_key": j, "delta": -mag * frac, "phase": "all", "label": "trunk back"})
        elif "right" in t:
            offsets.append({"joint_key": "trunk_sideways", "delta":  mag, "phase": "all", "label": "trunk right"})
        elif "left" in t:
            offsets.append({"joint_key": "trunk_sideways", "delta": -mag, "phase": "all", "label": "trunk left"})

    # ── Shoulders ─────────────────────────────────────────────────────────────
    if "shoulder" in t:
        ss = _sides(t) or ["left", "right"]
        for s in ss:
            if any(w in t for w in ["hunch", "round", "forward", "front"]):
                offsets.append({"joint_key": f"{s}_shoulder_fwd", "delta": -mag * 0.5, "phase": "all", "label": f"{s} shoulder rounded"})
            elif any(w in t for w in ["raise", "up", "out", "side", "abduct"]):
                sign = -1.0 if s == "right" else 1.0
                offsets.append({"joint_key": f"{s}_shoulder_side", "delta": sign * mag, "phase": "all", "label": f"{s} shoulder out"})
            elif any(w in t for w in ["forward", "flex", "front"]):
                offsets.append({"joint_key": f"{s}_shoulder_fwd", "delta": mag, "phase": "all", "label": f"{s} shoulder fwd"})
            elif "internal" in t:
                offsets.append({"joint_key": f"{s}_shoulder_rot", "delta":  mag, "phase": "all", "label": f"{s} shoulder internal rot"})
            elif "external" in t:
                offsets.append({"joint_key": f"{s}_shoulder_rot", "delta": -mag, "phase": "all", "label": f"{s} shoulder external rot"})

    # ── Elbows ────────────────────────────────────────────────────────────────
    if "elbow" in t:
        ss = _sides(t) or ["left", "right"]
        for s in ss:
            ph = _phase(t, s)
            if "reset" in t or "remove" in t:
                offsets.append({"joint_key": f"{s}_elbow", "delta": 0.0,   "phase": "all", "label": f"reset {s} elbow"})
            elif any(w in t for w in ["bend", "flex", "more", "bent"]):
                offsets.append({"joint_key": f"{s}_elbow", "delta": -mag,  "phase": ph,   "label": f"{s} elbow more bent"})
            elif any(w in t for w in ["straight", "extend", "less"]):
                offsets.append({"joint_key": f"{s}_elbow", "delta":  mag * 0.5, "phase": ph, "label": f"{s} elbow straighter"})
            else:
                offsets.append({"joint_key": f"{s}_elbow", "delta": -mag,  "phase": ph,   "label": f"{s} elbow adjusted"})

    # ── Wrists ────────────────────────────────────────────────────────────────
    if "wrist" in t:
        ss = _sides(t) or ["left", "right"]
        for s in ss:
            if "flex" in t or "bend" in t:
                offsets.append({"joint_key": f"{s}_wrist", "delta": -mag, "phase": "all", "label": f"{s} wrist flexed"})
            elif "extend" in t:
                offsets.append({"joint_key": f"{s}_wrist", "delta":  mag, "phase": "all", "label": f"{s} wrist extended"})

    # ── Hips ──────────────────────────────────────────────────────────────────
    if "hip" in t:
        ss = _sides(t) or []
        for s in ss:
            ph = _phase(t, s)
            if any(w in t for w in ["abduct", "out", "side"]):
                sign = 1.0 if s == "left" else -1.0
                offsets.append({"joint_key": f"{s}_hip_side", "delta": sign * mag, "phase": ph, "label": f"{s} hip abducted"})
            elif any(w in t for w in ["flex", "forward", "bend"]):
                offsets.append({"joint_key": f"{s}_hip_fwd",  "delta":  mag, "phase": ph, "label": f"{s} hip flexed"})
            elif "internal" in t:
                offsets.append({"joint_key": f"{s}_hip_rot",  "delta":  mag, "phase": ph, "label": f"{s} hip internal rot"})
            elif "external" in t:
                offsets.append({"joint_key": f"{s}_hip_rot",  "delta": -mag, "phase": ph, "label": f"{s} hip external rot"})

    # ── Knees ─────────────────────────────────────────────────────────────────
    if "knee" in t:
        ss = _sides(t) or []
        for s in ss:
            ph = _phase(t, s)
            if "reset" in t or "remove" in t:
                offsets.append({"joint_key": f"{s}_knee", "delta": 0.0,       "phase": "all", "label": f"reset {s} knee"})
            elif any(w in t for w in ["bend", "flex", "more", "bent"]):
                offsets.append({"joint_key": f"{s}_knee", "delta": -mag,      "phase": ph,    "label": f"{s} knee more bent"})
            elif any(w in t for w in ["straight", "extend", "less"]):
                offsets.append({"joint_key": f"{s}_knee", "delta":  mag * 0.5,"phase": ph,    "label": f"{s} knee straighter"})
            else:
                offsets.append({"joint_key": f"{s}_knee", "delta": -mag,      "phase": ph,    "label": f"{s} knee adjusted"})

    # ── Ankles / feet ─────────────────────────────────────────────────────────
    if any(w in t for w in ["ankle", "foot", "feet", "toe"]):
        ss = _sides(t) or []
        for s in ss:
            ph = _phase(t, s)
            if any(w in t for w in ["out", "splay", "toes out", "turned out"]):
                sign = 1.0 if s == "right" else -1.0
                offsets.append({"joint_key": f"{s}_foot_splay", "delta": sign * mag * 0.5, "phase": ph, "label": f"{s} foot toes out"})
            elif any(w in t for w in ["in", "pigeon", "toes in", "turned in"]):
                sign = -1.0 if s == "right" else 1.0
                offsets.append({"joint_key": f"{s}_foot_splay", "delta": sign * mag * 0.5, "phase": ph, "label": f"{s} foot toes in"})
            elif any(w in t for w in ["plantarflex", "tiptoe", "point"]):
                offsets.append({"joint_key": f"{s}_ankle",       "delta":  mag, "phase": ph, "label": f"{s} ankle plantarflexed"})
            elif any(w in t for w in ["dorsiflex", "lift", "dorsi"]):
                offsets.append({"joint_key": f"{s}_ankle",       "delta": -mag, "phase": ph, "label": f"{s} ankle dorsiflexed"})

    return {"offsets": offsets, "reset_all": False}


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API — parse
# ─────────────────────────────────────────────────────────────────────────────

def parse_custom_offset(user_prompt: str, llm_caller=None) -> dict:
    """
    Parse a natural language body-part offset request.

    Parameters
    ----------
    user_prompt : str
    llm_caller  : callable(str) -> str, optional
        Routes to your Vertex/OpenAI/Ollama chain.
        If None, falls back to rule-based parser.

    Returns
    -------
    dict:
        "offsets"   : list of offset dicts
        "reset_all" : bool
        "labels"    : list of human-readable strings
    """
    import json as _json

    raw = None
    if llm_caller:
        full_prompt = CUSTOM_OFFSET_PROMPT.replace("{prompt}", user_prompt)
        try:
            raw = llm_caller(full_prompt)
        except Exception as e:
            print(f"[CustomOffset] LLM failed: {e}")

    offsets   = []
    reset_all = False

    if raw:
        m = re.search(r"\{.*\}", (raw or "").strip(), re.DOTALL)
        if m:
            try:
                data      = _json.loads(m.group(0))
                offsets   = data.get("offsets", [])
                reset_all = bool(data.get("reset_all", False))
            except Exception:
                pass

    if not offsets and not reset_all:
        print("[CustomOffset] Using rule-based fallback")
        parsed    = _rule_parse(user_prompt)
        offsets   = parsed["offsets"]
        reset_all = parsed["reset_all"]

    # Validate + clean
    valid = []
    for o in offsets:
        key = o.get("joint_key", "")
        if key not in JOINT_MAP:
            print(f"[CustomOffset] Unknown joint key '{key}' — skipped")
            continue
        valid.append({
            "joint_key": key,
            "delta":     float(o.get("delta", 0.0)),
            "phase":     o.get("phase", "all"),
            "label":     o.get("label", key),
        })

    return {
        "offsets":   valid,
        "reset_all": reset_all,
        "labels":    [o["label"] for o in valid],
    }


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API — apply
# ─────────────────────────────────────────────────────────────────────────────

def apply_custom_offset(frames: list, offsets: list) -> None:
    """
    Apply additive offsets to BVH frames in-place.
    Call this AFTER apply_impairment().

    Parameters
    ----------
    frames  : mutable list of frame lists (from parse_bvh)
    offsets : list of offset dicts from parse_custom_offset()["offsets"]
    """
    if not offsets:
        return

    resolved = []
    for o in offsets:
        key = o.get("joint_key", "")
        if key not in JOINT_MAP:
            continue
        bvh_joint, axis = JOINT_MAP[key]
        offset_idx = 3 if bvh_joint == "ROOT" else 0
        flat_idx   = P[bvh_joint] + offset_idx + axis
        resolved.append((flat_idx, float(o["delta"]), o.get("phase", "all")))

    for frame in frames:
        for flat_idx, delta, phase in resolved:
            if phase == "swing_r"  and not _is_swing(frame, "right"): continue
            if phase == "swing_l"  and not _is_swing(frame, "left"):  continue
            if phase == "stance_r" and     _is_swing(frame, "right"): continue
            if phase == "stance_l" and     _is_swing(frame, "left"):  continue
            frame[flat_idx] += delta


# ─────────────────────────────────────────────────────────────────────────────
# MERGE HELPER  (for session state in app_server)
# ─────────────────────────────────────────────────────────────────────────────

def merge_offsets(existing: list, new_offsets: list) -> list:
    """
    Merge new offsets into session list.
    Same joint_key + phase → replace. delta == 0.0 → remove (reset that joint).
    """
    result = list(existing)
    for new in new_offsets:
        key    = (new["joint_key"], new.get("phase", "all"))
        result = [e for e in result
                  if (e["joint_key"], e.get("phase", "all")) != key]
        if abs(new.get("delta", 0.0)) > 0.001:
            result.append(new)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# STANDALONE FILE UTILITY
# ─────────────────────────────────────────────────────────────────────────────

def apply_custom_offset_to_bvh(input_bvh: str, output_bvh: str, offsets: list) -> str:
    """Read BVH, apply offsets, write result."""
    header, ft, frames = parse_bvh(input_bvh)
    frames = [list(f) for f in frames]
    apply_custom_offset(frames, offsets)
    os.makedirs(os.path.dirname(output_bvh) or ".", exist_ok=True)
    write_bvh(output_bvh, header, ft, frames)
    print(f"[CustomOffset] {input_bvh} → {output_bvh} ({len(frames)} frames, {len(offsets)} offsets)")
    return output_bvh


# ─────────────────────────────────────────────────────────────────────────────
# SELF-TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        ("patient's hands on their chest",             6),
        ("left knee more bent",                        1),
        ("head looking slightly down",                 2),
        ("right arm slightly more to the side",        1),
        ("both elbows slightly more bent",             2),
        ("shoulders more hunched",                     3),
        ("left foot turned slightly outward",          1),
        ("trunk leans forward more",                   3),
        ("right knee more bent only when swinging",    1),
        ("reset left knee",                            1),
        ("reset everything",                           0),
        ("right hip slightly abducted",                1),
        ("head turns slightly to the left",            1),
        ("both feet slightly turned out",              2),
    ]

    print("=" * 70)
    print("CUSTOM OFFSET PARSER — rule-based self-test")
    print("=" * 70)
    all_pass = True
    for prompt, expected_count in tests:
        result = parse_custom_offset(prompt)
        got    = len(result["offsets"])
        reset  = result["reset_all"]
        ok     = (reset and expected_count == 0) or (not reset and got >= expected_count)
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"\n{status}  {prompt}")
        if reset:
            print("  → RESET ALL")
        elif not result["offsets"]:
            print("  → (no offsets)")
        else:
            for o in result["offsets"]:
                print(f"  joint={o['joint_key']:25s}  delta={o['delta']:+7.1f}°  "
                      f"phase={o['phase']:9s}  {o['label']}")
    print(f"\n{'ALL PASS' if all_pass else 'SOME FAILED'}")
# =============================================================================
# UNIFIED ENTRY POINT  v6
# Handles both clinical gait impairments AND custom joint offsets in one call.
# =============================================================================

def apply_all(
    input_bvh:       str,
    output_bvh:      str,
    impairment_state: dict  = None,
    custom_offsets:   list  = None,
    seed:             int   = None,
) -> str:
    """
    Unified apply function — clinical impairments THEN custom offsets.

    Parameters
    ----------
    input_bvh        : path to clean base BVH (always read from this)
    output_bvh       : path to write result
    impairment_state : dict  {key: severity 0-1}  — gait syndrome parameters
                       Pass {} or None to skip impairments.
    custom_offsets   : list  [{joint_key, delta, phase, label}, ...]
                       Additive joint offsets applied AFTER impairments.
                       Pass [] or None to skip.
    seed             : int for reproducible noise (ataxic, choreic, FOG).

    Returns
    -------
    output_bvh path
    """
    impairment_state = impairment_state or {}
    custom_offsets   = custom_offsets   or []

    print(f"[BVH v6] Input : {input_bvh}")
    print(f"[BVH v6] Impairments : {list(impairment_state.keys())}")
    print(f"[BVH v6] Custom offsets: {len(custom_offsets)}")

    # ── Step 1: apply gait impairments ────────────────────────────────────────
    if impairment_state:
        apply_impairment(input_bvh, impairment_state, output_bvh, seed=seed)
        working_bvh = output_bvh
    else:
        import shutil, os
        os.makedirs(os.path.dirname(output_bvh) or ".", exist_ok=True)
        shutil.copy(input_bvh, output_bvh)
        working_bvh = output_bvh

    # ── Step 2: apply custom joint offsets ────────────────────────────────────
    if custom_offsets:
        header, ft, frames = parse_bvh(working_bvh)
        frames = [list(f) for f in frames]
        apply_custom_offset(frames, custom_offsets)
        write_bvh(output_bvh, header, ft, frames)
        print(f"[BVH v6] Applied {len(custom_offsets)} custom offset(s)")

    print(f"[BVH v6] Output: {output_bvh}")
    return output_bvh


# ─── Convenience: backward-compatible wrapper ──────────────────────────────
def apply_impairment_with_offsets(
    input_bvh:       str,
    state:           dict,
    output_bvh:      str,
    custom_offsets:  list = None,
    seed:            int  = None,
) -> str:
    """Backward-compatible wrapper. Same as apply_all()."""
    return apply_all(input_bvh, output_bvh, state, custom_offsets, seed)


# =============================================================================
# CLI TEST SUITE
# Usage: python bvh_impairment_engine.py [input.bvh]
# =============================================================================
if __name__ == "__main__":
    import sys
    inp = sys.argv[1] if len(sys.argv) > 1 else "bvh_folder/bvh_0_out.bvh"

    tests = [
        # ── Gait impairments only ──────────────────────────────────────────
        ("stiff_knee_R",      {"knee_stiffness_right": 0.85, "hip_hike_right": 0.70}, []),
        ("foot_drop_R",       {"ankle_drop_right": 0.85, "stride_asymmetry_right": 0.60}, []),
        ("hemiplegic_R",      {"hemiplegic_right": 0.80}, []),
        ("parkinson",         {"parkinsonian_shuffle": 0.75}, []),
        ("parkinson_fog",     {"parkinsonian_shuffle": 0.70, "freezing_of_gait": 0.60}, []),
        ("crouch",            {"crouch_gait": 0.80}, []),
        ("ataxic",            {"ataxic_gait": 0.80}, []),
        ("cerebellar",        {"cerebellar_ataxia": 0.75}, []),
        ("myopathic",         {"myopathic": 0.70}, []),
        ("diplegic",          {"diplegic": 0.75}, []),
        ("choreic",           {"choreic_gait": 0.70}, []),
        ("sensory_ataxia",    {"sensory_ataxia": 0.75}, []),
        ("dystonic_R",        {"dystonic_right": 0.80}, []),
        ("equinus_R",         {"equinus_right": 0.80}, []),
        ("lld_short_R",       {"leg_length_short_right": 0.65}, []),
        # ── Custom offsets only ────────────────────────────────────────────
        ("hands_on_chest",    {}, [
            {"joint_key":"left_shoulder_fwd",   "delta":-20.0, "phase":"all", "label":"L arm to chest"},
            {"joint_key":"left_shoulder_side",  "delta": 15.0, "phase":"all", "label":"L arm across"},
            {"joint_key":"left_elbow",          "delta":-70.0, "phase":"all", "label":"L elbow bent"},
            {"joint_key":"right_shoulder_fwd",  "delta":-20.0, "phase":"all", "label":"R arm to chest"},
            {"joint_key":"right_shoulder_side", "delta":-15.0, "phase":"all", "label":"R arm across"},
            {"joint_key":"right_elbow",         "delta":-70.0, "phase":"all", "label":"R elbow bent"},
        ]),
        # ── Combined: impairment + custom offsets ─────────────────────────
        ("hemi_R_hands_chest", {"hemiplegic_right": 0.80}, [
            {"joint_key":"left_shoulder_fwd",  "delta":-20.0, "phase":"all", "label":"L arm to chest"},
            {"joint_key":"left_elbow",         "delta":-70.0, "phase":"all", "label":"L elbow bent"},
            {"joint_key":"right_elbow",        "delta":-70.0, "phase":"all", "label":"R elbow bent"},
        ]),
        ("parkinson_head_down", {"parkinsonian_shuffle": 0.75}, [
            {"joint_key":"head",  "delta":15.0, "phase":"all", "label":"head bowed"},
            {"joint_key":"neck",  "delta": 8.0, "phase":"all", "label":"neck flexed"},
            {"joint_key":"left_knee",  "delta":-15.0, "phase":"all", "label":"L knee more bent"},
            {"joint_key":"right_knee", "delta":-15.0, "phase":"all", "label":"R knee more bent"},
        ]),
    ]

    import os
    os.makedirs("impaired_bvh_folder", exist_ok=True)
    for name, imp_state, offsets in tests:
        out = f"impaired_bvh_folder/test_{name}.bvh"
        apply_all(inp, out, imp_state, offsets, seed=42)
        print()