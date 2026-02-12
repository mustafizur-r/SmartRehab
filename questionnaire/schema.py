# questionnaire/schema.py
from __future__ import annotations

from typing import Any, Dict, List


# ---------------------------------------------------------------------
# Ground truth (flat list, used for Q11..Q14 titles)
# ---------------------------------------------------------------------
GROUND_TRUTH_BY_ANIM: Dict[int, List[str]] = {
    1: [
        "Left leg: ankle dorsiflexion limitation",
        "Left arm: reduced arm swing",
        "Trunk: mild forward lean",
        "Gait phase: affected side Left, stance prolonged, swing shortened",
    ],
    2: [
        "GT item for anim2 - 1",
        "GT item for anim2 - 2",
        "GT item for anim2 - 3",
        "GT item for anim2 - 4",
    ],
    3: [
        "GT item for anim3 - 1",
        "GT item for anim3 - 2",
        "GT item for anim3 - 3",
        "GT item for anim3 - 4",
    ],
    4: [
        "GT item for anim4 - 1",
        "GT item for anim4 - 2",
        "GT item for anim4 - 3",
        "GT item for anim4 - 4",
    ],
    5: [
        "GT item for anim5 - 1",
        "GT item for anim5 - 2",
        "GT item for anim5 - 3",
        "GT item for anim5 - 4",
    ],
}


# ---------------------------------------------------------------------
# Ground truth (structured, used for GT canvas rendering like the picture)
# ---------------------------------------------------------------------
GROUND_TRUTH_STRUCTURED_BY_ANIM: Dict[int, Dict[str, Any]] = {
    1: {
        "title": "Reference Gait Description (Ground Truth)",
        "sections": [
            {
                "header": "Lower limbs",
                "items": [
                    "Left leg: ankle dorsiflexion limitation",
                    "Right leg: no impairment",
                ],
            },
            {
                "header": "Upper limbs",
                "items": [
                    "Left arm: reduced arm swing",
                    "Right arm: no impairment",
                ],
            },
            {
                "header": "Trunk",
                "items": [
                    "Mild forward trunk lean",
                ],
            },
            {
                "header": "Head/Shoulder",
                "items": [
                    "No abnormality",
                ],
            },
            {
                "header": "Gait phase",
                "items": [
                    "Affected side: Left",
                    "Stance phase: prolonged",
                    "Swing phase: shortened",
                ],
            },
        ],
    },
    2: {
        "title": "Reference Gait Description (Ground Truth)",
        "sections": [
            {"header": "Lower limbs", "items": ["GT lower limb item 1", "GT lower limb item 2"]},
            {"header": "Upper limbs", "items": ["GT upper limb item 1", "GT upper limb item 2"]},
            {"header": "Trunk", "items": ["GT trunk item 1"]},
            {"header": "Head/Shoulder", "items": ["GT head/shoulder item 1"]},
            {"header": "Gait phase", "items": ["GT gait phase item 1", "GT gait phase item 2", "GT gait phase item 3"]},
        ],
    },
    3: {
        "title": "Reference Gait Description (Ground Truth)",
        "sections": [
            {"header": "Lower limbs", "items": ["GT lower limb item 1", "GT lower limb item 2"]},
            {"header": "Upper limbs", "items": ["GT upper limb item 1", "GT upper limb item 2"]},
            {"header": "Trunk", "items": ["GT trunk item 1"]},
            {"header": "Head/Shoulder", "items": ["GT head/shoulder item 1"]},
            {"header": "Gait phase", "items": ["GT gait phase item 1", "GT gait phase item 2", "GT gait phase item 3"]},
        ],
    },
    4: {
        "title": "Reference Gait Description (Ground Truth)",
        "sections": [
            {"header": "Lower limbs", "items": ["GT lower limb item 1", "GT lower limb item 2"]},
            {"header": "Upper limbs", "items": ["GT upper limb item 1", "GT upper limb item 2"]},
            {"header": "Trunk", "items": ["GT trunk item 1"]},
            {"header": "Head/Shoulder", "items": ["GT head/shoulder item 1"]},
            {"header": "Gait phase", "items": ["GT gait phase item 1", "GT gait phase item 2", "GT gait phase item 3"]},
        ],
    },
    5: {
        "title": "Reference Gait Description (Ground Truth)",
        "sections": [
            {"header": "Lower limbs", "items": ["GT lower limb item 1", "GT lower limb item 2"]},
            {"header": "Upper limbs", "items": ["GT upper limb item 1", "GT upper limb item 2"]},
            {"header": "Trunk", "items": ["GT trunk item 1"]},
            {"header": "Head/Shoulder", "items": ["GT head/shoulder item 1"]},
            {"header": "Gait phase", "items": ["GT gait phase item 1", "GT gait phase item 2", "GT gait phase item 3"]},
        ],
    },
}


def _clamp_anim_index(animation_index: int) -> int:
    if animation_index < 1:
        return 1
    if animation_index > 5:
        return 5
    return animation_index


def get_ground_truth_items(animation_index: int) -> List[str]:
    anim = _clamp_anim_index(animation_index)
    items = list(GROUND_TRUTH_BY_ANIM.get(anim, []))
    while len(items) < 4:
        items.append(f"Ground-truth feature {len(items) + 1}")
    return items[:4]


def get_ground_truth_structured(animation_index: int) -> Dict[str, Any]:
    anim = _clamp_anim_index(animation_index)
    gt = GROUND_TRUTH_STRUCTURED_BY_ANIM.get(anim)

    if not gt:
        return {
            "animation_index": anim,
            "title": "Reference Gait Description (Ground Truth)",
            "sections": [],
        }

    return {
        "animation_index": anim,
        "title": gt.get("title", "Reference Gait Description (Ground Truth)"),
        "sections": gt.get("sections", []),
    }


def get_phase1_schema(animation_index: int = 1) -> Dict[str, Any]:
    yes_partially_no = ["Yes", "Partially", "No"]
    locations_lower = ["Hip", "Knee", "Ankle", "Foot", "Toes"]
    locations_upper = ["Shoulder", "Elbow", "Wrist", "Hand"]
    confidence = [1, 2, 3, 4, 5]
    rating_1_5 = [1, 2, 3, 4, 5]

    gt_visibility = ["Clearly visible", "Partially visible", "Not visible"]

    gt_items = get_ground_truth_items(animation_index)
    gt_structured = get_ground_truth_structured(animation_index)

    return {
        "phase": 1,
        "animation_index": _clamp_anim_index(animation_index),

        # keep flat list for Q11..Q14 titles (backward compatible)
        "ground_truth_items": gt_items,

        # new structured GT block for canvas UI rendering
        "ground_truth_structured": gt_structured,

        "questions": [
            {
                "question_id": "P1_Q1",
                "title": "Left leg impairment present?",
                "type": "triple_with_location_confidence",
                "choice_options": yes_partially_no,
                "location_options": locations_lower,
                "confidence_options": confidence,
                "answer_shape": {
                    "choice": "string",
                    "locations": "string[]",
                    "confidence": "int(1..5)",
                },
            },
            {
                "question_id": "P1_Q2",
                "title": "Right leg impairment present?",
                "type": "triple_with_location_confidence",
                "choice_options": yes_partially_no,
                "location_options": locations_lower,
                "confidence_options": confidence,
                "answer_shape": {
                    "choice": "string",
                    "locations": "string[]",
                    "confidence": "int(1..5)",
                },
            },
            {
                "question_id": "P1_Q3",
                "title": "Left arm impairment present?",
                "type": "triple_with_location_confidence",
                "choice_options": yes_partially_no,
                "location_options": locations_upper,
                "confidence_options": confidence,
                "answer_shape": {
                    "choice": "string",
                    "locations": "string[]",
                    "confidence": "int(1..5)",
                },
            },
            {
                "question_id": "P1_Q4",
                "title": "Right arm impairment present?",
                "type": "triple_with_location_confidence",
                "choice_options": yes_partially_no,
                "location_options": locations_upper,
                "confidence_options": confidence,
                "answer_shape": {
                    "choice": "string",
                    "locations": "string[]",
                    "confidence": "int(1..5)",
                },
            },
            {
                "question_id": "P1_Q5",
                "title": "Trunk/posture abnormal?",
                "type": "triple_with_location_confidence",
                "choice_options": yes_partially_no,
                "location_options": ["Forward lean", "Backward lean", "Left lean", "Right lean", "Rotation"],
                "confidence_options": confidence,
                "answer_shape": {
                    "choice": "string",
                    "locations": "string[]",
                    "confidence": "int(1..5)",
                },
            },
            {
                "question_id": "P1_Q6",
                "title": "Head/shoulder abnormal?",
                "type": "triple_with_location_confidence",
                "choice_options": yes_partially_no,
                "location_options": ["Asymmetry", "Limited motion", "Compensation"],
                "confidence_options": confidence,
                "answer_shape": {
                    "choice": "string",
                    "locations": "string[]",
                    "confidence": "int(1..5)",
                },
            },
            {
                "question_id": "P1_Q7",
                "title": "Which side shows abnormal stance-swing timing?",
                "type": "single_choice",
                "options": ["Left", "Right", "Both", "None", "Unsure"],
                "answer_shape": {"value": "string"},
            },
            {
                "question_id": "P1_Q8",
                "title": "What is the abnormal pattern? (choose all that apply)",
                "type": "multi_choice",
                "options": [
                    "Short stance",
                    "Long stance",
                    "Short swing",
                    "Long swing",
                    "Dragging",
                ],
                "answer_shape": {"values": "string[]"},
            },
            {
                "question_id": "P1_Q9",
                "title": "Naturalness of the impairment (1-5)",
                "type": "rating_1_5",
                "options": rating_1_5,
                "answer_shape": {"value": "int(1..5)"},
            },
            {
                "question_id": "P1_Q10",
                "title": "Motion smoothness/stability (1-5)",
                "type": "rating_1_5",
                "options": rating_1_5,
                "answer_shape": {"value": "int(1..5)"},
            },
            {
                "question_id": "P1_Q11",
                "title": f"GT-1. {gt_items[0]} - Is this feature visible?",
                "type": "single_choice",
                "options": gt_visibility,
                "answer_shape": {"value": "string"},
            },
            {
                "question_id": "P1_Q12",
                "title": f"GT-2. {gt_items[1]} - Is this feature visible?",
                "type": "single_choice",
                "options": gt_visibility,
                "answer_shape": {"value": "string"},
            },
            {
                "question_id": "P1_Q13",
                "title": f"GT-3. {gt_items[2]} - Is this feature visible?",
                "type": "single_choice",
                "options": gt_visibility,
                "answer_shape": {"value": "string"},
            },
            {
                "question_id": "P1_Q14",
                "title": f"GT-4. {gt_items[3]} - Is this feature visible?",
                "type": "single_choice",
                "options": gt_visibility,
                "answer_shape": {"value": "string"},
            },
            {
                "question_id": "P1_Q15",
                "title": "Confidence in your judgments for this ground-truth section (1-5)",
                "type": "rating_1_5",
                "options": rating_1_5,
                "answer_shape": {"value": "int(1..5)"},
            },
        ],
    }


def get_phase2_schema() -> Dict[str, Any]:
    return {
        "phase": 2,
        "questions": [
            {
                "question_id": "P2_Q1",
                "title": "Which animation better represents the intended gait impairment?",
                "type": "single_choice",
                "options": ["Left", "Right", "No clear difference"],
                "answer_shape": {"value": "string"},
            },
            {
                "question_id": "P2_Q2",
                "title": "Which animation appears more natural and realistic?",
                "type": "single_choice",
                "options": ["Left", "Right", "No clear difference"],
                "answer_shape": {"value": "string"},
            },
        ],
    }


def get_schema_by_phase(phase: int, animation_index: int = 1) -> Dict[str, Any]:
    if phase == 1:
        return get_phase1_schema(animation_index=animation_index)
    if phase == 2:
        return get_phase2_schema()
    raise ValueError("phase must be 1 or 2")
