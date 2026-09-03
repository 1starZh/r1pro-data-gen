"""Task-independent completeness checks for natural-language GoalSpecs.

GoalSpec parsing verifies shape and grounding, but a syntactically valid spec
can still omit an explicit completion clause from the task description. This
module protects the freeze boundary from that under-specification without
encoding task names, action recipes, or evaluator implementations. It only
checks semantic obligations stated in the instruction and binds them to the
entities already present in the candidate GoalSpec.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from r1pro_data_gen.domain import GoalSpec, ObjectCapability, SceneModel


_RELEASE_RE = re.compile(
    r"\b(?:release|released|releases|releasing|let\s+go|drop|dropped|drops)\b"
    r"|释放|松手|放手|放开"
)
_SETTLE_RE = re.compile(
    r"\b(?:settle|settled|settles|settling|stable|stabilized|rest|resting|"
    r"stop|stopped|leave\s+(?:it|the\s+object)\s+there)\b"
    r"|稳定|静止|停下|放稳|落稳"
)
_GRASP_RE = re.compile(
    r"\b(?:grasp|grasped|grasps|grasping|pick|picked|picks|picking|"
    r"carry|carried|carries|carrying|hold|held|holds|holding)\b"
    r"|抓取|抓住|拿起|夹取|搬运|拿到|取出"
)
_LIFT_RE = re.compile(
    r"\b(?:lift|lifted|lifts|lifting|raise|raised|raises|raising|above)\b"
    r"|抬起|举起|提升|离开桌面|离地"
)
_NO_GRASP_RE = re.compile(
    r"\b(?:without|no|never|do\s+not|don't)\s+(?:physically\s+)?grasp"
    r"|不抓取|不抓|无需抓取|不用抓"
)
_CONTACT_RE = re.compile(
    r"\b(?:contact|touch|touching|press|pressed|presses|pressing)\b"
    r"|physical\s+contact|接触|触碰|按压"
)
_PLACEMENT_RE = re.compile(
    r"\b(?:inside|into|within|place|placed|places|placing|put|puts|putting|"
    r"move|moved|moves|moving|transfer|transferred|carry|carried|push|pushed|"
    r"pushes|pushing)\b"
    r"|放置|放到|放入|移到|移动|搬到|推送|推到"
)
_EXPLICIT_INSIDE_RE = re.compile(
    r"\b(?:inside|into|within|fully\s+inside)\b|完全在|进入|放入|到.*区域|到.*region"
)
_SUPPORT_RE = re.compile(
    r"\b(?:on|onto)\s+(?:the\s+)?(?:table|support|surface)\b"
    r"|\b(?:tabletop|supporting\s+surface)\b"
    r"|桌面|桌上|支撑面|台面"
)

_PLACEMENT_PREDICATES = frozenset(
    {"inside_region", "on_support", "object_at_pose"}
)
_TASK_SUBJECT_PREDICATES = _PLACEMENT_PREDICATES | frozenset(
    {"attached", "released", "settled", "lifted"}
)


def goal_spec_completeness_errors(
    spec: GoalSpec,
    instruction: str,
    scene: SceneModel,
) -> tuple[str, ...]:
    """Return explicit instruction obligations missing from ``spec``.

    The function deliberately does not invent a target relation. If the
    planner omitted the placement predicate itself, it reports that error so
    bounded planner repair can ground the relation from scene facts.
    """
    if not isinstance(spec, GoalSpec):
        raise TypeError("spec must be a GoalSpec")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("instruction must be a non-empty string")
    if not isinstance(scene, SceneModel):
        raise TypeError("scene must be a SceneModel")

    text = instruction.casefold()
    required = tuple(spec.required)
    placement_subjects = _subjects_for(
        required, _PLACEMENT_PREDICATES, spec, scene
    )
    task_subjects = _subjects_for(
        required, _TASK_SUBJECT_PREDICATES, spec, scene
    )
    inside_subjects = _subjects_for(required, {"inside_region"}, spec, scene)
    mentioned_movable = _mentioned_movable_objects(text, scene)
    errors: list[str] = []

    placement_requested = bool(_PLACEMENT_RE.search(text))
    inside_requested = bool(_EXPLICIT_INSIDE_RE.search(text))
    if placement_requested and not placement_subjects:
        errors.append(
            "instruction describes moving or placing an entity but GoalSpec "
            "has no terminal placement predicate"
        )
    if inside_requested and not inside_subjects:
        errors.append(
            "instruction requires an entity inside a region but GoalSpec has "
            "no inside_region predicate"
        )

    # When the instruction explicitly names multiple movable objects or says
    # every/all/each, ensure the candidate did not silently cover only a
    # subset. Object names come from SceneModel; no task-specific mappings are
    # embedded here.
    broad_coverage = bool(
        len(mentioned_movable) > 1
        or re.search(r"\b(?:all|every|each|both|three|four|five)\b", text)
        or re.search(r"所有|每个|全部|三[个件]|四[个件]|五[个件]", text)
    )
    if broad_coverage:
        missing = sorted(mentioned_movable - set(placement_subjects))
        if missing:
            errors.append(
                "GoalSpec does not cover every explicitly mentioned movable "
                f"entity in a terminal placement relation: {', '.join(missing)}"
            )

    # A named movable object must not be replaced by an unrelated bound object,
    # even when the instruction mentions only one object.  This is still
    # grounding-only: names and aliases come from the current SceneModel.
    action_requested = bool(
        placement_requested or _GRASP_RE.search(text) or _LIFT_RE.search(text)
    )
    if mentioned_movable and action_requested:
        missing = sorted(mentioned_movable - set(task_subjects))
        if missing:
            errors.append(
                "GoalSpec does not cover the explicitly mentioned movable "
                f"entity: {', '.join(missing)}"
            )

    release_requested = bool(_RELEASE_RE.search(text))
    settle_requested = bool(_SETTLE_RE.search(text))
    grasp_requested = bool(_GRASP_RE.search(text)) and not bool(_NO_GRASP_RE.search(text))

    if release_requested:
        errors.extend(
            _missing_subjects(
                task_subjects,
                required,
                "released",
                spec,
                scene,
                message="instruction explicitly requires release",
            )
        )
    if settle_requested:
        errors.extend(
            _missing_subjects(
                task_subjects,
                required,
                "settled",
                spec,
                scene,
                message="instruction explicitly requires the entity to settle",
            )
        )
    if grasp_requested:
        errors.extend(
            _missing_subjects(
                task_subjects,
                required,
                "attached",
                spec,
                scene,
                message="instruction explicitly requires grasp/carry evidence",
            )
        )

    if _LIFT_RE.search(text):
        errors.extend(
            _missing_subjects(
                task_subjects,
                required,
                "lifted",
                spec,
                scene,
                message="instruction explicitly requires lifting evidence",
            )
        )

    if _CONTACT_RE.search(text):
        contact_entities = _entities_in_contact_predicates(required, spec, scene)
        if not contact_entities:
            errors.append(
                "instruction explicitly requires physical contact but GoalSpec "
                "has no contact predicate"
            )
        else:
            missing = sorted(set(mentioned_movable) - contact_entities)
            if missing:
                errors.append(
                    "GoalSpec contact evidence does not involve the explicitly "
                    f"mentioned movable entity: {', '.join(missing)}"
                )

    if _SUPPORT_RE.search(text) and inside_subjects:
        support_subjects = _subjects_for(required, {"on_support"}, spec, scene)
        missing_support = sorted(set(inside_subjects) - set(support_subjects))
        if missing_support:
            errors.append(
                "instruction explicitly requires placement on a support surface "
                f"but GoalSpec lacks on_support for: {', '.join(missing_support)}"
            )

    return tuple(dict.fromkeys(errors))


def _subjects_for(
    predicates: Iterable[Any],
    names: set[str] | frozenset[str],
    spec: GoalSpec,
    scene: SceneModel,
) -> tuple[str, ...]:
    subjects: list[str] = []
    for predicate in predicates:
        if predicate.predicate not in names:
            continue
        raw = predicate.arguments.get("subject")
        resolved = _resolve_entity(raw, spec, scene)
        if resolved is not None and resolved not in subjects:
            subjects.append(resolved)
    return tuple(subjects)


def _missing_subjects(
    source_subjects: Iterable[str],
    predicates: Iterable[Any],
    predicate_name: str,
    spec: GoalSpec,
    scene: SceneModel,
    *,
    message: str,
) -> list[str]:
    existing = set(_subjects_for(predicates, {predicate_name}, spec, scene))
    missing = sorted(set(source_subjects) - existing)
    return [
        f"{message} but GoalSpec lacks {predicate_name} for: {', '.join(missing)}"
    ] if missing else []


def _resolve_entity(value: object, spec: GoalSpec, scene: SceneModel) -> str | None:
    if not isinstance(value, str):
        return None
    root = value.split(".", 1)[0]
    reference = spec.bindings.get(root)
    if reference is not None:
        name = reference.removeprefix("scene://")
    elif value.startswith("scene://"):
        name = value.removeprefix("scene://")
    else:
        name = root
    try:
        scene.object(name)
    except KeyError:
        return None
    return name


def _entities_in_contact_predicates(
    predicates: Iterable[Any],
    spec: GoalSpec,
    scene: SceneModel,
) -> set[str]:
    entities: set[str] = set()
    for predicate in predicates:
        if predicate.predicate != "contact":
            continue
        for key in ("entity_a", "entity_b"):
            resolved = _resolve_entity(predicate.arguments.get(key), spec, scene)
            if resolved is not None:
                entities.add(resolved)
    return entities


def _mentioned_movable_objects(text: str, scene: SceneModel) -> set[str]:
    names: set[str] = set()
    for obj in scene.objects:
        if ObjectCapability.MOVABLE not in set(obj.capabilities):
            continue
        candidates = (obj.name, *obj.aliases)
        if any(
            re.search(
                rf"(?<![a-z0-9_]){re.escape(candidate.casefold())}(?![a-z0-9_])",
                text,
            )
            for candidate in candidates
        ):
            names.add(obj.name)
    return names


__all__ = ["goal_spec_completeness_errors"]
