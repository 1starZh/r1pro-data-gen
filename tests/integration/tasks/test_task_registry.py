"""Public TaskSpec catalog and task identity contracts."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from tests.support import PROJECT_ROOT
from r1pro_data_gen.data.scenes import load_scene_data
from r1pro_data_gen.tasks import TaskCatalog, load_task_spec, write_task_spec


def test_public_catalog_contains_all_benchmark_tasks() -> None:
    catalog = TaskCatalog()
    ids = set(catalog.ids())
    assert {
        "navigation.arena_route",
        "pickplace.tabletop",
        "pickplace.tabletop_complete",
        "pickplace.floor_to_table_complete",
        "pickplace.holdout_prism_on_slate",
        "pickplace.holdout_floor_to_table",
        "push.box_to_region",
        "push.box_to_region_complete",
        "rearrangement.three_objects",
        "rearrangement.three_objects_complete",
    } <= ids


def test_task_id_resolves_scene_and_instruction() -> None:
    spec = load_task_spec("pickplace.tabletop_complete")
    assert spec.id == "pickplace.tabletop_complete"
    assert spec.family == "pickplace"
    assert spec.scene["name"] == "tabletop_cylinder_manipulation"
    assert spec.scene_human_verified is True
    assert load_scene_data(spec.scene).name == "tabletop_cylinder_manipulation"
    assert "grasp" in spec.instruction
    assert spec.source_path is not None


def test_human_verified_catalog_includes_checked_tabletop_scenes() -> None:
    verified = {
        spec.id for spec in TaskCatalog().load_all() if spec.scene_human_verified
    }
    assert verified == {"pickplace.tabletop", "pickplace.tabletop_complete"}


def test_task_spec_round_trip_keeps_embedded_scene_and_verification_state(
    tmp_path: Path,
) -> None:
    source = load_task_spec("pickplace.tabletop")
    generated = tmp_path / "derived_task.yaml"
    write_task_spec(
        source,
        generated,
        scene_human_verified=False,
    )
    loaded = load_task_spec(generated)
    assert loaded.schema_version == "task_spec.v2"
    assert loaded.scene == source.scene
    assert loaded.scene_human_verified is False
    with pytest.raises(ValueError, match="not human-verified"):
        loaded.require_human_verified()


def test_task_spec_rejects_benchmark_fields(tmp_path: Path) -> None:
    scene = tmp_path / "scene.yaml"
    scene.write_text("name: scene\n", encoding="utf-8")
    task = tmp_path / "task.yaml"
    task.write_text(
        "\n".join(
            (
                "schema_version: task_spec.v2",
                "id: test.invalid",
                "family: test",
                "scene_human_verified: false",
                "scene: {name: scene}",
                "instruction: Do the task.",
                "randomization: {}",
                "",
            )
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown fields"):
        load_task_spec(task)


def test_task_spec_rejects_external_instruction_file(tmp_path: Path) -> None:
    scene = tmp_path / "scene.yaml"
    scene.write_text("name: scene\n", encoding="utf-8")
    task = tmp_path / "task.yaml"
    task.write_text(
        "\n".join(
            (
                "schema_version: task_spec.v2",
                "id: test.invalid",
                "family: test",
                "scene_human_verified: false",
                "scene: {name: scene}",
                "instruction_file: goal.txt",
                "",
            )
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown fields|missing fields"):
        load_task_spec(task)


def test_task_spec_rejects_external_scene_reference(tmp_path: Path) -> None:
    task = tmp_path / "task.yaml"
    task.write_text(
        "\n".join(
            (
                "schema_version: task_spec.v2",
                "id: test.invalid",
                "family: test",
                "scene_human_verified: false",
                "scene: scene.yaml",
                "instruction: Do the task.",
                "tags: []",
                "",
            )
        ),
        encoding="utf-8",
    )
    with pytest.raises(TypeError, match="scene.*mapping"):
        load_task_spec(task)


def test_task_spec_requires_a_strict_boolean_verification_field(tmp_path: Path) -> None:
    task = tmp_path / "task.yaml"
    task.write_text(
        "\n".join(
            (
                "schema_version: task_spec.v2",
                "id: test.invalid",
                "family: test",
                "scene_human_verified: 1",
                "scene: {name: scene}",
                "instruction: Do the task.",
                "tags: []",
                "",
            )
        ),
        encoding="utf-8",
    )
    with pytest.raises(TypeError, match="scene_human_verified.*bool"):
        load_task_spec(task)


@pytest.mark.parametrize(
    "suite_name",
    ("first_four_families.yaml", "complete_task_episodes.yaml"),
)
def test_benchmark_cases_resolve_only_public_task_specs(suite_name: str) -> None:
    suite_path = PROJECT_ROOT / "benchmarks" / suite_name
    payload = yaml.safe_load(suite_path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    for family_group in ("families", "holdout_families"):
        for family in payload.get(family_group, []):
            for case in family["cases"]:
                assert "task" in case
                assert not {
                    "scene",
                    "instruction",
                    "instruction_file",
                    "task_module",
                } & set(case)
                spec = load_task_spec(case["task"])
                assert spec.source_path is not None
