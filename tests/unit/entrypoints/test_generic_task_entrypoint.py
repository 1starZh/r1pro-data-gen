from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

from r1pro_data_gen.domain import ObjectModel, ObjectType, RobotModel, SceneModel, WorldModel
from r1pro_data_gen.planning.llm.providers.protocol import ProviderResponse
from tests.support import PROJECT_ROOT

SCRIPT = PROJECT_ROOT / "scripts" / "tasks" / "run_task.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("run_task_product_helpers", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_task_product_helpers"] = module
    spec.loader.exec_module(module)
    return module


class _FakeProvider:
    name = "fake"
    model = "fake"

    def complete(self, *, system: str, user: str) -> ProviderResponse:
        return ProviderResponse(
            text=json.dumps(
                {
                    "schema_version": 1,
                    "bindings": {"subject": "scene://item"},
                    "required": [
                        {"predicate": "settled", "arguments": {"subject": "subject"}}
                    ],
                    "invariants": [],
                }
            ),
            provider=self.name,
            model=self.model,
        )


def _scene() -> SceneModel:
    return SceneModel(
        name="scene",
        world=WorldModel(),
        robot=RobotModel(asset="robot.usd"),
        objects=(
            ObjectModel(
                name="item",
                type=ObjectType.CUBOID,
                pos=(0.0, 0.0, 0.1),
                size=(0.1, 0.1, 0.1),
            ),
        ),
    )


def _task(tmp_path: Path, instruction: str) -> Path:
    scene_path = tmp_path / "scene.yaml"
    scene_path.write_text("name: scene\n", encoding="utf-8")
    task_path = tmp_path / "task.yaml"
    task_path.write_text(
        "\n".join(
            (
                "schema_version: task_spec.v2",
                "id: test.scene_task",
                "family: test",
                "scene_human_verified: true",
                "scene:",
                "  name: scene",
                "  robot:",
                "    asset: robot.usd",
                "  world: {}",
                "  objects: []",
                f"instruction: {instruction!r}",
                "tags: [test]",
                "",
            )
        ),
        encoding="utf-8",
    )
    return task_path


def test_product_entrypoint_freezes_goal_spec_without_task_module(tmp_path):
    module = _load_module()
    goal_path = tmp_path / "goal_spec.json"

    module.freeze_goal_spec(
        scene=_scene(),
        instruction="Make the item stable.",
        output_path=goal_path,
        provider=_FakeProvider(),
    )

    payload = json.loads(goal_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["bindings"] == {"subject": "scene://item"}
    assert "skill" not in payload
    assert "task_module" not in payload


def test_product_entrypoint_preserves_frozen_hash_on_loop_failure(tmp_path, monkeypatch):
    module = _load_module()
    scene_path = tmp_path / "scene.yaml"
    scene_path.write_text("name: scene\n", encoding="utf-8")
    urdf = tmp_path / "robot.urdf"
    urdf.write_text("placeholder", encoding="utf-8")
    output_dir = tmp_path / "output"
    frozen_hash = "a" * 64

    def fake_freeze_goal_spec(*, scene, instruction, output_path, provider=None):
        module.write_json(output_path, {"schema_version": 1})
        module.write_json(
            output_path.with_name(output_path.name + ".provenance.json"),
            {"goal_spec_hash": frozen_hash},
        )
        return object()

    monkeypatch.setattr(module, "freeze_goal_spec", fake_freeze_goal_spec)
    task_path = _task(tmp_path, "Release the item.")
    monkeypatch.setattr(module, "load_scene_data", lambda _data, **_: _scene())
    def fail_episode(**_kwargs):
        raise RuntimeError("planner failed")

    monkeypatch.setattr(module, "execute_product_episode", fail_episode)
    monkeypatch.setattr(
        module,
        "_probe_gpu_health",
        lambda _physical_gpu_id: {"healthy": True, "physical_gpu_id": 6},
    )

    monkeypatch.setattr(sys, "argv", [
        "run_task.py",
        "--task", str(task_path),
        "--urdf", str(urdf),
        "--output-dir", str(output_dir),
    ])
    assert module.main() == 1

    result = json.loads((output_dir / "result.json").read_text(encoding="utf-8"))
    assert result["result"] == "failed"
    assert result["evaluation_mode"] == "goal_spec"
    assert result["goal_spec_hash"] == frozen_hash
    assert result["evaluation"]["status"] == "failed"
    assert result["evaluation"]["evidence_complete"] is False
    assert result["evaluation"]["collision_observation_complete"] is False

    loop_result = json.loads((output_dir / "loop_result.json").read_text(encoding="utf-8"))
    assert loop_result["status"] == "failed"
    assert loop_result["evaluation_mode"] == "goal_spec"
    assert loop_result["goal_spec_hash"] == frozen_hash
    assert loop_result["evaluation"]["status"] == "failed"


def test_product_entrypoint_persists_non_operator_system_exit(tmp_path, monkeypatch):
    module = _load_module()
    scene_path = tmp_path / "scene.yaml"
    scene_path.write_text("name: scene\n", encoding="utf-8")
    urdf = tmp_path / "robot.urdf"
    urdf.write_text("placeholder", encoding="utf-8")
    output_dir = tmp_path / "output"

    def fake_freeze_goal_spec(*, scene, instruction, output_path, provider=None):
        module.write_json(output_path, {"schema_version": 1})
        module.write_json(
            output_path.with_name(output_path.name + ".provenance.json"),
            {"goal_spec_hash": "b" * 64, "goal_contract_hash": "c" * 64},
        )
        return object()

    monkeypatch.setattr(module, "freeze_goal_spec", fake_freeze_goal_spec)
    task_path = _task(tmp_path, "Make the item stable.")
    monkeypatch.setattr(module, "load_scene_data", lambda _data, **_: _scene())
    monkeypatch.setattr(
        module,
        "execute_product_episode",
        lambda **_kwargs: (_ for _ in ()).throw(SystemExit(0)),
    )
    monkeypatch.setattr(
        module,
        "_probe_gpu_health",
        lambda _physical_gpu_id: {"healthy": True, "physical_gpu_id": 6},
    )
    monkeypatch.setattr(sys, "argv", [
        "run_task.py",
        "--task", str(task_path),
        "--urdf", str(urdf),
        "--output-dir", str(output_dir),
    ])

    assert module.main() == 1
    result = json.loads((output_dir / "result.json").read_text(encoding="utf-8"))
    assert result["failure"]["category"] == "runtime_boundary"
    assert result["failure"]["exception_type"] == "SystemExit"
    assert result["failure"]["exception_code"] == 0
    assert result["evaluation"]["status"] == "failed"


def test_product_entrypoint_reuses_frozen_goal_spec(tmp_path, monkeypatch):
    module = _load_module()
    scene_path = tmp_path / "scene.yaml"
    scene_path.write_text("name: scene\n", encoding="utf-8")
    urdf = tmp_path / "robot.urdf"
    urdf.write_text("placeholder", encoding="utf-8")
    output_dir = tmp_path / "output"
    source = tmp_path / "frozen_goal_spec.json"
    source.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "bindings": {"subject": "scene://item"},
                "required": [{"predicate": "released", "arguments": {"subject": "subject"}}],
                "invariants": [],
            }
        ),
        encoding="utf-8",
    )

    def boom_freeze(**_kwargs):
        raise AssertionError("freeze_goal_spec should not run when --goal-spec is set")

    monkeypatch.setattr(module, "freeze_goal_spec", boom_freeze)
    task_path = _task(tmp_path, "Release the item.")
    monkeypatch.setattr(module, "load_scene_data", lambda _data, **_: _scene())

    captured = {}

    def fake_episode(**kwargs):
        captured.update(kwargs)
        module.write_json(kwargs["output_dir"] / "result.json", {"result": "passed", "status": "succeeded"})
        return {"result": "passed", "status": "succeeded"}

    monkeypatch.setattr(module, "execute_product_episode", fake_episode)
    monkeypatch.setattr(
        module,
        "_probe_gpu_health",
        lambda _physical_gpu_id: {"healthy": True, "physical_gpu_id": 6},
    )
    monkeypatch.setattr(sys, "argv", [
        "run_task.py",
        "--task", str(task_path),
        "--urdf", str(urdf),
        "--output-dir", str(output_dir),
        "--goal-spec", str(source),
    ])
    assert module.main() == 0
    reused = json.loads((output_dir / "goal_spec.json").read_text(encoding="utf-8"))
    assert reused["required"][0]["predicate"] == "released"
    assert captured["goal_spec_path"] == output_dir / "goal_spec.json"
