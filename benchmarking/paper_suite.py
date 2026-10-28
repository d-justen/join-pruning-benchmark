"""Run independent paper experiment groups sequentially in one container."""
from __future__ import annotations

import json
import os
from pathlib import Path

from benchmarking.suite_io import fingerprint, lock, sha256, write_json
from benchmarking.suite_plan import validate_plan


def summarize(run: Path, raw: dict) -> dict:
    groups = {}
    for group in raw["groups"]:
        directory = run / "groups" / group["name"]
        def read(name):
            path = directory / name
            return json.loads(path.read_text()) if path.exists() else {}
        state, coverage, analysis = read("run.json"), read("coverage.json"), read("analysis.json")
        complete = (state.get("status") == "complete" and coverage.get("status") == "complete"
                    and analysis.get("status") == "complete")
        groups[group["name"]] = {
            "status": "complete" if complete else "partial" if state else "pending",
            "directory": str(directory.relative_to(run)),
            "figures": [str(p.relative_to(run)) for p in sorted((directory / "paper/figures").glob("*.pdf"))],
            "failures": state.get("failures", []),
        }
    result = {"status": "complete" if all(g["status"] == "complete" for g in groups.values()) else "partial",
              "groups": groups}
    write_json(run / "summary.json", result)
    return result


def analyze(run: Path, raw: dict) -> None:
    from benchmarking import suite
    errors = []
    for group in raw["groups"]:
        directory = run / "groups" / group["name"]
        if not (directory / "config.json").exists():
            errors.append(group["name"] + ": not started")
            continue
        try:
            with lock(directory / ".lock"):
                suite.analyze(directory, paper=group["config"]["suite"].get("paper_figures", False))
        except Exception as exc:
            errors.append(group["name"] + ": " + str(exc))
            write_json(directory / "analysis.json", {"status": "failed", "error": str(exc)})
    summary = summarize(run, raw)
    if errors:
        summary["status"] = "partial"
        write_json(run / "summary.json", summary)
    write_json(run / "analysis.json", {"status": summary["status"], "errors": errors})
    if errors or summary["status"] != "complete":
        raise ValueError("paper suite is incomplete; see summary.json: " + "; ".join(errors))


def execute(raw: dict, stage: str, run: Path, cache: Path, resume: bool) -> int:
    from benchmarking import suite
    validate_plan(raw)
    for group in raw["groups"]:
        suite.validate(group["config"])
    run, cache = run.resolve(), cache.resolve()
    with lock(run / ".lock"), lock(cache / ".suite.lock"):
        identity = {"config_sha256": fingerprint(raw), "source": suite.source_identity(),
                    "image_id": json.loads(os.environ.get("BENCH_IMAGE_METADATA", "{}")).get("id"),
                    "binaries": {s["binary_path"]: sha256(Path(s["binary_path"]))
                                 for g in raw["groups"] for s in g["config"]["systems"]}}
        state_path = run / "run.json"
        if state_path.exists():
            previous = json.loads(state_path.read_text())
            if not resume:
                raise ValueError("run already exists; use --resume")
            if any(previous.get(k) != v for k, v in identity.items()):
                raise ValueError("paper configuration, source, image or binaries changed; start a new run")
        elif any(p.name != ".lock" for p in run.iterdir()):
            raise ValueError("refusing nonempty paper run directory without run.json")
        state = {**identity, "status": "running", "stage": stage, "updated_at": suite.now()}
        write_json(state_path, state)
        write_json(run / "config.json", raw)
        write_json(run / "request.json", raw)
        failures = []
        try:
            for group in raw["groups"]:
                suite.check_deadline()
                name = group["name"]
                config_path = run / "inputs" / (name + ".json")
                write_json(config_path, group["config"])
                directory = run / "groups" / name
                try:
                    code = suite._execute(config_path, stage, directory, cache,
                                          resume=resume and (directory / "run.json").exists(),
                                          package=False, cache_locked=True)
                    if code:
                        failures.append(name)
                except (InterruptedError, KeyboardInterrupt):
                    raise
                except Exception as exc:
                    failures.append(name + ": " + str(exc))
                    write_json(run / "errors" / (name + ".json"), {"error": str(exc)})
                summarize(run, raw)
                suite.checkpoint()
        except (InterruptedError, KeyboardInterrupt) as exc:
            failures.append(str(exc) or "interrupted")
            state["interruption"] = "deadline" if isinstance(exc, suite.DeadlineReached) else "signal"
        finally:
            summary = summarize(run, raw)
            if failures:
                summary.update(status="partial", failures=failures)
                write_json(run / "summary.json", summary)
            status = {"prepare": "prepared", "run": "measured"}.get(stage, summary["status"])
            state.update(status="partial" if failures else status, failures=failures, updated_at=suite.now())
            write_json(state_path, state)
            if stage == "all":
                suite.export(run)
        return int(bool(failures) or state["status"] not in ("complete", "prepared", "measured"))
