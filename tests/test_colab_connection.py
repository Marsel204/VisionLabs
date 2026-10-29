"""Browser sign-in and automatic Colab jobs, without Google access or paid runtimes."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest
from PIL import Image

from app.services.colab_connection import ColabConnection, ColabConnectionError
from app.services.dataset.repository import AnnotationRepository

pytestmark = pytest.mark.skipif(
    sys.platform not in {"linux", "darwin"},
    reason="Direct Colab sessions require Linux/macOS; Windows uses the notebook fallback",
)


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / "dataset"
    root.mkdir()
    for index in range(5):
        Image.new("RGB", (100, 80), "white").save(root / f"{index}.jpg")
    return root


def connected(tmp_path, monkeypatch):
    manager = ColabConnection(tmp_path / "private")
    manager.profile.mkdir(parents=True)
    manager.token_path.write_text('{"refresh_token":"test-only"}')
    monkeypatch.setattr(manager, "_ensure_cli", lambda: sys.executable)
    return manager


def wait(manager):
    manager._thread.join(timeout=5)
    assert not manager._thread.is_alive()
    return manager.status()


def results_for(manager):
    request = json.loads((manager._workspace / "request.json").read_text())
    return {
        "version": 1,
        "job_id": request["job_id"],
        "images": [
            {
                "key": row["key"],
                "sha256": row["sha256"],
                "result": {
                    "version": 1,
                    "width": 100,
                    "height": 80,
                    "detections": [
                        {
                            "class_name": "car",
                            "confidence": 0.9,
                            "box": [0.1, 0.2, 0.7, 0.8],
                            "polygon": [[0.1, 0.2], [0.7, 0.2], [0.7, 0.8]],
                        }
                    ],
                },
            }
            for row in request["images"]
        ],
    }


def executor(manager, calls, *, fail_exec=False, fail_stop=False, before_download=None):
    def execute(args, **kwargs):
        calls.append(args)
        if args[0] == "exec" and fail_exec:
            raise ColabConnectionError("Model execution failed")
        if args[0] == "stop" and fail_stop:
            raise ColabConnectionError("Cannot release runtime")
        if args[0] == "download":
            if before_download:
                before_download()
            Path(args[-1]).write_text(json.dumps(results_for(manager)))

    return execute


def test_login_opens_only_google_and_never_starts_a_gpu(tmp_path, monkeypatch):
    manager = ColabConnection(tmp_path / "private")
    monkeypatch.setattr(manager, "_ensure_cli", lambda: sys.executable)
    browser = Mock()
    monkeypatch.setattr("app.services.colab_connection.webbrowser.open", browser)
    calls = []

    def authenticate(args, **kwargs):
        calls.append(args)
        manager._event(
            'VISIONLAB_AUTH {"url":"https://accounts.google.com/o/oauth2/auth?state=test"}'
        )
        assert manager.status()["auth_state"] == "awaiting_code"
        manager.token_path.write_text('{"refresh_token":"test-only"}')

    monkeypatch.setattr(manager, "_execute", authenticate)
    manager.connect()
    assert wait(manager)["auth_state"] == "connected"
    browser.assert_called_once()
    assert calls == [["usage"]]
    assert "refresh_token" not in json.dumps(manager.status())
    browser.reset_mock()
    manager._event('VISIONLAB_AUTH {"url":"https://malicious.example/sign-in"}')
    browser.assert_not_called()


def test_code_is_transient_and_only_accepted_during_login(tmp_path):
    manager = ColabConnection(tmp_path)
    with pytest.raises(ColabConnectionError):
        manager.authorize("private-auth-code")
    process = Mock()
    manager._process = process
    manager._auth_state = "awaiting_code"
    manager.authorize("private-auth-code")
    process.stdin.write.assert_called_once_with("private-auth-code\n")
    assert "private-auth-code" not in json.dumps(manager.status())


def test_three_image_job_imports_masks_and_always_stops_runtime(dataset, tmp_path, monkeypatch):
    manager = connected(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(manager, "_execute", executor(manager, calls))
    manager.start_job(dataset, [dataset / f"{i}.jpg" for i in range(3)], ["car"], "dino-masks")
    status = wait(manager)
    assert status["job"]["state"] == "completed", status
    assert status["job"]["saved"] == 3
    assert calls[0][0] == "usage" and calls[1][0] == "new" and calls[-1][0] == "stop"
    assert len([c for c in calls if c[0] == "upload"]) == 2
    execution = next(c for c in calls if c[0] == "exec")
    assert execution[-2:] == ["--timeout", "1800"]
    request = json.loads((manager._workspace / "request.json").read_text())
    assert len(request["images"]) == 3
    assert str(dataset) not in json.dumps(request)
    repo = AnnotationRepository(dataset)
    stored = repo.read(dataset / "0.jpg")
    assert stored["status"] == "ai_labeled"
    assert stored["boxes"][0]["polygon_normalized"]
    assert not repo.canonical_path(dataset / "3.jpg").exists()


def test_execution_failure_still_releases_gpu(dataset, tmp_path, monkeypatch):
    manager = connected(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(manager, "_execute", executor(manager, calls, fail_exec=True))
    manager.start_job(dataset, [dataset / "0.jpg"], ["car"], "dino-masks")
    status = wait(manager)
    assert status["job"]["state"] == "failed"
    assert calls[-1][0] == "stop"
    assert not status["job"]["cleanup_required"]


def test_failed_release_remains_visible_and_credentials_are_retained(
    dataset, tmp_path, monkeypatch
):
    manager = connected(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(manager, "_execute", executor(manager, calls, fail_stop=True))
    manager.start_job(dataset, [dataset / "0.jpg"], ["car"], "dino-masks")
    status = wait(manager)
    assert status["job"]["cleanup_required"]
    assert manager.pending_path.exists()
    with pytest.raises(ColabConnectionError):
        manager.disconnect()
    assert manager.token_path.exists()


def test_late_results_do_not_replace_human_edits(dataset, tmp_path, monkeypatch):
    manager = connected(tmp_path, monkeypatch)
    repo = AnnotationRepository(dataset)

    def human_edit():
        repo.save(dataset / "0.jpg", [], status="reviewed", expected_revision=0)

    calls = []
    monkeypatch.setattr(manager, "_execute", executor(manager, calls, before_download=human_edit))
    manager.start_job(dataset, [dataset / "0.jpg"], ["car"], "dino-masks")
    status = wait(manager)
    assert status["job"]["saved"] == 0 and status["job"]["conflicts"] == 1
    assert repo.read(dataset / "0.jpg")["status"] == "reviewed"
    assert repo.read(dataset / "0.jpg")["boxes"] == []


def test_invalid_response_cannot_save_annotations(dataset, tmp_path, monkeypatch):
    manager = connected(tmp_path, monkeypatch)
    calls = []

    def execute(args, **kwargs):
        calls.append(args)
        if args[0] == "download":
            payload = results_for(manager)
            payload["images"][0]["sha256"] = "wrong"
            Path(args[-1]).write_text(json.dumps(payload))

    monkeypatch.setattr(manager, "_execute", execute)
    manager.start_job(dataset, [dataset / "0.jpg"], ["car"], "dino-masks")
    assert wait(manager)["job"]["state"] == "failed"
    assert not AnnotationRepository(dataset).canonical_path(dataset / "0.jpg").exists()
    assert calls[-1][0] == "stop"


def test_cancel_during_allocation_releases_gpu_without_uploading(dataset, tmp_path, monkeypatch):
    manager = connected(tmp_path, monkeypatch)
    calls = []

    def execute(args, **kwargs):
        calls.append(args)
        if args[0] == "new":
            assert kwargs["cleanup"]  # Allocation completes before cancellation is honored.
            manager.cancel()

    monkeypatch.setattr(manager, "_execute", execute)
    manager.start_job(dataset, [dataset / "0.jpg"], ["car"], "dino-boxes")
    assert wait(manager)["job"]["state"] == "canceled"
    assert [call[0] for call in calls] == ["usage", "new", "stop"]
    assert not manager.pending_path.exists()


def test_restart_can_retry_release_and_manual_confirmation_retains_credentials(
    tmp_path, monkeypatch
):
    manager = connected(tmp_path, monkeypatch)
    manager.pending_path.write_text(json.dumps({"session": "visionlab-test"}))
    restarted = ColabConnection(manager.profile)
    monkeypatch.setattr(restarted, "_ensure_cli", lambda: sys.executable)
    calls = []
    monkeypatch.setattr(restarted, "_execute", lambda args, **kwargs: calls.append(args))
    assert restarted.status()["job"]["cleanup_required"]
    restarted.cancel()
    assert not wait(restarted)["job"]["cleanup_required"]
    assert calls == [["stop", "-s", "visionlab-test"]]
    restarted.pending_path.write_text(json.dumps({"session": "visionlab-test"}))
    restarted.acknowledge_release()
    assert restarted.token_path.exists()
    assert not restarted.pending_path.exists()


def test_existing_boxes_are_preserved_without_duplicates_and_sources_are_correct(
    dataset, tmp_path, monkeypatch
):
    manager = connected(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(manager, "_execute", executor(manager, calls))
    manager.start_job(dataset, [dataset / "0.jpg"], ["car"], "dino-boxes")
    assert wait(manager)["job"]["saved"] == 1
    repo = AnnotationRepository(dataset)
    first = repo.read(dataset / "0.jpg")
    assert first["boxes"][0]["source"] == "grounding_dino"
    manager.start_job(dataset, [dataset / "0.jpg"], ["car"], "dino-boxes")
    assert wait(manager)["job"]["saved"] == 1
    assert repo.read(dataset / "0.jpg")["boxes"] == first["boxes"]


def test_cli_bridge_help_runs_with_private_profile_without_sign_in(tmp_path):
    pytest.importorskip("colab_cli")
    manager = ColabConnection(tmp_path / "private")
    manager._prepare()
    manager._execute(["exec", "--help"], timeout=20)
    manager._execute(["stop", "--help"], timeout=20)
    assert not manager.token_path.exists()
    assert not manager.pending_path.exists()
    assert manager.status()["auth_state"] == "disconnected"


def test_expired_authorization_fails_before_requesting_a_gpu(dataset, tmp_path, monkeypatch):
    manager = connected(tmp_path, monkeypatch)
    calls = []

    def execute(args, **kwargs):
        calls.append(args)
        raise ColabConnectionError("Authorization needs refreshing")

    monkeypatch.setattr(manager, "_execute", execute)
    manager.start_job(dataset, [dataset / "0.jpg"], ["car"], "dino-boxes")
    status = wait(manager)
    assert status["job"]["state"] == "failed"
    assert calls == [["usage"]]
    assert not manager.pending_path.exists()


@pytest.mark.parametrize("gpu", ["T4", "L4", "A100", "H100", "G4"])
def test_each_supported_gpu_is_forwarded_without_substitution(dataset, tmp_path, monkeypatch, gpu):
    manager = connected(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(manager, "_execute", executor(manager, calls))
    assert gpu in manager.status()["supported_gpus"]
    manager.start_job(dataset, [dataset / "0.jpg"], ["car"], "dino-boxes", gpu=gpu)
    status = wait(manager)
    assert status["job"]["state"] == "completed"
    creation = next(call for call in calls if call[0] == "new")
    assert creation[-2:] == ["--gpu", gpu]
    assert len([call for call in calls if call[0] == "new"]) == 1
    assert calls[-1][0] == "stop"


@pytest.mark.parametrize("reason", ["gpu_rejected", "capacity"])
def test_allocation_rejection_explains_the_reason_without_trying_another_gpu(
    dataset, tmp_path, monkeypatch, reason
):
    manager = connected(tmp_path, monkeypatch)
    calls = []

    def execute(args, **kwargs):
        calls.append(args)
        if args[0] == "new":
            record = json.loads(manager.pending_path.read_text())
            record.update(allocation_state="refused", allocation_error=reason)
            manager.pending_path.write_text(json.dumps(record))
            raise ColabConnectionError("Allocation refused")

    monkeypatch.setattr(manager, "_execute", execute)
    manager.start_job(dataset, [dataset / "0.jpg"], ["car"], "dino-boxes")
    status = wait(manager)
    assert status["job"]["state"] == "failed"
    assert "T4" in status["error"]
    assert ("do not guarantee" if reason == "gpu_rejected" else "HTTP 412") in status["error"]
    assert status["job"]["stage"] == "No Colab GPU was allocated"
    assert [call[0] for call in calls] == ["usage", "new", "stop"]
    assert not manager.pending_path.exists()


def test_unknown_gpu_is_rejected_before_any_cloud_request(dataset, tmp_path, monkeypatch):
    manager = connected(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(manager, "_execute", executor(manager, calls))
    with pytest.raises(ColabConnectionError):
        manager.start_job(dataset, [dataset / "0.jpg"], ["car"], "dino-boxes", gpu="unknown")
    assert calls == []


def account_message(email="free@example.com", compute_units=0, active_runtimes=1):
    return "VISIONLAB_ACCOUNT " + json.dumps(
        {
            "email": email,
            "compute_units": compute_units,
            "usage_rate_hourly": 1.07,
            "active_runtimes": active_runtimes,
        }
    )


def test_switch_account_forces_login_without_deleting_jobs_or_stopping_other_runtimes(
    tmp_path, monkeypatch
):
    manager = connected(tmp_path, monkeypatch)
    transfer = manager.profile / "jobs/existing/request.json"
    transfer.parent.mkdir(parents=True)
    transfer.write_text("preserve this transfer")
    manager._event(account_message(), allow_account=True)
    calls = []

    def authenticate(args, **kwargs):
        assert not manager.token_path.exists()
        calls.append(args)
        manager.token_path.write_text('{"refresh_token":"new-test-account"}')
        manager._event(account_message("paid@example.com", 80, 0), allow_account=True)

    monkeypatch.setattr(manager, "_execute", authenticate)
    manager.switch_account()
    status = wait(manager)
    assert status["auth_state"] == "connected"
    assert status["account"]["email"] == "paid@example.com"
    assert status["account"]["compute_units"] == 80
    assert transfer.read_text() == "preserve this transfer"
    assert calls == [["usage"]]


def test_switch_account_is_blocked_until_owned_runtime_cleanup_finishes(tmp_path, monkeypatch):
    manager = connected(tmp_path, monkeypatch)
    manager.pending_path.write_text(json.dumps({"session": "visionlab-test"}))
    with pytest.raises(ColabConnectionError):
        manager.switch_account()
    assert manager.token_path.exists() and manager.pending_path.exists()


def test_account_details_refresh_is_read_only_and_disconnect_clears_identity(tmp_path, monkeypatch):
    manager = connected(tmp_path, monkeypatch)
    calls = []

    def inspect(args, **kwargs):
        calls.append(args)
        manager._event(account_message(), allow_account=True)

    monkeypatch.setattr(manager, "_execute", inspect)
    manager.refresh_account()
    status = wait(manager)
    assert status["account"]["email"] == "free@example.com"
    assert status["account"]["compute_units"] == 0
    assert status["account"]["active_runtimes"] == 1
    assert calls == [["usage"]]
    assert not manager.pending_path.exists()
    manager.disconnect()
    assert manager.status()["account"] is None


def test_remote_model_output_cannot_change_connected_account(tmp_path, monkeypatch):
    manager = connected(tmp_path, monkeypatch)
    manager._event(account_message("fake@example.com", 100, 0))
    assert manager.status()["account"] is None
