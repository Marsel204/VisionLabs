"""Exercise the pinned official CLI adapter without Google login or allocation."""

from __future__ import annotations

import io
import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform not in {"linux", "darwin"},
    reason="The official Colab CLI supports Linux/macOS only",
)


@pytest.mark.parametrize("rejection", ["capacity", "gpu_rejected"])
def test_bridge_uses_official_remote_login_and_tracks_only_owned_runtimes(
    tmp_path, monkeypatch, capsys, rejection
):
    pytest.importorskip("colab_cli")
    from colab_cli import auth, cli, common
    from colab_cli.client import Client, ColabRequestError, TooManyAssignmentsError
    from colab_cli.commands import session, usage
    from google_auth_oauthlib.flow import InstalledAppFlow

    from app.services.colab_cli_bridge import main

    # Register restoration of globals the child-process adapter normally owns.
    for target, name in (
        (auth, "TOKEN_CONFIG_PATH"),
        (auth, "_run_remote_flow"),
        (common, "setup_logging"),
        (common.state, "_history"),
        (common.state, "_settings_store"),
        (session, "stop"),
        (usage, "usage"),
    ):
        monkeypatch.setattr(target, name, getattr(target, name))
    monkeypatch.setenv("VISIONLAB_COLAB_PROFILE", str(tmp_path))
    monkeypatch.setenv("VISIONLAB_COLAB_LOGIN", "1")
    monkeypatch.setattr("sys.stdin", io.StringIO("transient-test-code\n"))
    flow = Mock()
    flow.authorization_url.return_value = ("https://accounts.google.com/test", "state")
    factory = Mock(return_value=flow)
    monkeypatch.setattr(InstalledAppFlow, "from_client_config", factory)
    underlying_assign = Mock(return_value=SimpleNamespace(endpoint="owned-endpoint"))
    monkeypatch.setattr(Client, "assign", underlying_assign)
    client = Mock()
    client.get_consumption_user_info.return_value = SimpleNamespace(
        paid_compute_units_balance=42.5,
        consumption_rate_hourly=1.07,
        assignments_count=1,
    )
    client.session.get.return_value.json.return_value = {"email": "signed-in@example.com"}
    monkeypatch.setattr(common.state, "_client", client)
    store = Mock()
    store.get.return_value = None
    monkeypatch.setattr(common.state, "_store", store)
    pending = tmp_path / "pending-runtime.json"

    def exercise():
        assert auth.TOKEN_CONFIG_PATH == str(tmp_path / "token.json")
        assert auth._run_remote_flow({"installed": {}}) is flow.credentials
        factory.assert_called_once_with({"installed": {}}, auth.PUBLIC_SCOPES)
        assert flow.redirect_uri == auth.REMOTE_REDIRECT_URI
        flow.authorization_url.assert_called_once_with(
            prompt="select_account consent", token_usage="remote"
        )
        flow.fetch_token.assert_called_once_with(code="transient-test-code")
        usage.usage()
        client.session.get.assert_called_once_with(
            "https://www.googleapis.com/oauth2/v2/userinfo",
            timeout=15,
        )

        pending.write_text(
            json.dumps({"session": "visionlab-test", "allocation_state": "requested"})
        )
        with pytest.raises(RuntimeError, match="could not be confirmed"):
            session.stop(session="visionlab-test")
        client.unassign.assert_not_called()
        Client.assign(client)
        record = json.loads(pending.read_text())
        assert record["endpoint"] == "owned-endpoint" and record["allocation_state"] == "confirmed"
        with pytest.raises(RuntimeError, match="does not belong"):
            session.stop(session="someone-elses-runtime")
        session.stop(session="visionlab-test")
        client.unassign.assert_called_once_with("owned-endpoint")

        underlying_assign.side_effect = (
            TooManyAssignmentsError("No capacity")
            if rejection == "capacity"
            else ColabRequestError(
                "Private response detail", request=None, response=SimpleNamespace(status_code=400)
            )
        )
        pending.write_text(
            json.dumps({"session": "visionlab-test", "allocation_state": "requested"})
        )
        with pytest.raises((TooManyAssignmentsError, ColabRequestError)):
            Client.assign(client)
        assert json.loads(pending.read_text())["allocation_state"] == "refused"
        assert json.loads(pending.read_text())["allocation_error"] == rejection
        assert "Private response detail" not in pending.read_text()
        session.stop(session="visionlab-test")
        assert client.unassign.call_count == 1

    monkeypatch.setattr(cli, "main", exercise)
    main()
    output = capsys.readouterr()
    assert "VISIONLAB_AUTH " in output.out
    event = next(line for line in output.out.splitlines() if line.startswith("VISIONLAB_ACCOUNT "))
    assert json.loads(event[len("VISIONLAB_ACCOUNT ") :]) == {
        "email": "signed-in@example.com",
        "compute_units": 42.5,
        "usage_rate_hourly": 1.07,
        "active_runtimes": 1,
    }
    assert "transient-test-code" not in output.out + output.err
    assert not (tmp_path / "token.json").exists()
