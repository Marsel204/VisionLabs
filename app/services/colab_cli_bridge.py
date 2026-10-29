"""Run the official Colab CLI with VisionLab-owned credentials and structured login events."""

from __future__ import annotations

import contextlib
import json
import logging
import os
import sys
from functools import wraps
from pathlib import Path


def main() -> None:
    from colab_cli import auth, common
    from colab_cli.history import HistoryLogger
    from colab_cli.state import SettingsStore
    from google_auth_oauthlib.flow import InstalledAppFlow

    profile = Path(os.environ["VISIONLAB_COLAB_PROFILE"])
    profile.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.umask(0o077)
    auth.TOKEN_CONFIG_PATH = str(profile / "token.json")
    common.state._history = HistoryLogger(str(profile / "history"))
    common.state._settings_store = SettingsStore(str(profile / "settings.json"))
    settings = common.state._settings_store.load()
    settings.enable_update_check = False
    common.state._settings_store.save(settings)

    # Journal the assigned endpoint before the CLI writes its own session store.
    # A missing CLI session alone does not prove that an interrupted allocation stopped.
    from colab_cli.client import Client, ColabRequestError, TooManyAssignmentsError
    from colab_cli.commands import session as session_commands
    from colab_cli.commands import usage as usage_commands
    from colab_cli.utils import get_status_code

    pending_path = profile / "pending-runtime.json"

    def journal(**values):
        record = json.loads(pending_path.read_text())
        record.update(values)
        temporary = pending_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(record))
        temporary.replace(pending_path)

    assign = Client.assign

    @wraps(assign)
    def tracked_assign(client, *args, **kwargs):
        try:
            assigned = assign(client, *args, **kwargs)
        except (TooManyAssignmentsError, ColabRequestError) as error:
            if isinstance(error, TooManyAssignmentsError) or get_status_code(error) == 400:
                journal(
                    allocation_state="refused",
                    allocation_error=(
                        "capacity" if isinstance(error, TooManyAssignmentsError) else "gpu_rejected"
                    ),
                )
            raise
        journal(allocation_state="confirmed", endpoint=assigned.endpoint)
        return assigned

    Client.assign = tracked_assign

    @wraps(session_commands.stop)
    def stop_owned_runtime(session=None):
        record = json.loads(pending_path.read_text())
        if not session or record.get("session") != session:
            raise RuntimeError("This runtime does not belong to the current VisionLab job")
        local = common.state.store.get(session)
        endpoint = local.endpoint if local else record.get("endpoint")
        if not endpoint:
            if record.get("allocation_state") == "refused":
                return
            raise RuntimeError("Runtime allocation could not be confirmed; check Google Colab")
        try:
            common.state.client.unassign(endpoint)
        except ColabRequestError as error:
            if get_status_code(error) != 404:
                raise
        common.state.store.remove(session)

    session_commands.stop = stop_owned_runtime

    @wraps(usage_commands.usage)
    def account_usage():
        client = common.state.client
        usage = client.get_consumption_user_info()
        email = None
        with contextlib.suppress(Exception):
            response = client.session.get(
                "https://www.googleapis.com/oauth2/v2/userinfo", timeout=15
            )
            response.raise_for_status()
            email = response.json().get("email")
        print(
            "VISIONLAB_ACCOUNT "
            + json.dumps(
                {
                    "email": email,
                    "compute_units": usage.paid_compute_units_balance,
                    "usage_rate_hourly": usage.consumption_rate_hourly,
                    "active_runtimes": usage.assignments_count,
                },
                allow_nan=False,
            ),
            flush=True,
        )

    usage_commands.usage = account_usage

    def login(config):
        if os.environ.get("VISIONLAB_COLAB_LOGIN") != "1":
            raise RuntimeError("Reconnect Colab in VisionLab to refresh authorization")
        flow = InstalledAppFlow.from_client_config(config, auth.PUBLIC_SCOPES)
        flow.redirect_uri = auth.REMOTE_REDIRECT_URI
        url, _ = flow.authorization_url(prompt="select_account consent", token_usage="remote")
        print("VISIONLAB_AUTH " + json.dumps({"url": url}), flush=True)
        code = sys.stdin.readline().strip()
        if not code:
            raise RuntimeError("Sign-in was canceled")
        flow.fetch_token(code=code)
        return flow.credentials

    auth._run_remote_flow = login
    # Keep SDK diagnostics out of another tool's global profile.
    common.setup_logging = lambda *args, **kwargs: logging.basicConfig(level=logging.WARNING)
    from colab_cli.cli import main as cli_main

    cli_main()


if __name__ == "__main__":
    main()
