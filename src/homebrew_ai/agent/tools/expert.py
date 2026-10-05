"""Expert-only escape hatch: run a shell command (always confirmed by the user)."""

from __future__ import annotations

from typing import Any

from homebrew_ai.agent.tools.base import ToolContext, ToolError, tool
from homebrew_ai.agent.tools.compute import bootstrap_spec
from homebrew_ai.remote.target import LocalTarget, SSHTarget


@tool(
    "run_shell",
    """Run a shell command on this computer or the connected server, for debugging (expert level only; the user confirms
every command). Prefer the dedicated tools for anything they cover. Local commands use the system shell on Windows
and Bash or sh on POSIX systems. Remote commands use Bash.""",
    {"command": {"type": "string"}, "where": {"type": "string", "enum": ["local", "remote"]}, "timeout_s": {"type": "integer"}},
    ["command", "where"],
    levels=("expert",),
)
def run_shell(ctx: ToolContext, args: dict[str, Any]) -> Any:
    compute = ctx.project.state.compute
    if args["where"] == "remote":
        if compute.kind != "ssh" or not compute.ssh:
            raise ToolError("no server connected")
        target = SSHTarget(bootstrap_spec(compute.ssh))
    else:
        target = LocalTarget()
    ctx.confirm_or_raise(f"Run this command on {target.label}?", details=args["command"])
    res = target.run(args["command"], timeout=min(int(args.get("timeout_s") or 300), 3600))
    return {"exit_code": res.code, "stdout": res.stdout[-6000:], "stderr": res.stderr[-3000:]}
