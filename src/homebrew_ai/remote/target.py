"""Where commands run: this machine, or a GPU server over SSH.

Both targets expose the same small surface (``run``, ``put``, ``get``) so the
job manager does not care where training happens. File transfer streams a
tar archive through the SSH connection, which works with any SSH server that
allows commands (no rsync or SFTP needed on the remote side).
"""

from __future__ import annotations

import contextlib
import os
import shlex
import shutil
import subprocess
import sys
import tarfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from homebrew_ai.remote.sshutil import SSHSpec, check_option, validate_spec


@dataclass
class RunResult:
    code: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.code == 0

    def raise_for_error(self, what: str = "command") -> "RunResult":
        if self.code != 0:
            tail = (self.stderr or self.stdout).strip().splitlines()[-15:]
            raise RuntimeError(f"{what} failed (exit {self.code}):\n" + "\n".join(tail))
        return self


def _tar_filter_kwargs() -> dict:
    return {"filter": "data"} if hasattr(tarfile, "data_filter") else {}


def _local_shell(cmd: str) -> str | list[str]:
    if sys.platform == "win32":
        return cmd  # subprocess uses the Windows system shell, without an extra shell dependency.
    shell = shutil.which("bash")
    if shell:
        return [shell, "-lc", cmd]
    shell = shutil.which("sh")
    if not shell:
        raise RuntimeError("Local shell commands require Bash or sh.")
    return [shell, "-c", cmd]


class LocalTarget:
    kind = "local"

    def __init__(self, label: str = "this computer"):
        self.label = label

    def run(self, cmd: str, *, timeout: float | None = 600, input: str | None = None) -> RunResult:
        shell = _local_shell(cmd)
        proc = subprocess.run(shell, shell=isinstance(shell, str), capture_output=True, text=True, encoding="utf-8", errors="replace",
                              timeout=timeout, input=input, env=dict(os.environ, PYTHONUTF8="1"),
                              creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0)
        return RunResult(proc.returncode, proc.stdout, proc.stderr)

    def stream(self, cmd: str, on_line: Callable[[str], None], *, input: str | None = None) -> int:
        shell = _local_shell(cmd)
        proc = subprocess.Popen(shell, shell=isinstance(shell, str), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                stdin=subprocess.PIPE if input else None, text=True, encoding="utf-8", errors="replace",
                                env=dict(os.environ, PYTHONUTF8="1"),
                                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0)
        if input and proc.stdin:
            proc.stdin.write(input)
            proc.stdin.close()
        assert proc.stdout is not None
        for line in proc.stdout:
            on_line(line.rstrip("\n"))
        return proc.wait()

    def put(self, local: str | Path, remote: str) -> None:
        local, dest = Path(local), Path(os.path.expanduser(remote))
        if local.resolve() == dest.resolve():
            return
        if local.is_dir():
            shutil.copytree(local, dest, dirs_exist_ok=True)
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(local, dest)

    def get(self, remote: str, local: str | Path) -> None:
        self.put(Path(os.path.expanduser(remote)), str(local))

    def to_dict(self) -> dict:
        return {"kind": "local", "label": self.label}


class SSHTarget:
    kind = "ssh"

    def __init__(self, spec: SSHSpec, connect_timeout: int = 15):
        validate_spec(spec)
        self.spec = spec
        self.connect_timeout = connect_timeout
        self.label = spec.label

    def _base(self) -> list[str]:
        if not shutil.which("ssh"):
            raise RuntimeError("the 'ssh' command was not found; install OpenSSH")
        args = [
            "ssh",
            "-p", str(self.spec.port),
            "-o", "BatchMode=yes",
            "-o", "StrictHostKeyChecking=accept-new",
            "-o", f"ConnectTimeout={self.connect_timeout}",
            "-o", "ServerAliveInterval=30",
            "-o", "ServerAliveCountMax=6",
        ]
        if self.spec.key:
            args += ["-i", os.path.expanduser(self.spec.key), "-o", "IdentitiesOnly=yes"]
        for opt in self.spec.options:
            opt = check_option(opt)
            key, _, value = opt.partition("=")
            if key.lower() == "userknownhostsfile":
                value = Path(value).expanduser().as_posix() if value != "/dev/null" else value
                value = value.replace("\\", "\\\\").replace('"', '\\"')
                opt = f'{key}="{value}"'
            args += ["-o", opt]
        args.append(f"{self.spec.user}@{self.spec.host}")
        return args

    def _wrap(self, cmd: str) -> str:
        return "bash -lc " + shlex.quote(cmd)

    def run(self, cmd: str, *, timeout: float | None = 600, input: str | None = None) -> RunResult:
        try:
            proc = subprocess.run(self._base() + [self._wrap(cmd)], capture_output=True, text=True, encoding="utf-8",
                                  errors="replace", timeout=timeout, input=input)
        except subprocess.TimeoutExpired:
            return RunResult(124, "", f"timed out after {timeout}s")
        return RunResult(proc.returncode, proc.stdout, proc.stderr)

    def stream(self, cmd: str, on_line: Callable[[str], None], *, input: str | None = None) -> int:
        proc = subprocess.Popen(
            self._base() + [self._wrap(cmd)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace"
        )
        assert proc.stdin is not None and proc.stdout is not None
        if input:
            proc.stdin.write(input)
        proc.stdin.close()
        for line in proc.stdout:
            on_line(line.rstrip("\n"))
        return proc.wait()

    def put(self, local: str | Path, remote: str) -> None:
        """Copy a file or directory to ``remote`` (the destination path itself)."""
        local = Path(local)
        dest = remote.rstrip("/")
        parent = dest.rsplit("/", 1)[0] if "/" in dest else "."
        name = dest.rsplit("/", 1)[-1]
        cmd = f"mkdir -p -- {_q(parent)} && tar xzf - -C {_q(parent)}"
        proc = subprocess.Popen(self._base() + [self._wrap(cmd)], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        assert proc.stdin is not None and proc.stderr is not None
        err, drain = _drain(proc.stderr)
        try:
            with tarfile.open(fileobj=proc.stdin, mode="w|gz") as tf:
                tf.add(str(local), arcname=name, filter=_skip_junk)
        except BrokenPipeError:
            pass  # the remote side stopped reading; its error output below says why
        except BaseException:
            proc.kill()
            proc.wait()
            raise
        finally:
            with contextlib.suppress(OSError, ValueError):
                proc.stdin.close()
        code = proc.wait()
        drain.join(timeout=30)
        if code != 0:
            raise RuntimeError(f"upload to {self.label}:{remote} failed (exit {code}): {_tail(err)}")

    def get(self, remote: str, local: str | Path) -> None:
        """Copy remote file/dir ``remote`` to the local path ``local``."""
        local = Path(local)
        src = remote.rstrip("/")
        parent = src.rsplit("/", 1)[0] if "/" in src else "."
        name = src.rsplit("/", 1)[-1]
        cmd = f"tar czf - -C {_q(parent)} -- {_q(name)}"
        local.parent.mkdir(parents=True, exist_ok=True)
        staging = local.parent / f".incoming-{name}-{os.getpid()}-{threading.get_ident()}"
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir()
        proc = subprocess.Popen(self._base() + [self._wrap(cmd)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        assert proc.stdout is not None and proc.stderr is not None
        err, drain = _drain(proc.stderr)
        read_error: Exception | None = None
        try:
            with tarfile.open(fileobj=proc.stdout, mode="r|gz") as tf:
                tf.extractall(staging, **_tar_filter_kwargs())
        except (tarfile.TarError, OSError, EOFError) as exc:
            read_error = exc  # usually an empty stream because the remote tar failed; its error output says why
        finally:
            with contextlib.suppress(OSError, ValueError):
                proc.stdout.read()
        code = proc.wait()
        drain.join(timeout=30)
        if code != 0 or read_error is not None:
            shutil.rmtree(staging, ignore_errors=True)
            why = _tail(err) if code != 0 else f"{type(read_error).__name__}: {read_error}"
            raise RuntimeError(f"download of {self.label}:{remote} failed (exit {code}): {why}")
        fetched = staging / name
        if local.exists() and local.is_dir() and fetched.is_dir():
            shutil.copytree(fetched, local, dirs_exist_ok=True)
        else:
            if local.exists() and local.is_dir():
                shutil.rmtree(local)
            elif local.exists():
                local.unlink()
            shutil.move(str(fetched), str(local))
        shutil.rmtree(staging, ignore_errors=True)

    def test(self) -> RunResult:
        return self.run("echo homebrew-ok && uname -sm", timeout=self.connect_timeout + 15)

    def to_dict(self) -> dict:
        return {"kind": "ssh", "label": self.label, **self.spec.to_dict()}


def _drain(stream) -> tuple[list[bytes], threading.Thread]:
    """Read ``stream`` to the end in the background so a chatty remote can never block the transfer."""
    sink: list[bytes] = []
    thread = threading.Thread(target=lambda: sink.append(stream.read()), daemon=True)
    thread.start()
    return sink, thread


def _tail(chunks: list[bytes]) -> str:
    text = b"".join(chunks).decode(errors="replace").strip()
    return text[-500:] or "no error output"


def _q(path: str) -> str:
    """Quote a remote path but keep a leading ``~/`` expandable."""
    if path.startswith("~/"):
        return '"$HOME"/' + shlex.quote(path[2:])
    if path == "~":
        return '"$HOME"'
    return shlex.quote(path)


def _skip_junk(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
    parts = info.name.split("/")
    if any(p in ("__pycache__", ".pytest_cache", ".git") for p in parts) or info.name.endswith((".pyc", ".pyo")):
        return None
    return info


def target_from_dict(data: dict) -> LocalTarget | SSHTarget:
    if data.get("kind") == "ssh":
        spec = SSHSpec(
            host=data["host"], user=data.get("user", "root"), port=int(data.get("port", 22)), key=data.get("key"),
            options=list(data.get("options", [])), provider=data.get("provider"),
        )
        return SSHTarget(spec)
    return LocalTarget(data.get("label", "this computer"))


def python_executable() -> str:
    return sys.executable
