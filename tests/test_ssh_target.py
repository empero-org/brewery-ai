"""SSHTarget end-to-end against a fake ``ssh`` that runs commands locally (no server needed)."""

import os
import stat
import sys
from pathlib import Path

import pytest

from homebrew_ai.remote.sshutil import SSHSpec
from homebrew_ai.remote.target import SSHTarget

FAKE_SSH = """#!/usr/bin/env python3
import subprocess, sys
args = sys.argv[1:]
# skip options (with their values) and the user@host argument; the last argument is the command
skip_value = {"-p", "-o", "-i", "-l", "-L", "-R", "-D", "-J", "-F"}
rest, i = [], 0
while i < len(args):
    a = args[i]
    if a in skip_value:
        i += 2
        continue
    if a.startswith("-"):
        i += 1
        continue
    rest.append(a)
    i += 1
cmd = rest[-1]
sys.exit(subprocess.call(["bash", "-c", cmd]))
"""


@pytest.fixture
def fake_ssh(tmp_path, monkeypatch):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    exe = bindir / "ssh"
    exe.write_text(FAKE_SSH.replace("#!/usr/bin/env python3", f"#!{sys.executable}"))
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir()
    return SSHTarget(SSHSpec(host="example.invalid", port=2222))


def test_run_put_get_roundtrip(fake_ssh, tmp_path):
    t = fake_ssh
    assert t.test().ok
    res = t.run("echo hi; echo err >&2; exit 3")
    assert res.code == 3 and res.stdout.strip() == "hi" and "err" in res.stderr
    src = tmp_path / "pkg"
    (src / "sub").mkdir(parents=True)
    (src / "a.txt").write_text("A")
    (src / "sub" / "b.txt").write_text("B")
    (src / "__pycache__").mkdir()
    (src / "__pycache__" / "x.pyc").write_text("junk")
    remote_root = tmp_path / "remote" / "deep"
    t.put(src, f"{remote_root}/pkg")
    assert (remote_root / "pkg" / "sub" / "b.txt").read_text() == "B"
    assert not (remote_root / "pkg" / "__pycache__").exists()
    t.put(src / "a.txt", "~/work/a.txt")  # ~ expands on the remote side
    assert (tmp_path / "home" / "work" / "a.txt").read_text() == "A"
    back = tmp_path / "back"
    t.get(f"{remote_root}/pkg", back)
    assert (back / "sub" / "b.txt").read_text() == "B"
    t.get(f"{remote_root}/pkg/a.txt", tmp_path / "single.txt")
    assert (tmp_path / "single.txt").read_text() == "A"


def test_errors_carry_the_remote_reason(fake_ssh, tmp_path):
    with pytest.raises(RuntimeError, match="download of .* failed"):
        fake_ssh.get(f"{tmp_path}/does-not-exist", tmp_path / "x")
    assert not any(p.name.startswith(".incoming-") for p in tmp_path.iterdir())
    blocker = tmp_path / "file"
    blocker.write_text("x")
    with pytest.raises(RuntimeError, match="upload to .* failed"):
        fake_ssh.put(blocker, f"{blocker}/inside/target")  # parent is a file: mkdir fails remotely
