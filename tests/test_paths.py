"""Portable project paths, subprocess arguments, and SSH path quoting."""

import hashlib
import io
import json
import shlex
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import SimpleNamespace

import pytest

from brewery_ai.jobs.manager import JobManager
from brewery_ai.project import DatasetEntry, Project
from brewery_ai.remote import bootstrap, sshutil, target


def test_project_paths_survive_persistence_and_relocation(tmp_path):
    project = Project.create(tmp_path / "project with spaces Ω")
    file = project.data_dir / "nested folder" / "sample's Ω.jsonl"
    file.parent.mkdir()
    file.write_text("{}\n", encoding="utf-8")
    relative = project.rel(file)
    assert relative == "data/nested folder/sample's Ω.jsonl"
    project.state.datasets["sample"] = DatasetEntry(path=relative)
    project.save()
    copy = tmp_path / "project copy Ω"
    shutil.copytree(project.root, copy)
    moved = Project.load(copy)
    assert moved.abs(moved.state.datasets["sample"].path).read_text(encoding="utf-8") == "{}\n"
    assert PureWindowsPath(relative).parts == PurePosixPath(relative).parts
    external = tmp_path / "external file.txt"
    assert Path(project.rel(external)) == external.resolve()


def test_local_file_transfer_and_worker_keep_space_and_unicode_paths(tmp_path):
    source = tmp_path / "source folder" / "records Ω.jsonl"
    source.parent.mkdir()
    source.write_text(json.dumps({"text": "hello Ω"}, ensure_ascii=False) + "\n", encoding="utf-8")
    project = Project.create(tmp_path / "worker project Ω")
    run_dir = project.runs_dir / "run with spaces"
    run_dir.mkdir()
    local = target.LocalTarget()
    local.put(source, str(run_dir / source.name))
    back = tmp_path / "copied folder" / source.name
    local.get(str(run_dir / source.name), back)
    assert back.read_bytes() == source.read_bytes()
    project.state.jobs.append({"job_id": "run with spaces", "run_dir": str(run_dir), "target_spec": {"kind": "local"}})
    output = JobManager(project).run_worker("run with spaces", ["etf", "stats", source.name], timeout=30)
    assert json.loads(output)["records"] == 1


@pytest.mark.parametrize("platform,text,expected", [
    ("win32", r'ssh -i C:\Users\Example\.ssh\key root@host', r'C:\Users\Example\.ssh\key'),
    ("win32", r'ssh -i "C:\Users\Example User\.ssh\my key" root@host', r'C:\Users\Example User\.ssh\my key'),
    ("win32", r'ssh -i "\\server\shared keys\my key" root@host', r'\\server\shared keys\my key'),
    ("win32", r'"C:\Program Files\OpenSSH\ssh.exe" -i "C:\key folder\my key" root@host', r'C:\key folder\my key'),
    ("linux", r'ssh -i /home/example/key\ folder/my\ key root@host', '/home/example/key folder/my key'),
    ("linux", "'/opt/SSH tools/ssh' -i '/home/example/key folder/my key' root@host", '/home/example/key folder/my key'),
    ("darwin", "ssh -i '/Users/Example User/key folder/my key' root@host", '/Users/Example User/key folder/my key'),
])
def test_ssh_key_parsing_uses_client_platform(monkeypatch, platform, text, expected):
    monkeypatch.setattr(sshutil.sys, "platform", platform)
    assert sshutil.parse_ssh_command(text).key == expected


def test_known_hosts_paths_use_native_validation_and_openssh_quotes(tmp_path, monkeypatch):
    home = tmp_path / "Client Home"
    monkeypatch.setattr(sshutil.Path, "home", lambda: home)
    known = home / ".ssh" / "known hosts.txt"
    option = sshutil.check_option(f'UserKnownHostsFile="{known.as_posix()}"')
    with pytest.raises(sshutil.SSHParseError):
        sshutil.check_option(f"UserKnownHostsFile={home.as_posix()}/.ssh/../outside.txt")
    ssh = target.SSHTarget(sshutil.SSHSpec(host="example.invalid", options=[option]))
    args = ssh._base()
    encoded = next(arg for arg in args if arg.startswith("UserKnownHostsFile="))
    assert shlex.split(encoded) == ["UserKnownHostsFile=" + known.as_posix()]
    config = tmp_path / "empty ssh config"
    config.write_text("", encoding="utf-8")
    result = subprocess.run([args[0], "-F", str(config), "-G", *args[1:]], capture_output=True, text=True,
                            encoding="utf-8", errors="replace", timeout=15)
    assert result.returncode == 0, result.stderr
    line = next(line for line in result.stdout.splitlines() if line.startswith("userknownhostsfile "))
    assert known.as_posix() in line


@pytest.mark.parametrize("platform,shell,expected", [
    ("linux", "bash", ["-lc"]),
    ("darwin", "sh", ["-c"]),
])
def test_posix_local_shell_selection(monkeypatch, platform, shell, expected):
    monkeypatch.setattr(target.sys, "platform", platform)
    executable = "/shell path with spaces/" + shell
    monkeypatch.setattr(target.shutil, "which", lambda name: executable if name == shell else None)
    command = "command with quoted paths"
    assert target._local_shell(command) == [executable, *expected, command]


def test_windows_shell_needs_no_powershell_or_bash(monkeypatch):
    monkeypatch.setattr(target.sys, "platform", "win32")
    monkeypatch.setattr(target.shutil, "which", lambda name: pytest.fail("Windows shell must not require another shell"))
    command = '"C:\\Program Files\\tool.exe" "file with spaces"'
    assert target._local_shell(command) == command


def test_native_local_shell_run_and_stream_keep_arguments(tmp_path):
    script = tmp_path / "script folder" / "argument echo.py"
    script.parent.mkdir()
    script.write_text("import json,sys\nprint(json.dumps(sys.argv[1:],ensure_ascii=False),flush=True)\n", encoding="utf-8")
    values = ["file with spaces", "quote's file", "Unicode Ω"]
    arguments = [sys.executable, str(script), *values]
    if sys.platform == "win32":
        command = subprocess.list2cmdline(arguments)
    else:
        command = shlex.join(arguments)
    local = target.LocalTarget()
    result = local.run(command, timeout=30)
    assert result.ok, result.stderr
    assert json.loads(result.stdout) == values
    output = []
    assert local.stream(command, output.append) == 0
    assert json.loads(output[0]) == values


def test_remote_probe_quotes_the_interpreter_and_work_path():
    commands = []
    remote = SimpleNamespace(run=lambda command, **kwargs: (commands.append(command) or target.RunResult(0, "{}", "")))
    python = "/opt/Python environments/worker's python"
    work = "/home/example/training data/$input; keep"
    bootstrap.probe_remote(remote, python=python, path=work)
    assert shlex.split(commands[0]) == [python, "-", "--path", work, "--torch"]


@pytest.mark.parametrize("flavor,root", [(PureWindowsPath, "C:/package root"), (PurePosixPath, "/package root")])
def test_worker_hash_uses_portable_names_and_order(monkeypatch, flavor, root):
    contents = {"alpha/module.py": b"alpha\n", "B.py": b"beta\n"}
    base = SimpleNamespace(root=flavor(root))

    class File:
        def __init__(self, name):
            self.name = name
            self.path = base.root / name
            self.parts = self.path.parts
            self.suffix = self.path.suffix

        def is_file(self):
            return True

        def __lt__(self, other):
            return self.path < other.path

        def relative_to(self, folder):
            return self.path.relative_to(folder.root)

        def read_bytes(self):
            return contents[self.name]

    base.rglob = lambda pattern: [File(name) for name in contents]
    monkeypatch.setattr(bootstrap, "package_dir", lambda: base)
    expected = hashlib.sha256()
    for name in sorted(contents):
        expected.update(name.encode("utf-8") + b"\0" + contents[name])
    assert bootstrap.package_hash() == expected.hexdigest()[:12]


def test_ssh_upload_quotes_remote_paths_and_keeps_archive_names(tmp_path, monkeypatch):
    source = tmp_path / "local package Ω"
    source.mkdir()
    (source / "file with spaces.txt").write_text("payload Ω", encoding="utf-8")
    commands = []

    class Pipe(io.BytesIO):
        def close(self):
            self.payload = self.getvalue()
            super().close()

    pipe = Pipe()
    process = SimpleNamespace(stdin=pipe, stderr=io.BytesIO(), wait=lambda: 0)
    monkeypatch.setattr(target.subprocess, "Popen", lambda args, **kwargs: (commands.append(args) or process))
    remote = target.SSHTarget(sshutil.SSHSpec(host="example.invalid"))
    parent = "/home/example/worker's files/$input; safe"
    name = "uploaded package Ω"
    remote.put(source, parent + "/" + name)
    outer = shlex.split(commands[0][-1])
    assert outer[:2] == ["bash", "-lc"]
    assert shlex.split(outer[2]) == ["mkdir", "-p", "--", parent, "&&", "tar", "xzf", "-", "-C", parent]
    with tarfile.open(fileobj=io.BytesIO(pipe.payload), mode="r:gz") as archive:
        assert archive.extractfile(name + "/file with spaces.txt").read().decode("utf-8") == "payload Ω"


def test_adopt_server_keeps_job_names_with_spaces(tmp_path):
    project = Project.create(tmp_path / "project with spaces")
    project.state.jobs.append({"job_id": "job with spaces", "target_spec": {"kind": "ssh", "host": "old"}})
    remote = SimpleNamespace(run=lambda *a, **kw: target.RunResult(0, "job with spaces\n", ""),
                             to_dict=lambda: {"kind": "ssh", "host": "new"}, label="new server")
    assert JobManager(project).adopt_server(remote, "/worker root with spaces") == ["job with spaces"]
    assert Project.load(project.root).job()["remote_run_dir"] == "/worker root with spaces/runs/job with spaces"


def test_ssh_download_keeps_leading_dash_name_as_an_operand(tmp_path, monkeypatch):
    commands = []

    def capture(args, **kwargs):
        commands.append(args)
        raise RuntimeError("command captured before transfer")

    monkeypatch.setattr(target.subprocess, "Popen", capture)
    remote = target.SSHTarget(sshutil.SSHSpec(host="example.invalid"))
    parent = "/home/example/worker's files"
    with pytest.raises(RuntimeError, match="command captured before transfer"):
        remote.get(parent + "/-results file.json", tmp_path / "downloaded results.json")
    shell = shlex.split(commands[0][-1])
    assert shlex.split(shell[2]) == ["tar", "czf", "-", "-C", parent, "--", "-results file.json"]
