"""Parsing the SSH commands GPU providers show, and managing Brewery's SSH key."""

from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath, PureWindowsPath

DEFAULT_KEY = Path.home() / ".ssh" / "brewery_ed25519"
if not DEFAULT_KEY.exists() and (Path.home() / ".ssh" / "homebrew_ed25519").exists():
    DEFAULT_KEY = Path.home() / ".ssh" / "homebrew_ed25519"  # created before the rename; providers already know it
# Flags with a value that Brewery drops (port forwards such as Vast's "-L 8080:localhost:8080", ciphers, ...).
_FLAGS_WITH_ARG = {"-L", "-R", "-D", "-W", "-b", "-c", "-E", "-e", "-m", "-O", "-Q", "-S", "-w", "-B", "-I"}
# Flags that jump through other hosts or load other configs; refused because they can run local commands.
_REFUSED_FLAGS = {"-J", "-F"}
# ``-o`` options Brewery passes on. Everything else (ProxyCommand, LocalCommand, Include, Match, SendEnv,
# forwarding ...) could run commands on this computer or leak data, so it is rejected.
SAFE_OPTIONS = {
    "stricthostkeychecking": None,
    "userknownhostsfile": None,
    "serveraliveinterval": r"\d+",
    "serveralivecountmax": r"\d+",
    "connecttimeout": r"\d+",
    "connectionattempts": r"\d+",
    "tcpkeepalive": r"yes|no",
    "compression": r"yes|no",
    "loglevel": r"[A-Za-z0-9]+",
    "identitiesonly": r"yes|no",
    "addressfamily": r"any|inet|inet6",
    "checkhostip": r"yes|no",
    "hostkeyalias": r"[A-Za-z0-9._:\[\]-]+",
    "pubkeyacceptedkeytypes": r"[A-Za-z0-9@.,+_-]+",
    "pubkeyacceptedalgorithms": r"[A-Za-z0-9@.,+_-]+",
    "hostkeyalgorithms": r"[A-Za-z0-9@.,+_-]+",
    "preferredauthentications": r"[A-Za-z0-9.,_-]+",
}
_HOST_RE = re.compile(r"^[A-Za-z0-9_.:\[\]%-]+$")
_USER_RE = re.compile(r"^[A-Za-z0-9_.][A-Za-z0-9_.-]*$")


class SSHParseError(ValueError):
    pass


@dataclass
class SSHSpec:
    host: str
    user: str = "root"
    port: int = 22
    key: str | None = None
    options: list[str] = field(default_factory=list)
    provider: str | None = None

    @property
    def label(self) -> str:
        return f"{self.user}@{self.host}:{self.port}"

    def to_dict(self) -> dict:
        return {"host": self.host, "user": self.user, "port": self.port, "key": self.key, "options": self.options, "provider": self.provider}


def check_option(option: str) -> str:
    """Validate one ``-o Key=Value`` option against :data:`SAFE_OPTIONS`; returns it normalised."""
    key, sep, value = option.partition("=")
    if not sep:
        key, _, value = option.partition(" ")
    key, value = key.strip(), value.strip()
    rule = SAFE_OPTIONS.get(key.lower(), "refused")
    if rule == "refused":
        raise SSHParseError(f"the SSH option {key!r} is not allowed (Brewery only passes simple connection options)")
    if key.lower() == "stricthostkeychecking":
        ok = value.lower() in ("yes", "no", "accept-new", "ask", "off")
    elif key.lower() == "userknownhostsfile":
        if len(value) >= 2 and value[0] in ("'", '"') and value[-1] == value[0]:
            value = value[1:-1]
        ok = value == "/dev/null" or Path(value).expanduser().resolve().is_relative_to((Path.home() / ".ssh").resolve())
    else:
        ok = re.fullmatch(rule, value) is not None
    if not ok:
        raise SSHParseError(f"the value of SSH option {key!r} is not allowed: {value!r}")
    return f"{key}={value}"


def validate_spec(spec: SSHSpec) -> None:
    """Re-check a stored spec (e.g. from brewery.yaml) before it is handed to ssh."""
    if not _HOST_RE.match(spec.host or "") or spec.host.startswith("-"):
        raise SSHParseError(f"invalid SSH host {spec.host!r}")
    if not _USER_RE.match(spec.user or ""):
        raise SSHParseError(f"invalid SSH user {spec.user!r}")
    if spec.key and str(spec.key).startswith("-"):
        raise SSHParseError("invalid SSH key path")
    for opt in spec.options:
        check_option(opt)


def guess_provider(host: str) -> str | None:
    h = host.lower()
    if "runpod" in h:
        return "runpod"
    if "vast.ai" in h:
        return "vast"
    return None


def parse_ssh_command(text: str) -> SSHSpec:
    """Parse e.g. ``ssh root@1.2.3.4 -p 22022 -i ~/.ssh/key`` or ``root@host:2222``."""
    text = text.strip()
    if not text:
        raise SSHParseError("empty SSH command")
    try:
        lexer = shlex.shlex(text, posix=True)
        lexer.whitespace_split = True
        lexer.commenters = ""
        if sys.platform == "win32":
            lexer.escape = ""  # Backslashes are path separators in Windows shell input.
        tokens = list(lexer)
    except ValueError as exc:
        raise SSHParseError(f"could not read the command: {exc}") from exc
    path_type = PureWindowsPath if sys.platform == "win32" else PurePosixPath
    if tokens and path_type(tokens[0]).name.lower() in ("ssh", "ssh.exe"):
        tokens = tokens[1:]
    user = None
    port = 22
    key = None
    options: list[str] = []
    target = None
    i = 0
    def value(i: int) -> str:
        if i + 1 >= len(tokens):
            raise SSHParseError(f"{tokens[i]} needs a value")
        return tokens[i + 1]

    while i < len(tokens):
        tok = tokens[i]
        try:
            if tok == "-p":
                port = int(value(i))
                i += 2
                continue
            if tok.startswith("-p") and tok[2:].isdigit():
                port = int(tok[2:])
            elif tok in ("-i", "-l", "-o"):
                arg = value(i)
                if tok == "-i":
                    key = arg
                elif tok == "-l":
                    user = arg
                else:
                    options.append(check_option(arg))
                i += 2
                continue
            elif tok[:2] in ("-i", "-l", "-o") and len(tok) > 2:  # attached form: -oKey=value, -lroot
                arg = tok[2:]
                if tok[:2] == "-i":
                    key = arg
                elif tok[:2] == "-l":
                    user = arg
                else:
                    options.append(check_option(arg))
            elif tok[:2] in _REFUSED_FLAGS:
                raise SSHParseError(f"the SSH flag {tok[:2]} is not supported (it uses other hosts or config files); paste the plain 'ssh -p PORT user@host' command")
            elif tok in _FLAGS_WITH_ARG:
                i += 2
                continue
            elif tok[:2] in _FLAGS_WITH_ARG:
                pass  # attached form, e.g. -L8080:localhost:8080
            elif tok.startswith("-"):
                pass  # -t, -A, -v ... irrelevant for us
            elif target is None:
                target = tok
            else:
                raise SSHParseError("please paste only the SSH command (no remote command after the host)")
        except ValueError as exc:
            if isinstance(exc, SSHParseError):
                raise
            raise SSHParseError(f"could not read the port in {tok!r}") from exc
        i += 1
    if target is None:
        raise SSHParseError("no host found in the SSH command")
    if "@" in target:
        u, _, target = target.rpartition("@")
        user = user or u
    if target.count(":") == 1:
        target, _, p = target.partition(":")
        if p.isdigit():
            port = int(p)
    if key:
        key = os.path.expanduser(key)
        if key.startswith("-"):
            raise SSHParseError("the key path must not start with '-'")
    user = user or "root"
    if not _USER_RE.match(user):
        raise SSHParseError(f"that does not look like a user name: {user!r}")
    if not _HOST_RE.match(target) or target.startswith("-"):
        raise SSHParseError(f"that does not look like a host name or IP address: {target!r}")
    if not 0 < port < 65536:
        raise SSHParseError(f"port {port} is out of range")
    spec = SSHSpec(host=target, user=user, port=port, key=key, options=options, provider=guess_provider(target))
    if spec.host.lower() == "ssh.runpod.io":
        raise SSHParseError(
            "That is Runpod's proxy SSH (ssh.runpod.io), which cannot copy files. In the pod's Connect menu, "
            "use 'SSH over exposed TCP' instead (looks like: ssh root@<IP> -p <port> -i ~/.ssh/...)."
        )
    return spec


def ensure_key(path: Path = DEFAULT_KEY) -> tuple[Path, str, bool]:
    """Create an ed25519 key pair for Brewery if missing. Returns ``(path, public_key, created)``."""
    pub = path.with_suffix(path.suffix + ".pub") if path.suffix else Path(str(path) + ".pub")
    created = False
    if not path.exists():
        if not shutil.which("ssh-keygen"):
            raise RuntimeError("ssh-keygen was not found; install OpenSSH first")
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            path.parent.chmod(0o700)
        except OSError:
            pass
        subprocess.run(["ssh-keygen", "-t", "ed25519", "-f", str(path), "-N", "", "-C", "brewery-ai"], check=True, capture_output=True)
        created = True
    return path, pub.read_text(encoding="utf-8").strip(), created


def existing_public_keys() -> list[Path]:
    ssh = Path.home() / ".ssh"
    return sorted(p for p in ssh.glob("*.pub") if p.is_file()) if ssh.is_dir() else []
