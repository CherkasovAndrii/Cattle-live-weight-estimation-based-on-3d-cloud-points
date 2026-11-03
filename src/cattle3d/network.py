"""How a phone reaches the demo running on this computer.

Owns: the computer's LAN addresses ranked by how likely a phone can reach
them, and a temporary public https address through a Cloudflare quick tunnel
(downloading ``cloudflared`` once when it is missing). Must not know about
herds, models or HTTP routes.
"""

from __future__ import annotations

import atexit
import os
import platform
import re
import shutil
import socket
import stat
import subprocess
import tarfile
import threading
import time
import urllib.request
from pathlib import Path


def _lan_rank(address: str) -> int:
    """Home and office Wi-Fi almost always hand out 192.168.x.x, then 10.x.x.x.
    172.16-31.x.x is where WSL, Hyper-V and Docker put their virtual adapters,
    and anything else (Radmin VPN 26.x, Hamachi 25.x, CGNAT 100.64.x) is a VPN
    a phone cannot reach. The QR code encodes the best-ranked address."""
    a, b = (int(x) for x in address.split(".")[:2])
    if a == 192 and b == 168:
        return 0
    if a == 10:
        return 1
    if a == 172 and 16 <= b <= 31:
        return 2
    return 3


def lan_addresses() -> list[str]:
    """This computer's IPv4 addresses a phone could reach, best first (see
    ``_lan_rank``; ties keep the address on the default route first). No
    packet is sent. Virtual adapters show up too, so all of them are offered."""
    found: list[str] = []
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))
            found.append(s.getsockname()[0])
    except OSError:
        pass
    try:
        infos = socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)
    except OSError:
        infos = []
    for info in infos:
        address = str(info[4][0])
        if address not in found:
            found.append(address)
    usable = [a for a in found if not a.startswith(("127.", "169.254.", "0."))]
    return sorted(usable, key=_lan_rank) or ["127.0.0.1"]


CLOUDFLARED_URL = "https://github.com/cloudflare/cloudflared/releases/latest/download/"


def cloudflared_path(cache: Path) -> str | None:
    """``cloudflared`` from PATH, else a copy downloaded once into ``cache``
    from Cloudflare's official GitHub releases. None when neither works."""
    exe = shutil.which("cloudflared")
    if exe:
        return exe
    system, machine = platform.system().lower(), platform.machine().lower()
    arch = "arm64" if machine in ("arm64", "aarch64") else "amd64"
    name = {"windows": f"cloudflared-windows-{arch}.exe", "linux": f"cloudflared-linux-{arch}",
            "darwin": f"cloudflared-darwin-{arch}.tgz"}.get(system)
    if name is None:
        return None
    target = cache / ("cloudflared.exe" if system == "windows" else "cloudflared")
    if target.is_file():
        return str(target)
    print(f"  downloading cloudflared ({name}) from github.com/cloudflare once…", flush=True)
    try:
        cache.mkdir(parents=True, exist_ok=True)
        tmp = cache / (name + ".part")
        urllib.request.urlretrieve(CLOUDFLARED_URL + name, tmp)
        if name.endswith(".tgz"):
            with tarfile.open(tmp) as tar:
                member = next(m for m in tar.getmembers() if m.name.endswith("cloudflared"))
                with tar.extractfile(member) as src, open(target, "wb") as dst:  # type: ignore[union-attr]
                    shutil.copyfileobj(src, dst)
            tmp.unlink()
        else:
            os.replace(tmp, target)
        target.chmod(target.stat().st_mode | stat.S_IEXEC)
    except Exception as e:  # no internet, proxy, blocked download: the LAN address still works
        print(f"  could not download cloudflared: {e}", flush=True)
        return None
    return str(target)


def start_tunnel(port: int, cache: Path, timeout_s: float = 40.0) -> tuple[str, "subprocess.Popen[str]"] | None:
    """A public https address for this computer through a Cloudflare quick
    tunnel (no account), so a phone on any network, mobile data included,
    reaches the demo. Uses ``cloudflared_path``; None when it is missing or
    does not answer in time."""
    exe = cloudflared_path(cache)
    if exe is None:
        return None
    proc = subprocess.Popen(
        [exe, "tunnel", "--no-autoupdate", "--url", f"http://127.0.0.1:{port}"],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace",
    )
    found: list[str] = []

    def read() -> None:
        assert proc.stderr is not None
        for line in proc.stderr:  # keep draining so the pipe never fills
            match = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", line)
            if match and not found:
                found.append(match.group(0))

    threading.Thread(target=read, daemon=True).start()
    deadline = time.monotonic() + timeout_s
    while not found and proc.poll() is None and time.monotonic() < deadline:
        time.sleep(0.2)
    if not found:
        proc.terminate()
        return None
    atexit.register(proc.terminate)
    return found[0], proc
