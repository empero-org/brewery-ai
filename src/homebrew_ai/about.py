"""Who made Homebrew and where to find them.

Every credit line, link and contact address Homebrew prints or writes into a
model card comes from here, so a change of URL is a one-line edit.
"""

from __future__ import annotations

from homebrew_ai import __version__

PRODUCT = "Homebrew"
DIST_NAME = "homebrew-ai"
VERSION = __version__

ORG = "Empero"
ORG_TAGLINE = "Independent AI research lab · Open by default · Built in Germany"
WEBSITE = "https://empero.org"
GITHUB_ORG = "https://github.com/empero-org"
REPO_URL = "https://github.com/empero-org/homebrew-ai"
HF_ORG = "https://huggingface.co/empero-ai"
CONTACT = "hello@empero.org"

# Tags every model card gets, so brews are discoverable on the Hub.
HUB_TAGS = ["homebrew-ai", "empero"]

CREDIT_LINE = f"Brewed with [{PRODUCT}]({REPO_URL}) by [{ORG}]({WEBSITE})"

BANNER = r"""
 _   _                      _
| | | | ___  _ __ ___   ___| |__  _ __ _____      __
| |_| |/ _ \| '_ ` _ \ / _ \ '_ \| '__/ _ \ \ /\ / /
|  _  | (_) | | | | | |  __/ |_) | | |  __/\ V  V /
|_| |_|\___/|_| |_| |_|\___|_.__/|_|  \___| \_/\_/
"""


def user_agent() -> str:
    return f"{DIST_NAME}/{VERSION} (+{REPO_URL})"
