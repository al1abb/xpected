"""Guards the one place the Tailwind build cannot see for itself.

Tailwind scans template SOURCE, so `h-{{ size }} w-{{ size }}` in
app/templates/_badge.html is invisible to it — those utilities exist only
because tailwind.config.js safelists them. A new badge size added to a
template would then render with no height or width at all, silently, with
nothing failing anywhere. This makes that fail loudly instead.
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

BASE_DIR = Path(__file__).resolve().parent.parent
TEMPLATES = BASE_DIR / "app" / "templates"
TAILWIND_CONFIG = BASE_DIR / "tailwind.config.js"

BADGE_CALL = re.compile(r"\b(?:team_badge|competition_badge)\s*\(([^)]*)\)")
MACRO_DEFAULT = re.compile(r"macro\s+(?:team_badge|competition_badge)\([^)]*size\s*=\s*(\d+)")


def _sizes_used() -> set[int]:
    sizes: set[int] = set()
    for path in TEMPLATES.rglob("*.html"):
        text = path.read_text(encoding="utf-8")
        sizes.update(int(m) for m in MACRO_DEFAULT.findall(text))
        for args in BADGE_CALL.findall(text):
            # Size is the trailing positional argument, when given.
            last = args.rsplit(",", 1)[-1].strip()
            if last.isdigit():
                sizes.add(int(last))
    return sizes


def _safelisted() -> set[str]:
    text = TAILWIND_CONFIG.read_text(encoding="utf-8")
    block = text.split("safelist:", 1)[1].split("]", 1)[0]
    return set(re.findall(r"'([^']+)'", block))


def test_every_badge_size_used_in_templates_is_safelisted():
    safelist = _safelisted()
    used = _sizes_used()
    assert used, "found no badge sizes at all — the call-site regex has drifted"

    missing = sorted(
        f"h-{n}/w-{n}" for n in used if f"h-{n}" not in safelist or f"w-{n}" not in safelist
    )
    assert not missing, (
        f"Badge sizes rendered but not safelisted: {missing}. "
        f"Add them to tailwind.config.js and re-run `npm run build:css`, "
        f"or those badges render with no width/height."
    )


def test_safelist_has_no_stale_entries():
    """Not a correctness problem, just dead weight in the stylesheet — but it
    also flags a size that was removed from the UI and might indicate a
    half-finished change."""
    used = _sizes_used()
    stale = sorted(
        entry
        for entry in _safelisted()
        if re.fullmatch(r"[hw]-(\d+)", entry) and int(entry.split("-")[1]) not in used
    )
    assert not stale, f"safelisted but no longer used by any badge call: {stale}"
