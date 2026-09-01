"""Verify the built Tailwind stylesheet actually covers every class the app
renders.

Swapping the Tailwind Play CDN for a compiled stylesheet trades runtime class
generation for build-time scanning, and the failure mode is silent: a class the
scanner missed simply has no rule, so the element renders unstyled with no
error anywhere. That is exactly the kind of regression that is invisible
without looking at the page.

This crawls the running app, extracts every class token from the HTML it
actually serves, and asserts each one is defined in public/static/app.css.
Anything missing is printed and the script exits non-zero.

Run the app first, then:
    python scripts/check_css_coverage.py [--base http://127.0.0.1:8130]
"""

from __future__ import annotations

import argparse
import re
import sys
import urllib.request
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
CSS_PATH = BASE_DIR / "public" / "static" / "app.css"

# Classes that are never Tailwind utilities — hand-written CSS in base.html's
# <style> block, or hooks used only by JavaScript/CSS selectors.
NON_TAILWIND = {
    "dark",
    "theme-switching",
    "theme-ucl",
    "no-scrollbar",
    "hero-band",
    "zone-relegation",
    "zone-playoff",
    "group",
    "peer",
    "tab-btn",  # JS hook for the competition page's tab switcher, not a utility
}

# Tailwind emits variants as escaped compound selectors (`.sm\:flex:hover`),
# so a rendered "sm:flex" must be looked up as its base utility plus variant.
VARIANT_RE = re.compile(r"^(?:[a-z0-9-]+:)*")


def rendered_classes(base: str, paths: list[str]) -> set[str]:
    found: set[str] = set()
    for path in paths:
        with urllib.request.urlopen(base + path, timeout=30) as resp:
            html = resp.read().decode("utf-8", "replace")
        for attr in re.findall(r'class="([^"]*)"', html):
            for token in attr.split():
                if token and "{{" not in token:
                    found.add(token)
    return found


_HEX_ESCAPE = re.compile(r"\\([0-9a-fA-F]{1,6})[ ]?")


def css_selectors(css: str) -> set[str]:
    """Every class name defined in the stylesheet, with CSS escapes resolved.

    Hand-parsed rather than regex'd because CSS identifier escaping has two
    forms and both appear in Tailwind's output: a backslash before a literal
    character (`.sm\\:flex`, `.w-\\[9rem\\]`) *and* a hexadecimal code point
    with an optional trailing space (`.transition-\\[max-height\\2c opacity\\]`
    — that is a comma). Naively stripping backslashes mangles the second form,
    and letting a match run past an unescaped '.' merges compound selectors
    like `.\\[\\&\\.htmx-request\\]\\:block.htmx-request` into one bogus name.
    Both mistakes report classes as missing when they are actually present.
    """
    names: set[str] = set()
    for block in re.findall(r"([^{}]*)\{", css):
        i = 0
        while i < len(block):
            if block[i] != ".":
                i += 1
                continue
            i += 1
            buf: list[str] = []
            while i < len(block):
                ch = block[i]
                if ch == "\\":
                    hex_match = _HEX_ESCAPE.match(block, i)
                    if hex_match:
                        buf.append(chr(int(hex_match.group(1), 16)))
                        i = hex_match.end()
                    else:
                        buf.append(block[i + 1])
                        i += 2
                    continue
                if ch.isalnum() or ch in "-_":
                    buf.append(ch)
                    i += 1
                    continue
                break  # unescaped delimiter ends this class name
            if buf:
                names.add("".join(buf))
    return names


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://127.0.0.1:8130")
    parser.add_argument(
        "--paths",
        nargs="*",
        default=[
            "/",
            "/accuracy",
            "/accuracy/tracked",
            "/compare",
            "/competition/premier-league",
            "/competition/bundesliga",
            "/competition/eredivisie",
            "/competition/champions-league",
            "/competition/azerbaijan-premyer-liqa",
        ],
    )
    args = parser.parse_args()

    if not CSS_PATH.exists():
        print(f"missing {CSS_PATH} — run `npm run build:css` first")
        return 1

    css = CSS_PATH.read_text(encoding="utf-8")
    defined = css_selectors(css)
    used = rendered_classes(args.base, args.paths)

    missing = set()
    for token in used:
        if token in NON_TAILWIND or token in defined:
            continue
        # A variant-prefixed utility ("sm:flex", "dark:bg-gray-900",
        # "group-open:rotate-90") is emitted under its full escaped name, which
        # the unescaping above already normalises — so reaching here means it
        # genuinely has no rule. Check the bare utility too, in case only the
        # variant form differs.
        bare = VARIANT_RE.sub("", token, count=1)
        if bare and bare in defined:
            continue
        missing.add(token)

    print(f"pages crawled     : {len(args.paths)}")
    print(f"classes rendered  : {len(used)}")
    print(f"classes in CSS    : {len(defined)}")
    print(f"missing from CSS  : {len(missing)}")
    if missing:
        print("\nThese render but have no rule in the built stylesheet:")
        for token in sorted(missing):
            print(f"  {token}")
        print("\nAdd them to tailwind.config.js's safelist (or fix the template),")
        print("then re-run `npm run build:css`.")
        return 1

    print("\nOK: every rendered class is covered by the built stylesheet.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
