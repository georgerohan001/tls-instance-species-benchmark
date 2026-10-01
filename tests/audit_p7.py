"""One-shot P7 audit for the public TLS repo."""
from __future__ import annotations

import py_compile
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    bad = []
    text_ext = {".py", ".md", ".yaml", ".yml", ".txt"}
    roots = [
        REPO / "src",
        REPO / "scripts",
        REPO / "tests",
        REPO / "configs",
        REPO / "documentation",
    ]
    for root in roots:
        for p in root.rglob("*"):
            if not p.is_file() or p.suffix.lower() not in text_ext:
                continue
            t = p.read_text(encoding="utf-8", errors="replace")
            needles = [
                "Users" + "\\georg",
                "Desktop" + "\\a MSc",
                "New" + "_Trees",
                "GROUND" + "_DIR",
                "C:" + "\\Users",
            ]
            for k in needles:
                if k in t:
                    bad.append((str(p.relative_to(REPO)), k))
    # Ignore this audit file's own needle literals if any remain
    bad = [b for b in bad if not b[0].endswith("audit_p7.py")]
    print("OVERFIT", bad or "NONE")
    if bad:
        raise SystemExit(1)

    for rel in [
        "src/seg/analyze_per_tree.py",
        "src/paths.py",
        "src/seg/build_aligned.py",
    ]:
        t = (REPO / rel).read_text(encoding="utf-8")
        if re.search(r"site_id\s*==\s*['\"](ettenheim|mathislewald)", t, re.I):
            raise SystemExit(f"SPECIALCASE {rel}")
    print("specialcase OK")

    glued = []
    for p in (REPO / "src").rglob("*.py"):
        t = p.read_text(encoding="utf-8", errors="replace")
        if "tree_(\\d+)" in t and "(?:_(\\d+))?" not in t and "re.search" in t:
            glued.append(str(p.relative_to(REPO)))
    print("GLUED", glued or "NONE")

    req = (REPO / "requirements.txt").read_text(encoding="utf-8")
    assert "shapely" in req
    print("requirements OK")

    for p in (REPO / "documentation").glob("*"):
        t = p.read_text(encoding="utf-8", errors="replace")
        for k in ["86.3", "21.5", "N_GT=72"]:
            if k in t:
                raise SystemExit(f"STALE {p.name} {k}")
        if "¤ä" in t:
            raise SystemExit(f"MOJIBAKE {p.name}")
    print("docs OK")

    for rel in [
        "src/seg/build_aligned.py",
        "src/seg/analyze_per_tree.py",
        "src/paths.py",
        "src/seg/run_metrics.py",
        "scripts/analyze_per_tree.py",
        "scripts/build_aligned.py",
    ]:
        py_compile.compile(str(REPO / rel), doraise=True)
    print("compile OK")


if __name__ == "__main__":
    main()
