#!/usr/bin/env python3
"""Build index.html from index.template.html + content.yaml.

Usage:
    python tools/build_html.py
"""
import re
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    sys.exit("pip install pyyaml")

try:
    import markdown
except ImportError:
    sys.exit("pip install markdown")

ROOT = Path(__file__).resolve().parent.parent


def to_html(text):
    """Convert Markdown string to HTML, stripping outer <p> for single paragraphs."""
    html = markdown.markdown(str(text).strip(), extensions=["extra"])
    # Strip outer <p>...</p> when there is exactly one paragraph
    if html.count("<p>") == 1:
        m = re.match(r"^<p>(.*)</p>$", html.strip(), re.DOTALL)
        if m:
            return m.group(1)
    return html


def main():
    content = yaml.safe_load((ROOT / "content.yaml").read_text())
    template = (ROOT / "index.template.html").read_text()

    result = template
    for key, value in content.items():
        marker = "{{ " + key + " }}"
        if marker in result:
            result = result.replace(marker, to_html(value))
        else:
            print(f"  warning: key '{key}' not found in template")

    remaining = re.findall(r"\{\{[^}]+\}\}", result)
    if remaining:
        print(f"  warning: unreplaced markers: {remaining}")

    out = ROOT / "index.html"
    out.write_text(result)
    print(f"wrote {out} ({len(result):,} bytes)")


if __name__ == "__main__":
    main()
