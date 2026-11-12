"""Turn docs/remontada.md into docs/remontada.pdf, the file we upload with the form.

    python scripts/export_pdf.py

Needs pandoc (Markdown to HTML) and Microsoft Edge or Google Chrome (HTML to PDF, headless). Works
from any folder. After editing the document (for example the Authors line), run it again and commit
both files.
"""
import argparse
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO_URL = "https://github.com/pyyroo-a/Remontada/blob/main/"
LINK = re.compile(r"\]\((?!https?://|mailto:|#)([^)#]+)(#[^)]*)?\)")
SOURCE = ROOT / "docs" / "remontada.md"
TARGET = ROOT / "docs" / "remontada.pdf"
BROWSERS = ("msedge", "chrome", "google-chrome", "chromium")
EDGE_PATHS = (
    Path("C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"),
    Path("C:/Program Files/Microsoft/Edge/Application/msedge.exe"),
    Path("C:/Program Files/Google/Chrome/Application/chrome.exe"),
    Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
    Path("/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"),
)
STYLE = """
body { font-family: "Segoe UI", Helvetica, Arial, sans-serif; font-size: 10.5pt; line-height: 1.45;
       max-width: 46em; margin: 0 auto; color: #1a1a1a; }
h1 { font-size: 19pt; margin-bottom: 0.2em; }
h2 { font-size: 13.5pt; margin-top: 1.3em; border-bottom: 1px solid #ccc; padding-bottom: 0.15em; }
table { border-collapse: collapse; margin: 0.6em 0; font-size: 9.5pt; }
th, td { border: 1px solid #bbb; padding: 0.25em 0.55em; text-align: left; }
th { background: #f0f0f0; }
code, pre { font-family: Consolas, Menlo, monospace; font-size: 9pt; }
pre { background: #f5f5f5; padding: 0.5em 0.7em; white-space: pre-wrap; }
a { color: #1a4f9c; }
@page { size: A4; margin: 18mm 16mm; }
"""


def absolute_links(markdown, base):
    """Relative links (which only work inside the repo) become links to the file on GitHub, so they
    still work in the PDF instead of pointing at a temporary folder on the machine that built it."""
    def to_github(match):
        path = (Path(base) / match.group(1)).resolve().relative_to(ROOT).as_posix()
        return "]({}{}{})".format(REPO_URL, path, match.group(2) or "")
    return LINK.sub(to_github, markdown)


def find_browser():
    for name in BROWSERS:
        found = shutil.which(name)
        if found:
            return found
    for path in EDGE_PATHS:
        if path.is_file():
            return str(path)
    return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=TARGET)
    args = parser.parse_args(argv)

    pandoc = shutil.which("pandoc")
    if not pandoc:
        raise SystemExit("pandoc not found: install it from https://pandoc.org and run this again")
    browser = find_browser()
    if not browser:
        raise SystemExit("no Edge or Chrome found: install one of them and run this again")

    with tempfile.TemporaryDirectory() as tmp:
        css = Path(tmp) / "style.css"
        css.write_text(STYLE, encoding="utf-8")
        source = Path(tmp) / "remontada.md"
        source.write_text(absolute_links(SOURCE.read_text(encoding="utf-8"), SOURCE.parent), encoding="utf-8")
        html = Path(tmp) / "remontada.html"
        subprocess.run([pandoc, str(source), "--standalone", "--embed-resources", "--css", str(css),
                        "--metadata", "pagetitle=Remontada", "--resource-path", str(SOURCE.parent),
                        "-o", str(html)], check=True)
        out = args.out.resolve()
        out.unlink(missing_ok=True)  # so a browser that writes nothing can never leave the old PDF behind
        subprocess.run([browser, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                        "--print-to-pdf={}".format(out), html.as_uri()], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if not out.is_file() or out.stat().st_size == 0:
        raise SystemExit("the browser did not write {}".format(out))
    print("wrote {} ({} KB)".format(out, out.stat().st_size // 1024))
    return out


if __name__ == "__main__":
    main()
