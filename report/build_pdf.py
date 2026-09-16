"""Build report/report.pdf from report/report.html with headless Microsoft Edge.

Injects PREREGISTRATION.md verbatim into Appendix A, prints to PDF, then renders every page to
PNG under report/build/pages/ so the layout can be checked without opening a viewer.

    python report/build_pdf.py
"""

import html
import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent
REPO = ROOT.parent
EDGE_CANDIDATES = [
    pathlib.Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
    pathlib.Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
]


def main():
    edge = next((p for p in EDGE_CANDIDATES if p.exists()), None)
    if edge is None:
        print("Microsoft Edge not found; open report/_build.html in a browser and print to PDF.")
        return 1

    source = (ROOT / "report.html").read_text(encoding="utf-8")
    prereg = html.escape((REPO / "PREREGISTRATION.md").read_text(encoding="utf-8"))
    built = ROOT / "_build.html"  # same folder as report.html, so figures/ paths resolve
    built.write_text(source.replace("<!--PREREGISTRATION-->", prereg), encoding="utf-8")

    pdf = ROOT / "report.pdf"
    with tempfile.TemporaryDirectory() as profile:
        subprocess.run([
            str(edge), "--headless=new", "--disable-gpu", f"--user-data-dir={profile}",
            "--no-pdf-header-footer", "--virtual-time-budget=10000",
            "--run-all-compositor-stages-before-draw",
            f"--print-to-pdf={pdf}", built.as_uri(),
        ], check=True, timeout=180, capture_output=True)

    try:
        import fitz  # PyMuPDF
    except ImportError:
        print(f"wrote {pdf}")
        return 0

    pages_dir = ROOT / "build" / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)
    for old in pages_dir.glob("page_*.png"):
        old.unlink()
    doc = fitz.open(pdf)
    for number, page in enumerate(doc, start=1):
        page.get_pixmap(dpi=70).save(pages_dir / f"page_{number:02d}.png")
    tbd = sum(page.get_text().count("TBD") for page in doc)
    print(f"wrote {pdf}: {len(doc)} pages, {tbd} TBD markers remaining")
    return 0


if __name__ == "__main__":
    sys.exit(main())
