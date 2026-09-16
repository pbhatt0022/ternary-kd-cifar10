"""Write report/report_gdocs.html: a self-contained copy for pasting into Google Docs.

Figures are embedded as data URIs so they travel with a copy and paste, the pre-registration is
inlined into Appendix A, and image widths are fixed in pixels to fit a Docs page.

    python report/make_gdocs.py
    then open report/report_gdocs.html in Chrome, Ctrl+A, Ctrl+C, and paste into the Google Doc.
"""

import base64
import html
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent
REPO = ROOT.parent
WIDTH = {"fig4_learning_curves": 560, "fig7_temperature_sweep": 400}  # px; default below
DEFAULT_WIDTH = 620


def main():
    page = (ROOT / "report.html").read_text(encoding="utf-8")
    page = page.replace("<!--PREREGISTRATION-->",
                        html.escape((REPO / "PREREGISTRATION.md").read_text(encoding="utf-8")))

    def embed(match):
        name = match.group(1)
        data = base64.b64encode((ROOT / "figures" / f"{name}.png").read_bytes()).decode()
        width = WIDTH.get(name, DEFAULT_WIDTH)
        return f'<img src="data:image/png;base64,{data}" width="{width}" style="width:{width}px"'

    page, count = re.subn(r'<img src="figures/([a-z0-9_]+)\.png"', embed, page)
    out = ROOT / "report_gdocs.html"
    out.write_text(page, encoding="utf-8")
    print(f"wrote {out}: {count} figures embedded, {out.stat().st_size / 2**20:.1f} MB, "
          f"{page.count('TBD')} TBD markers")


if __name__ == "__main__":
    main()
