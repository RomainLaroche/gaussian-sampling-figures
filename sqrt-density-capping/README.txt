Generic Gaussian square-root density capping study

Open explorer.html, retention.html, matched.html, shape.html or tails.html.
All five interactive figures run offline with inline code and data.

Reproduce:
  uv run --with numpy --with matplotlib python study.py
  uv run --with numpy python build_figures.py
  uv run --with playwright playwright install chromium
  uv run --with playwright python validate_figures.py

All quantities are analytical population calculations. The source pool size
in the HTML is an illustration, not an empirical sample. Thresholds are
calibrated independently to match achieved STD between cap methods.

The square-root cap requires STD factor < sqrt(2) for positive retention.
The source-pool multiplier for equal expected retained count is R_flat/R_sqrt
with each retention evaluated at the same achieved STD.
