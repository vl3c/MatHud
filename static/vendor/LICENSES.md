# Vendored third-party libraries

MatHud serves these browser libraries from `static/vendor/` so the app works
without network access. Files are unmodified upstream copies, pinned by version
and SHA-256 in `scripts/vendor_js_libs.py` (run it with `--check` to verify).
Each library's own license file is kept next to it.

| Library | Version | License | Upstream | License file |
|---|---|---|---|---|
| Brython | 3.12.5 | BSD-3-Clause | https://brython.info (cdnjs `brython/3.12.5`) | `brython/3.12.5/LICENCE.txt` |
| math.js | 14.5.2 | Apache-2.0 | https://mathjs.org (cdnjs `mathjs/14.5.2`) | `mathjs/14.5.2/LICENSE`, `mathjs/14.5.2/NOTICE` |
| nerdamer | 1.1.13 | MIT | https://nerdamer.com (npm `nerdamer@1.1.13`) | `nerdamer/1.1.13/license.txt` |
| MathJax | 3.2.2 | Apache-2.0 | https://www.mathjax.org (npm `mathjax@3.2.2`) | `mathjax/3.2.2/LICENSE` |
| Inter font | 4.x (Fontsource 5.3.0) | SIL OFL 1.1 | https://rsms.me/inter (npm `@fontsource-variable/inter@5.3.0`) | `inter/5.3.0/LICENSE` |

## What is included

- **Brython**: `brython.min.js` and `brython_stdlib.min.js`.
- **math.js**: `math.min.js`.
- **nerdamer**: `nerdamer.core.js` plus the `Algebra`, `Calculus`, `Solve` and `Extra` add-ons.
- **MathJax**: only the `es5/tex-mml-chtml.js` combined component, the TeX
  extensions it can load on demand (`es5/input/tex/extensions/`, loaded by
  `autoload`/`\require`), and the CHTML output fonts (`es5/output/chtml/fonts/woff-v2/`).
  Components that only the MathJax context menu can switch on (SVG output,
  accessibility explorer and speech rules) are not vendored.
- **Inter**: the variable `woff2` subsets. `inter/5.3.0/inter.css` is a small
  hand-written stylesheet (part of MatHud) that declares them as font family
  `Inter`, mirroring the Google Fonts stylesheet the app used before.
