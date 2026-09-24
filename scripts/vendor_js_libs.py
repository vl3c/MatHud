"""Download and verify the third-party browser libraries served from static/vendor/.

MatHud serves Brython, math.js, nerdamer, MathJax and the Inter font from the
repository so the app works fully offline (for example with LocalAgent).
Every file is pinned below by version, upstream URL and SHA-256.

Usage:
    python scripts/vendor_js_libs.py          # download missing or changed files, then verify
    python scripts/vendor_js_libs.py --check  # verify existing files only (no network access)

To upgrade a library, change its version and URLs, set the new hashes, run the
script, and review the page in the browser (plus the client test suite).
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

VENDOR_DIR = Path(__file__).resolve().parent.parent / "static" / "vendor"
ALLOWED_URL_PREFIXES = ("https://cdn.jsdelivr.net/", "https://cdnjs.cloudflare.com/")
# Hand-maintained files that live in static/vendor/ but are not downloaded.
LOCAL_FILES = (".gitattributes", "LICENSES.md", "inter/5.3.0/inter.css")
DOWNLOAD_TIMEOUT_SECONDS = 60


@dataclass(frozen=True)
class VendorFile:
    """One pinned upstream file and where it is stored under static/vendor/."""

    url: str
    dest: str
    sha256: str


@dataclass(frozen=True)
class VendorLibrary:
    """A pinned library version and its files."""

    name: str
    version: str
    license: str
    files: tuple[VendorFile, ...]


def _files(base_url: str, dest_dir: str, hashes: dict[str, str]) -> tuple[VendorFile, ...]:
    """Build VendorFile entries for paths relative to both base_url and dest_dir."""
    return tuple(VendorFile(f"{base_url}/{path}", f"{dest_dir}/{path}", sha) for path, sha in hashes.items())


_MATHJAX = "https://cdn.jsdelivr.net/npm/mathjax@3.2.2"
_MATHJAX_TEX_EXT = "es5/input/tex/extensions"
_MATHJAX_FONTS = "es5/output/chtml/fonts/woff-v2"
_INTER = "https://cdn.jsdelivr.net/npm/@fontsource-variable/inter@5.3.0"

LIBRARIES: tuple[VendorLibrary, ...] = (
    VendorLibrary(
        name="Brython",
        version="3.12.5",
        license="BSD-3-Clause",
        files=_files(
            "https://cdnjs.cloudflare.com/ajax/libs/brython/3.12.5",
            "brython/3.12.5",
            {
                "brython.min.js": "623ccd79286272280f0d016e75042a8e5739f83ea36907661a6a0f9cc1c6fed0",
                "brython_stdlib.min.js": "38e41f56562edae31a320c98d21d911133fa02c2d1af7861a0400c7d99169c51",
            },
        )
        + _files(
            "https://cdn.jsdelivr.net/gh/brython-dev/brython@3.12.5",
            "brython/3.12.5",
            {"LICENCE.txt": "e3f7e1ea7ac06a4ddb92e476bba7a37e467d3c5937082279fcaf4fe6b73d00c8"},
        ),
    ),
    VendorLibrary(
        name="math.js",
        version="14.5.2",
        license="Apache-2.0",
        files=_files(
            "https://cdnjs.cloudflare.com/ajax/libs/mathjs/14.5.2",
            "mathjs/14.5.2",
            {"math.min.js": "8aa663175b2016b2c66645f46d32ef5bcaf221be0a2aa945aaa7907a43be12ab"},
        )
        + _files(
            "https://cdn.jsdelivr.net/npm/mathjs@14.5.2",
            "mathjs/14.5.2",
            {
                "LICENSE": "3b0f65a9308588e12b619b2f05c788b4a5f41220006ce563b826b0feb489fd26",
                "NOTICE": "4ffe61bf0e3474513ef782fc9678d83325120eab1ddfa0dc375f89dfa38f1b68",
            },
        ),
    ),
    VendorLibrary(
        name="nerdamer",
        version="1.1.13",
        license="MIT",
        files=_files(
            "https://cdn.jsdelivr.net/npm/nerdamer@1.1.13",
            "nerdamer/1.1.13",
            {
                "nerdamer.core.js": "cfabae9fd3adc5e751d42f6614c677c147a09947f84155722f27399462b9607d",
                "Algebra.js": "b5d74ec1c6269f0c08cf3b61f37fa8ba2e9969f49a81468871deafad2b435435",
                "Calculus.js": "5a606f0cb473cc68fbccb707d59d08bd17e94bfe01d1b20e2be935ce367e88d7",
                "Solve.js": "9f13af4aba5e85127cfe1ee304dd1497db2aa1a148e81dc956d621a15f8dbd26",
                "Extra.js": "39c68d04fd6de1bc0ff241bb882584a483855191a5104a198d85908b61c2393f",
                "license.txt": "45bf0020b3c5923c88528825462bdf1edc9bc8ed53a87a00f71ff92969d721a6",
            },
        ),
    ),
    VendorLibrary(
        name="MathJax",
        version="3.2.2",
        license="Apache-2.0",
        # The page loads tex-mml-chtml.js; it lazy-loads TeX extensions through
        # autoload/require and the CHTML woff-v2 fonts relative to its own URL.
        files=_files(
            _MATHJAX,
            "mathjax/3.2.2",
            {
                "LICENSE": "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30",
                "es5/tex-mml-chtml.js": "300480069078b5892d2363a2b65e2dfbbf30fe5c80f83edbfecf4610fd093862",
            },
        )
        + _files(
            f"{_MATHJAX}/{_MATHJAX_TEX_EXT}",
            f"mathjax/3.2.2/{_MATHJAX_TEX_EXT}",
            {
                "action.js": "37b44965893dafb424811f3a78718aba2cfd450ce2a9344b53a5a7e24f2874a7",
                "ams.js": "624d59b8b8585d3d69a8106b87bac62f8a2495b6caea7906a258ce1f15ff1ea4",
                "amscd.js": "7dc7256ebc5822379cdf9f504559309d97c5f26e09e2150d25dd46e35a6acf94",
                "autoload.js": "7656bc471cb75a43167f6bdeb284c95f323317893d26df4ceaa676d409ac9dbf",
                "bbox.js": "73d208a3425ed27dea7ed61f5984b43accfba8949bad7077d3eac47e668ba764",
                "boldsymbol.js": "d6771fee0772db2657796c8d0e20e1878bb3237f6d3ed1e828e1834a4ff743ca",
                "braket.js": "2abb032ca0eaff9fb2322a28b5d14ba058e7934d3e56974c55f574059e907a4c",
                "bussproofs.js": "47c46e011949f0700ed02698b1a7fc243cc0dfa9529e49935e45ee5e09299bda",
                "cancel.js": "0d263ecf2671d1cc13135b8b4afcdca4d40151a9aea51b69402e68adde4cf8c9",
                "cases.js": "9c11e0644dabdb5adf9065adef7f6224340eaecb96e48a7ce9a27de3c7056f3c",
                "centernot.js": "a1e4be1e20d190a553c74796054335f5f7790414deb966cdc8a78e941a28a351",
                "color.js": "7fdc8f7cda1630f35ccd579379ba04add9dec50c66cebbc05a4b129027b32435",
                "colortbl.js": "d9c0e39634f43f0e29c6d6e82a3624e09f9835747fae5ba56156660fd4aab79b",
                "colorv2.js": "30c09ca19b5d778c1718c47ee3d76b0a301dd2040f11f0b4ae3b2d59d67ffe21",
                "configmacros.js": "a6d4604cbd3a7e7b9691eb108291ddd0093b9750355357478c6b637aad0d5419",
                "empheq.js": "f425b835aeb4e4373b6176b622590d72692f1a85f05018e1035fbfcfc3be1842",
                "enclose.js": "8804f756f369946919c388f587f80cf2d1fd9151f76ed37570d01af8d7fb3f4c",
                "extpfeil.js": "d8d5ecd75abc4306a6d589442dfa66a4345dee1288347591bddaf19551388163",
                "gensymb.js": "9f55af256604b89a7980b9f903c6f33ce3c2d0e279fd347b9a433fe4c58de794",
                "html.js": "e3647c848f26ad228d07043f1e63c56a646da9d20a6c0f180a4067fa9e10facc",
                "mathtools.js": "04272d69a0b64783cd5b66820d6f8220ee510409d0b12c38626404be9ed9689c",
                "mhchem.js": "aec2e1e1c423129cf3d2fe86980642335127d1161734ffe47dd99ed676038627",
                "newcommand.js": "a41669a4ae924ab83cbc3d08f95ae90490a33cd649ed865e83675011b8708ab9",
                "noerrors.js": "9317a33b00e114e679bb3cf07df06102342131412a3daced25b733da334f2ccd",
                "noundefined.js": "b90ecf838a6f5363b260a4dd4e939812abaf062f91234fe20fd63d1c7e88a43f",
                "physics.js": "83d5ad43ea478baf855e0ef9456d63d2d735d10c78ab7544019e0cb0b8083a83",
                "require.js": "d508f823c2cf39509bdd088f6dd17fde20b481a8b1d702aa24061a00f38006c3",
                "setoptions.js": "a259b86c9f49fea08abbc05bf4e5ed510e65870b9d7d127404e8486493b8ad40",
                "tagformat.js": "a1407f0ba6abddf53ed3048aaeec7d5290dfa213878181679f4a62c604515784",
                "textcomp.js": "6eaa4c938145a6ace042806b2d0c710c8d4d806386c97e9d0ec0c6d6e5fc30cf",
                "textmacros.js": "cd783f048f64f6a3750d716af0e10ed2f6539cd181dacf1204e323e2e41bd640",
                "unicode.js": "903e7d0d11d5d639eb13931fd3da78d49bbf9c59b57e82a945da9fb7b710137d",
                "upgreek.js": "4335d722671b77186d6490e326adad5f727ed2db954deafece2d10803cb651a1",
                "verb.js": "fdaface3c13d4f2a9e3013bf986fca04d63175c69eec99b17705f006a6528dc7",
            },
        )
        + _files(
            f"{_MATHJAX}/{_MATHJAX_FONTS}",
            f"mathjax/3.2.2/{_MATHJAX_FONTS}",
            {
                "MathJax_AMS-Regular.woff": "3de784d07b9fa8f104c10928a878ee879cf3305cae5195cba663c9c2bb0195eb",
                "MathJax_Calligraphic-Bold.woff": "af04542b29eaac04550a140c5f1760a649783989426f2540855bf4157819367d",
                "MathJax_Calligraphic-Regular.woff": "26683bf201fb258a2237d9754616de9d4ecf4cc1cd39dd1902476df7d75f1d16",
                "MathJax_Fraktur-Bold.woff": "721921bab0d001ebff0206c24cb5de6ca136467bf1843a7aa32030ba061d1e92",
                "MathJax_Fraktur-Regular.woff": "870673df72e70f87c91a5a317d558c2c3b54392264ad79bbadc6d424ae8765fe",
                "MathJax_Main-Bold.woff": "88b98cad3688915e50da55553ff6ad185e0dce134b47f176e91b100f8a9b175c",
                "MathJax_Main-Italic.woff": "355254db9ca10a09a3b5f0929d74eb4670f44fbe864c07526a06213e0a0caf6c",
                "MathJax_Main-Regular.woff": "1cb1c39ea642f26a4dfed230b4aea1c3c218689421f6e9c0a7c1811693c4fa07",
                "MathJax_Math-BoldItalic.woff": "8ea8dbb1b02e6f730f55b4cb5d413b785b9f5c39807d0a0fa7da206a0824a457",
                "MathJax_Math-Italic.woff": "a009bea404f7a500ded48f8b9ad9cf16e12504b3195dd9e25975289b8256b0f0",
                "MathJax_Math-Regular.woff": "c01d3321e89b403c4b811aa153c4e618eda3421f92d8a072a02c8d190782a191",
                "MathJax_SansSerif-Bold.woff": "32792104b5ef69eded905b6d1598ed05d8087684d38e7a94d52e3c38ba16f47e",
                "MathJax_SansSerif-Italic.woff": "fc6ddf5df402b263cfb158aed8e89972542c34b719cd87b1db30461985f7bd5b",
                "MathJax_SansSerif-Regular.woff": "b418136e3b384baaadecb70bd3c48a8da9825210130b2897db808c44efc883cb",
                "MathJax_Script-Regular.woff": "af96f67d7accf5fd2a4a682d0b9f8b339f8ea6fe34c310c1694c8ba7f6ddc96f",
                "MathJax_Size1-Regular.woff": "c49810b53ecc0d87d8028762c518924197dd9d3f905b08f99ea241301085b9cb",
                "MathJax_Size2-Regular.woff": "30e889b58cbc51adfbb038ab1a96dc4025aa3542a3cff7712fc55ece510675e2",
                "MathJax_Size3-Regular.woff": "5cda41563a095bd70c78e2dda13d0f8cb922c71c90fffd0a0044c65173a66e83",
                "MathJax_Size4-Regular.woff": "3bc6ecaae7ecf6f8d7f8ea07bddbaca8601e2cb6e89d6aef58e78d9f6d8a398f",
                "MathJax_Typewriter-Regular.woff": "c56da8d69f1a0208b8e0703656c3264b6dd748bd452524c82b0385b60f6a68c1",
                "MathJax_Vector-Bold.woff": "36e0d72d8a7afc696a3e7a5c7369807634640de9b02257bca447bdf126221a27",
                "MathJax_Vector-Regular.woff": "72bc573386dd1d48c5bbd286302ca9e3400c5eb0f298e7cfbf945b9d08fc688f",
                "MathJax_Zero.woff": "481e39042508ae313a60618af1e37146ab93e9324c98e4c78b8f17fe55d41e0b",
            },
        ),
    ),
    VendorLibrary(
        name="Inter (Fontsource variable)",
        version="5.3.0",
        license="OFL-1.1",
        # Byte-identical to the Inter v20 variable woff2 files Google Fonts serves;
        # inter.css (hand-maintained) declares them as font-family 'Inter'.
        files=_files(
            _INTER,
            "inter/5.3.0",
            {
                "LICENSE": "3b0a5fca3d17942cde889069889dedbbbd075e9b599968c82a95f4d944e9b345",
                "files/inter-cyrillic-ext-wght-normal.woff2": "ca157063339ac4ad418f214f3abfed119b0798ab4d377386ce5c9e5a7a435ebd",
                "files/inter-cyrillic-wght-normal.woff2": "71d5ee93cc1e9f1d520a3a8b66456de18c7879d8df09d57fcd2eaff75fef0075",
                "files/inter-greek-ext-wght-normal.woff2": "6e9e020a25f9b56d418f2c085b1d3c09725a4da23fe693a5b463064606732190",
                "files/inter-greek-wght-normal.woff2": "1be3448e292fbf05ffe176fe1e43f135013d50b1e7d324ad1a558f623d3bb6f6",
                "files/inter-vietnamese-wght-normal.woff2": "5c66f9e07e90c6d4ac4922cc68d60de26c17b1858e677fb5e603fce3952b3ff2",
                "files/inter-latin-ext-wght-normal.woff2": "34b9c504cab7a73e37b746343a449132e56cf7b5481af2cb81dc74dcff25c956",
                "files/inter-latin-wght-normal.woff2": "3100e775e8616cd2611beecfa23a4263d7037586789b43f035236a2e6fbd4c62",
            },
        ),
    ),
)


def iter_files() -> list[VendorFile]:
    """Return every pinned file across all libraries."""
    return [f for lib in LIBRARIES for f in lib.files]


def sha256_of(path: Path) -> str:
    """Return the hex SHA-256 digest of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(vendor_dir: Path = VENDOR_DIR) -> list[str]:
    """Verify every pinned file exists with the expected hash and nothing unexpected is present.

    Returns a list of human-readable problems (empty when everything matches).
    """
    problems: list[str] = []
    for f in iter_files():
        path = vendor_dir / f.dest
        if not path.is_file():
            problems.append(f"missing: {f.dest}")
        elif sha256_of(path) != f.sha256:
            problems.append(f"hash mismatch: {f.dest}")
    for name in LOCAL_FILES:
        if not (vendor_dir / name).is_file():
            problems.append(f"missing hand-maintained file: {name}")
    expected = {f.dest for f in iter_files()} | set(LOCAL_FILES)
    if vendor_dir.is_dir():
        for path in sorted(vendor_dir.rglob("*")):
            rel = path.relative_to(vendor_dir).as_posix()
            if path.is_file() and rel not in expected:
                problems.append(f"unexpected file: {rel}")
    return problems


def _download(url: str) -> bytes:
    if not url.startswith(ALLOWED_URL_PREFIXES):
        raise ValueError(f"refusing to download from non-allowlisted URL: {url}")
    with urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response:
        data: bytes = response.read()
    return data


def fetch(vendor_dir: Path = VENDOR_DIR) -> list[str]:
    """Download files that are missing or do not match their pinned hash.

    A downloaded file is written only when its hash matches the manifest.
    Returns a list of problems (empty on success).
    """
    problems: list[str] = []
    for f in iter_files():
        path = vendor_dir / f.dest
        if path.is_file() and sha256_of(path) == f.sha256:
            continue
        print(f"downloading {f.url}")
        data = _download(f.url)
        digest = hashlib.sha256(data).hexdigest()
        if digest != f.sha256:
            problems.append(f"hash mismatch for {f.url}: got {digest}, expected {f.sha256}")
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return problems


def total_size(vendor_dir: Path = VENDOR_DIR) -> int:
    """Return the combined size in bytes of the pinned files present on disk."""
    return sum((vendor_dir / f.dest).stat().st_size for f in iter_files() if (vendor_dir / f.dest).is_file())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="verify existing files only; never download")
    args = parser.parse_args(argv)

    problems = [] if args.check else fetch()
    problems += check()
    for problem in problems:
        print(problem, file=sys.stderr)
    if problems:
        return 1
    for lib in LIBRARIES:
        size = sum((VENDOR_DIR / f.dest).stat().st_size for f in lib.files)
        print(f"ok  {lib.name} {lib.version} ({lib.license}): {len(lib.files)} files, {size / 1024:.0f} KiB")
    print(f"total: {total_size() / (1024 * 1024):.2f} MiB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
