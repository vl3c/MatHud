"""Utility script to regenerate linear algebra test fixtures.

Executes the matrix operations used in the client test suite and prints the
resulting arrays with stable precision. Requires numpy (install with
``python -m pip install numpy`` if it is not already available).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, cast

import numpy as np


def main() -> None:
    np.set_printoptions(linewidth=200, suppress=True)

    a_matrix = np.array(
        [
            [42, -17, 63, -5],
            [-28, 91, -74, 60],
            [39, -56, 81, -13],
            [22, -48, 9, 100],
        ],
        dtype=float,
    )

    b_matrix = np.array(
        [
            [15, -88, 71, 20],
            [-93, 7, -44, 55],
            [61, -36, 29, 90],
            [-14, 66, -53, 77],
        ],
        dtype=float,
    )

    a_plus_b = a_matrix + b_matrix
    a_inverse = np.linalg.inv(a_matrix)
    b_inverse = np.linalg.inv(b_matrix)
    ba_inverse = np.linalg.inv(b_matrix @ a_matrix)
    a_inv_b_inv = a_inverse @ b_inverse

    def _format_float(value: Any) -> str:
        return f"{float(value):.14f}"

    formatter: Dict[str, Callable[[Any], str]] = {"float_kind": _format_float}
    formatter_any = cast(Any, formatter)

    print("A + B =")
    print(a_plus_b.astype(int))

    print("\nA inverse =")
    print(np.array2string(a_inverse, formatter=formatter_any))

    print("\nB inverse =")
    print(np.array2string(b_inverse, formatter=formatter_any))

    print("\nA^{-1} * B^{-1} =")
    print(np.array2string(a_inv_b_inv, formatter=formatter_any))

    print("\n(BA)^{-1} =")
    print(np.array2string(ba_inverse, formatter=formatter_any))

    max_difference: float = float(np.max(np.abs(a_inv_b_inv - ba_inverse)))
    print(f"\nmax |A^{-1}B^{-1} - (BA)^{-1}| = {max_difference:.18e}")


if __name__ == "__main__":
    main()
