"""Dense 2-D matrix stored in row-major order in one contiguous block.

Element (r, c) lives at flat index ``r * cols + c``, the layout C and NumPy
use by default. Row-major storage makes walking along a row cache-friendly
(consecutive addresses), and walking down a column strided.

Complexities for an R x C matrix:

    get / set (r, c)       O(1)
    insert_row / delete_row  O(R * C)   rebuild block (O(C*(R-r)) shift in place
                                        would be possible; we rebuild for clarity)
    insert_col / delete_col  O(R * C)   every row's layout changes
    transpose                O(R * C)
    add                      O(R * C)
    multiply (R x C)(C x P)  O(R * C * P)
"""

from typing import Any, Iterable, List, Sequence

from .array import _alloc


class Matrix:
    """Row-major dense matrix with insertion/deletion of rows and columns."""

    def __init__(self, rows: int, cols: int, fill: Any = 0) -> None:
        if rows < 0 or cols < 0:
            raise ValueError("dimensions must be non-negative")
        self.rows = rows
        self.cols = cols
        self._data = _alloc(rows * cols)
        for i in range(rows * cols):
            self._data[i] = fill

    @classmethod
    def from_rows(cls, rows: Sequence[Sequence[Any]]) -> "Matrix":
        r = len(rows)
        c = len(rows[0]) if r else 0
        if any(len(row) != c for row in rows):
            raise ValueError("all rows must have the same length")
        m = cls(r, c)
        for i in range(r):
            for j in range(c):
                m._data[i * c + j] = rows[i][j]
        return m

    # ---- access ----------------------------------------------------------
    def _offset(self, r: int, c: int) -> int:
        if not (0 <= r < self.rows and 0 <= c < self.cols):
            raise IndexError(f"({r}, {c}) out of range for {self.rows}x{self.cols} matrix")
        return r * self.cols + c

    def __getitem__(self, rc) -> Any:
        """``m[r, c]`` in O(1)."""
        r, c = rc
        return self._data[self._offset(r, c)]

    def __setitem__(self, rc, value: Any) -> None:
        r, c = rc
        self._data[self._offset(r, c)] = value

    @property
    def shape(self):
        return (self.rows, self.cols)

    def row(self, r: int) -> List[Any]:
        return [self[r, c] for c in range(self.cols)]

    def to_lists(self) -> List[List[Any]]:
        return [self.row(r) for r in range(self.rows)]

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Matrix) and self.shape == other.shape and \
            all(self._data[i] == other._data[i] for i in range(self.rows * self.cols))

    def __repr__(self) -> str:
        return f"Matrix({self.to_lists()})"

    # ---- structural insertion / deletion --------------------------------
    def insert_row(self, r: int, values: Iterable[Any]) -> None:
        """Insert a row before index ``r`` (0 <= r <= rows). O(R*C)."""
        values = list(values)
        if len(values) != self.cols:
            raise ValueError(f"row must have {self.cols} values")
        if not 0 <= r <= self.rows:
            raise IndexError("row index out of range")
        new = _alloc((self.rows + 1) * self.cols)
        split = r * self.cols
        for i in range(split):
            new[i] = self._data[i]
        for j in range(self.cols):
            new[split + j] = values[j]
        for i in range(split, self.rows * self.cols):
            new[i + self.cols] = self._data[i]
        self._data, self.rows = new, self.rows + 1

    def delete_row(self, r: int) -> None:
        """Remove row ``r``. O(R*C)."""
        if not 0 <= r < self.rows:
            raise IndexError("row index out of range")
        new = _alloc((self.rows - 1) * self.cols)
        k = 0
        for i in range(self.rows):
            if i == r:
                continue
            for j in range(self.cols):
                new[k] = self._data[i * self.cols + j]
                k += 1
        self._data, self.rows = new, self.rows - 1

    def insert_col(self, c: int, values: Iterable[Any]) -> None:
        """Insert a column before index ``c``. O(R*C)."""
        values = list(values)
        if len(values) != self.rows:
            raise ValueError(f"column must have {self.rows} values")
        if not 0 <= c <= self.cols:
            raise IndexError("column index out of range")
        nc = self.cols + 1
        new = _alloc(self.rows * nc)
        for i in range(self.rows):
            for j in range(nc):
                if j < c:
                    new[i * nc + j] = self._data[i * self.cols + j]
                elif j == c:
                    new[i * nc + j] = values[i]
                else:
                    new[i * nc + j] = self._data[i * self.cols + j - 1]
        self._data, self.cols = new, nc

    def delete_col(self, c: int) -> None:
        """Remove column ``c``. O(R*C)."""
        if not 0 <= c < self.cols:
            raise IndexError("column index out of range")
        nc = self.cols - 1
        new = _alloc(self.rows * nc)
        for i in range(self.rows):
            for j in range(nc):
                src = j if j < c else j + 1
                new[i * nc + j] = self._data[i * self.cols + src]
        self._data, self.cols = new, nc

    # ---- arithmetic ------------------------------------------------------
    def transpose(self) -> "Matrix":
        t = Matrix(self.cols, self.rows)
        for i in range(self.rows):
            for j in range(self.cols):
                t._data[j * self.rows + i] = self._data[i * self.cols + j]
        return t

    def __add__(self, other: "Matrix") -> "Matrix":
        if self.shape != other.shape:
            raise ValueError("shape mismatch")
        out = Matrix(self.rows, self.cols)
        for i in range(self.rows * self.cols):
            out._data[i] = self._data[i] + other._data[i]
        return out

    def __matmul__(self, other: "Matrix") -> "Matrix":
        """Standard O(R*C*P) product. Loop order i-k-j keeps the inner loop
        walking rows of both operands (row-major friendly)."""
        if self.cols != other.rows:
            raise ValueError("inner dimensions do not match")
        R, C, P = self.rows, self.cols, other.cols
        out = Matrix(R, P)
        for i in range(R):
            for k in range(C):
                a = self._data[i * C + k]
                if a == 0:
                    continue
                for j in range(P):
                    out._data[i * P + j] += a * other._data[k * P + j]
        return out
