# xarray-npz

An [xarray](https://xarray.dev) backend engine for reading and writing NumPy `.npz` files.

## Installation

**From source (current):**

```bash
git clone https://github.com/jychoi-hpc/xarray-npz.git
cd xarray-npz
pip install .
```

For development (editable install):

```bash
pip install -e .
```

Or install directly from the repository without cloning:

```bash
pip install git+ssh://git@github.com/jychoi-hpc/xarray-npz.git
```

**From PyPI (once published):**

```bash
pip install xarray-npz
```

Requires Python 3.10+, NumPy 1.24+, and xarray 2023.1+.

## Usage

### Reading

Once installed, xarray picks up the backend automatically via its entry-point system. Pass `engine="npz"` to `xr.open_dataset`:

```python
import xarray as xr

ds = xr.open_dataset("data.npz", engine="npz")
```

Arrays are loaded **lazily** — data is read from disk only when accessed.

Drop specific variables at open time:

```python
ds = xr.open_dataset("data.npz", engine="npz", drop_variables=["temp", "pressure"])
```

### Writing

```python
from xarray_npz import to_npz

to_npz(ds, "data.npz")
```

`to_npz` preserves dimension names, coordinate variables, and per-variable and dataset-level attributes, enabling full round-trip fidelity.

### Round-trip example

```python
import numpy as np
import xarray as xr
from xarray_npz import to_npz

ds = xr.Dataset(
    {
        "temperature": (["time", "x"], np.random.rand(10, 5), {"units": "K"}),
        "pressure":    (["time", "x"], np.random.rand(10, 5), {"units": "Pa"}),
    },
    coords={
        "time": ("time", np.arange(10), {"axis": "T"}),
        "x":    ("x",    np.linspace(0, 1, 5)),
    },
    attrs={"source": "simulation"},
)

to_npz(ds, "output.npz")

ds2 = xr.open_dataset("output.npz", engine="npz")
# ds2 is identical to ds: same dims, coords, and attrs
```

## Reading plain `.npz` files

Files created with `numpy.savez` (without xarray metadata) are supported. Each array is loaded as a data variable with auto-generated dimension names (`{varname}_dim_0`, `{varname}_dim_1`, …). Scalar arrays are treated as 0-dimensional variables.

```python
import numpy as np

np.savez("plain.npz", a=np.ones((3, 4)), b=np.zeros(3))
ds = xr.open_dataset("plain.npz", engine="npz")
# ds["a"] has dims ("a_dim_0", "a_dim_1")
# ds["b"] has dims ("b_dim_0",)
```

### Inferring dimension names with `hint`

For plain files you know the structure of, pass `hint` to assign meaningful dimension names without writing a full `metadata` dict. Variables whose axes match are automatically aligned.

**Size-based hint** — maps axis length to a dimension name:

```python
# temperature shape (10, 5), pressure shape (10, 5), time shape (10,)
ds = xr.open_dataset("data.npz", engine="npz", hint={10: "time", 5: "x"})
# ds["temperature"].dims == ("time", "x")
# ds["pressure"].dims    == ("time", "x")  ← shared, so xarray aligns them
# ds["time"].dims        == ("time",)
```

Any axis whose length appears in the dict gets the corresponding name. Axes with unrecognised lengths fall back to `{varname}_dim_{i}`.

**Positional hint** — assigns names by axis position across all arrays:

```python
ds = xr.open_dataset("data.npz", engine="npz", hint=["time", "x", "z"])
# 1-D arrays  → ("time",)
# 2-D arrays  → ("time", "x")
# 3-D arrays  → ("time", "x", "z")
```

**Combining `hint` with `metadata`** — `metadata["dims"]` takes precedence over `hint`, so you can use `hint` as a default and override specific variables:

```python
ds = xr.open_dataset(
    "data.npz",
    engine="npz",
    hint={10: "time", 5: "x"},
    metadata={"dims": {"mask": ["x"]}},  # override just "mask"
)
```

## File format

`to_npz` produces a standard `.npz` archive (a ZIP file of `.npy` entries) with two conventions:

| Key pattern | Content |
|---|---|
| `<varname>` | Data variable array |
| `__coord__<name>` | Coordinate array |
| `__xarray_metadata__` | UTF-8 JSON bytes encoding dims, attrs, and coord names |

Because the metadata is stored as a plain `uint8` byte array, `allow_pickle=False` is used throughout and no pickle data is ever written or read.

## Security

`numpy.load` is called with `allow_pickle=False` for all reads, preventing execution of arbitrary pickled Python objects embedded in untrusted files.

## License

MIT
