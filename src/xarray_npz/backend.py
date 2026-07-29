from __future__ import annotations

import json
import zipfile
from collections.abc import Iterable
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr
from xarray.backends import BackendEntrypoint
from xarray.backends.common import BackendArray
from xarray.core.indexing import (
    IndexingSupport,
    LazilyIndexedArray,
    explicit_indexing_adapter,
)

_METADATA_KEY = "__xarray_metadata__"
_COORD_PREFIX = "__coord__"


class _NpzArrayWrapper(BackendArray):
    def __init__(
        self,
        path: Path,
        name: str,
        dtype: np.dtype,
        shape: tuple[int, ...],
    ) -> None:
        self.path = path
        self.name = name
        self.dtype = dtype
        self.shape = shape

    def __getitem__(self, key: Any) -> np.ndarray:
        return explicit_indexing_adapter(
            key, self.shape, IndexingSupport.BASIC, self._getitem
        )

    def _getitem(self, key: Any) -> np.ndarray:
        with np.load(self.path, allow_pickle=False) as archive:
            return archive[self.name][key]

# Approach from https://stackoverflow.com/a/35990775 — read only .npy headers
# from the ZIP index without decompressing array data.
def _npz_headers(
    path: Path,
) -> Iterable[tuple[str, tuple[int, ...], np.dtype]]:
    """Yield (name, shape, dtype) for each .npy entry without loading array data."""
    with zipfile.ZipFile(path) as zf:
        for zip_name in zf.namelist():
            if not zip_name.endswith(".npy"):
                continue
            with zf.open(zip_name) as f:
                version = np.lib.format.read_magic(f)
                try:
                    # numpy 2.0+ public API
                    if version == (1, 0):
                        shape, _, dtype = np.lib.format.read_array_header_1_0(f)
                    else:
                        shape, _, dtype = np.lib.format.read_array_header_2_0(f)
                except AttributeError:
                    # numpy < 2.0 private API
                    shape, _, dtype = np.lib.format._read_array_header(f, version)
            yield zip_name[:-4], shape, dtype


def _fallback_dims(name: str, ndim: int) -> tuple[str, ...]:
    if ndim == 0:
        return ()
    return tuple(f"{name}_dim_{i}" for i in range(ndim))


def _infer_dims_from_hint(
    name: str,
    shape: tuple[int, ...],
    hint: dict[int, str] | list[str],
) -> tuple[str, ...]:
    dims = []
    for i, size in enumerate(shape):
        if isinstance(hint, dict):
            dims.append(hint.get(size, f"{name}_dim_{i}"))
        else:
            dims.append(hint[i] if i < len(hint) else f"{name}_dim_{i}")
    return tuple(dims)


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)


class NPZBackendEntrypoint(BackendEntrypoint):
    """xarray backend for NumPy .npz files."""

    description = "Read NumPy NPZ archives as xarray datasets"
    open_dataset_parameters = ["filename_or_obj", "drop_variables", "metadata", "hint"]
    url = "https://numpy.org/doc/stable/reference/generated/numpy.load.html"

    def open_dataset(
        self,
        filename_or_obj: str | PathLike[str],
        *,
        drop_variables: str | Iterable[str] | None = None,
        metadata: dict[str, Any] | None = None,
        hint: dict[int, str] | list[str] | None = None,
    ) -> xr.Dataset:
        path = Path(filename_or_obj)

        if drop_variables is None:
            dropped: set[str] = set()
        elif isinstance(drop_variables, str):
            dropped = {drop_variables}
        else:
            dropped = set(drop_variables)

        # Read metadata content only (small uint8 array, negligible cost).
        with np.load(path, allow_pickle=False) as archive:
            meta: dict[str, Any] = {}
            if _METADATA_KEY in archive.files:
                meta = json.loads(archive[_METADATA_KEY].tobytes().decode())
        if metadata is not None:
            meta = metadata

        data_vars: dict[str, xr.Variable] = {}
        coords: dict[str, xr.Variable] = {}

        # Read only .npy headers — no array data is decompressed.
        for key, shape, dtype in _npz_headers(path):
            if key == _METADATA_KEY:
                continue

            is_coord = key.startswith(_COORD_PREFIX)
            logical_name = key[len(_COORD_PREFIX):] if is_coord else key

            if logical_name in dropped:
                continue

            if key in meta.get("dims", {}):
                dims = tuple(meta["dims"][key])
            elif hint is not None and len(shape) > 0:
                dims = _infer_dims_from_hint(logical_name, shape, hint)
            else:
                dims = _fallback_dims(logical_name, len(shape))

            attrs: dict[str, Any] = (
                meta.get("coord_attrs", {}).get(logical_name, {})
                if is_coord
                else meta.get("var_attrs", {}).get(logical_name, {})
            )
            wrapper = _NpzArrayWrapper(path, key, dtype, shape)
            var = xr.Variable(
                dims,
                LazilyIndexedArray(wrapper),
                attrs=attrs,
                encoding={"source": str(path), "dtype": dtype},
            )
            if is_coord:
                coords[logical_name] = var
            else:
                data_vars[logical_name] = var

        return xr.Dataset(
            data_vars=data_vars,
            coords=coords,
            attrs=meta.get("attrs", {}),
        )

    def guess_can_open(self, filename_or_obj: object) -> bool:
        if isinstance(filename_or_obj, (str, PathLike)):
            return str(filename_or_obj).lower().endswith(".npz")
        if hasattr(filename_or_obj, "read"):
            try:
                magic = filename_or_obj.read(4)
                if hasattr(filename_or_obj, "seek"):
                    filename_or_obj.seek(0)
                return magic == b"PK\x03\x04"
            except Exception:
                return False
        return False


def to_npz(ds: xr.Dataset, path: str | PathLike[str]) -> None:
    """Save an xarray Dataset to a .npz file, preserving dims, coords, and attrs."""
    arrays: dict[str, np.ndarray] = {}
    meta: dict[str, Any] = {
        "dims": {},
        "attrs": dict(ds.attrs),
        "var_attrs": {},
        "coord_attrs": {},
        "coords": [],
    }

    for name, var in ds.data_vars.items():
        arrays[name] = var.to_numpy()
        meta["dims"][name] = list(var.dims)
        meta["var_attrs"][name] = dict(var.attrs)

    for name, coord in ds.coords.items():
        key = f"{_COORD_PREFIX}{name}"
        arrays[key] = coord.to_numpy()
        meta["dims"][key] = list(coord.dims)
        meta["coord_attrs"][name] = dict(coord.attrs)
        meta["coords"].append(name)

    meta_bytes = json.dumps(meta, cls=_NumpyEncoder).encode()
    arrays[_METADATA_KEY] = np.frombuffer(meta_bytes, dtype=np.uint8).copy()
    np.savez(path, **arrays)
