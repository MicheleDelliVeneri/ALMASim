"""Native ALMASim MeasurementSet storage."""
from __future__ import annotations

import math
import struct
from pathlib import Path
from typing import Any

import numpy as np

from .ms_model import (
    MeasurementSetModel,
    MeasurementSetTable,
    build_measurement_set_model,
)

_AIPSIO_MAGIC = 0xBEBEBEBE

DATA_TYPE_CODES = {
    "bool": 0,           # TpBool
    "u1": 2,             # TpUChar
    "i2": 3,             # TpShort
    "u2": 4,             # TpUShort
    "i4": 5,             # TpInt
    "u4": 6,             # TpUInt
    "f4": 7,             # TpFloat
    "f8": 8,             # TpDouble
    "c8": 9,             # TpComplex
    "c16": 10,           # TpDComplex
    "str": 11,           # TpString
    "i8": 29,            # TpInt64
}


def export_native_ms(
    *,
    ms_path: str | Path,
    visibility_table: dict[str, Any],
    project_name: str,
    source_name: str,
    telescope_name: str = "ALMA",
) -> str:
    """Write a native `.ms` directory tree from ALMASim's logical MS model."""
    model = build_measurement_set_model(
        visibility_table=visibility_table,
        project_name=project_name,
        source_name=source_name,
        telescope_name=telescope_name,
    )
    return write_native_ms(ms_path=ms_path, model=model)


def write_native_ms(
    *,
    ms_path: str | Path,
    model: MeasurementSetModel,
) -> str:
    """Write a native `.ms` directory tree to disk."""
    ms_path = Path(ms_path).expanduser().resolve()
    if ms_path.exists():
        raise FileExistsError(f"MeasurementSet path already exists: {ms_path}")
    ms_path.parent.mkdir(parents=True, exist_ok=True)

    _write_native_table_root(
        table_root=ms_path,
        table_name="MAIN",
        table=model.main,
        main_ncolumns=len(model.main.columns),
        subtype="ALMASim",
        readme_lines=["Experimental native MeasurementSet layout written by ALMASim."],
        keywords=_normalize_keywords(model.main_keywords),
    )

    for subtable_name, subtable in model.subtables.items():
        subtable_path = ms_path / subtable_name
        _write_native_table_root(
            table_root=subtable_path,
            table_name=subtable_name,
            table=subtable,
            main_ncolumns=len(subtable.columns),
            subtype=subtable_name,
            readme_lines=[f"ALMASim MeasurementSet subtable: {subtable_name}"],
            keywords={},
        )
    return str(ms_path)


def _write_native_table_root(
    *,
    table_root: Path,
    table_name: str,
    table: MeasurementSetTable,
    main_ncolumns: int,
    subtype: str,
    readme_lines: list[str],
    keywords: dict[str, Any],
) -> None:
    table_root.mkdir(parents=True, exist_ok=False)
    (table_root / "table.info").write_text(_build_table_info(subtype, readme_lines))
    (table_root / "table.lock").write_bytes(
        _build_table_lock(
            nrows=table.nrows,
            ncolumns=main_ncolumns,
        )
    )
    (table_root / "table.dat").write_bytes(
        _build_plain_table_dat(
            table_name=table_name,
            table=table,
            keywords=keywords,
        )
    )
    (table_root / "table.f0").write_bytes(_build_standard_stman_file(table))


def _normalize_keywords(keywords: dict[str, Any]) -> dict[str, Any]:
    normalized = {}
    for key, value in keywords.items():
        if isinstance(value, np.generic):
            normalized[key] = value.item()
        else:
            normalized[key] = value
    return normalized


def _build_table_info(subtype: str, readme_lines: list[str]) -> str:
    _ = readme_lines
    lines = [
        "Type = MeasurementSet",
        f"SubType = {subtype}",
    ]
    return "\n".join(lines) + "\n"


def _build_table_lock(*, nrows: int, ncolumns: int) -> bytes:
    aips = _AipsIOWriter()
    sync_payload = (
        aips.uInt(max(0, nrows))
        + aips.uInt(max(0, ncolumns))
        + aips.uInt(2)
        + aips.uInt(1)
        + aips.block_uint([1])
    )
    sync_stream = aips.object("sync", 1, sync_payload, with_magic=True)
    process_list = bytes(260)
    return process_list + aips.uInt(len(sync_stream)) + sync_stream


class _AipsIOWriter:
    def object(
        self,
        object_type: str,
        version: int,
        payload: bytes,
        *,
        with_magic: bool = False,
    ) -> bytes:
        header = self.uInt(0) + self.string(object_type) + self.uInt(version)
        total_len = len(header) + len(payload)
        header = self.uInt(total_len) + self.string(object_type) + self.uInt(version)
        blob = header + payload
        if with_magic:
            return self.uInt(_AIPSIO_MAGIC) + blob
        return blob

    def bool(self, value: bool) -> bytes:
        return b"\x01" if value else b"\x00"

    def int(self, value: int) -> bytes:
        return struct.pack(">i", int(value))

    def uInt(self, value: int) -> bytes:
        return struct.pack(">I", int(value) & 0xFFFFFFFF)

    def int64(self, value: int) -> bytes:
        return struct.pack(">q", int(value))

    def float(self, value: float) -> bytes:
        return struct.pack(">f", float(value))

    def double(self, value: float) -> bytes:
        return struct.pack(">d", float(value))

    def complex64(self, value: complex) -> bytes:
        return self.float(float(np.real(value))) + self.float(float(np.imag(value)))

    def complex128(self, value: complex) -> bytes:
        return self.double(float(np.real(value))) + self.double(float(np.imag(value)))

    def string(self, value: str) -> bytes:
        encoded = value.encode("ascii", errors="replace")
        return self.uInt(len(encoded)) + encoded

    def block_uint(self, values: list[int], *, int64_values: bool = False) -> bytes:
        payload = self.uInt(len(values))
        for value in values:
            payload += self.int64(value) if int64_values else self.uInt(value)
        return self.object("Block", 1, payload)

    def iposition(self, shape: list[int]) -> bytes:
        payload = self.uInt(len(shape))
        for value in shape:
            payload += self.int64(int(value))
        return self.object("IPosition", 2, payload)

    def array(self, value: np.ndarray) -> bytes:
        arr = np.asarray(value)
        payload = self.uInt(arr.ndim)
        for size in arr.shape:
            payload += self.uInt(int(size))
        payload += b"".join(self.int(0) for _ in range(arr.ndim))
        payload += self._array_data(arr)
        return self.object("Array", 3, payload)

    def _array_data(self, arr: np.ndarray) -> bytes:
        if arr.dtype == np.bool_:
            packed = np.packbits(arr.reshape(-1, order="F").astype(np.uint8), bitorder="little")
            return packed.tobytes()
        if arr.dtype.kind in {"S", "U"}:
            payload = b""
            flat = arr.reshape(-1, order="F")
            for item in flat:
                payload += self.string(str(item))
            return payload
        if arr.dtype == np.int16:
            return b"".join(self.int(int(v))[2:] for v in arr.reshape(-1, order="F"))
        if arr.dtype == np.uint16:
            return b"".join(self.uInt(int(v))[2:] for v in arr.reshape(-1, order="F"))
        if arr.dtype == np.int32:
            return b"".join(self.int(int(v)) for v in arr.reshape(-1, order="F"))
        if arr.dtype == np.uint32:
            return b"".join(self.uInt(int(v)) for v in arr.reshape(-1, order="F"))
        if arr.dtype == np.int64:
            return b"".join(self.int64(int(v)) for v in arr.reshape(-1, order="F"))
        if arr.dtype == np.float32:
            return arr.astype(">f4", copy=False).reshape(-1, order="F").tobytes()
        if arr.dtype == np.float64:
            return arr.astype(">f8", copy=False).reshape(-1, order="F").tobytes()
        if arr.dtype == np.complex64:
            return arr.astype(">c8", copy=False).reshape(-1, order="F").tobytes()
        if arr.dtype == np.complex128:
            return arr.astype(">c16", copy=False).reshape(-1, order="F").tobytes()
        if arr.dtype == np.uint8:
            return arr.reshape(-1, order="F").tobytes()
        raise TypeError(f"Unsupported array dtype for AipsIO serialization: {arr.dtype}")


class _LocalStManWriter(_AipsIOWriter):
    def int(self, value: int) -> bytes:
        return struct.pack("<i", int(value))

    def uInt(self, value: int) -> bytes:
        return struct.pack("<I", int(value) & 0xFFFFFFFF)

    def int64(self, value: int) -> bytes:
        return struct.pack("<q", int(value))

    def float(self, value: float) -> bytes:
        return struct.pack("<f", float(value))

    def double(self, value: float) -> bytes:
        return struct.pack("<d", float(value))

    def complex64(self, value: complex) -> bytes:
        return self.float(float(np.real(value))) + self.float(float(np.imag(value)))

    def complex128(self, value: complex) -> bytes:
        return self.double(float(np.real(value))) + self.double(float(np.imag(value)))


def _build_plain_table_dat(
    *,
    table_name: str,
    table: MeasurementSetTable,
    keywords: dict[str, Any],
) -> bytes:
    aips = _AipsIOWriter()
    column_names = list(table.columns.keys())
    column_descs = [
        _serialize_column_desc(name, table.columns[name], aips, table.nrows)
        for name in column_names
    ]
    table_desc_payload = (
        aips.string(table_name)
        + aips.string("ALMASim")
        + aips.string("")
        + _serialize_table_record(keywords, aips)
        + _serialize_table_record({}, aips)
        + aips.uInt(len(column_names))
        + b"".join(column_descs)
    )
    table_desc = aips.object("TableDesc", 2, table_desc_payload)
    column_set = _serialize_column_set(table, column_names, aips)
    payload = (
        aips.uInt(table.nrows)
        + aips.uInt(1)
        + aips.string("PlainTable")
        + table_desc
        + column_set
    )
    return aips.object("Table", 2, payload, with_magic=True)


def _serialize_column_set(
    table: MeasurementSetTable,
    column_names: list[str],
    aips: _AipsIOWriter,
) -> bytes:
    values = [table.columns[name] for name in column_names]
    rows_per_bucket = _standard_stman_rows_per_bucket()
    bucket_offsets = _compute_bucket_offsets(values, rows_per_bucket, table.nrows)
    dm_info = _serialize_standard_stman_dm_info(bucket_offsets)
    payload = (
        aips.int(-2)
        + aips.uInt(table.nrows)
        + aips.uInt(1)
        + aips.uInt(1)
        + aips.string("StandardStMan")
        + aips.uInt(0)
    )
    for name in column_names:
        value = table.columns[name]
        payload += aips.int(2)
        payload += aips.string(name)
        payload += aips.uInt(1)
        payload += aips.uInt(0)
        arr = np.asarray(value)
        if not _is_scalar_value(arr, table.nrows):
            payload += aips.bool(True)
            payload += aips.iposition(list(arr.shape[1:] if table.nrows > 1 else arr.shape))
    payload += aips.uInt(len(dm_info)) + dm_info
    return payload


def _serialize_standard_stman_dm_info(offsets: list[int]) -> bytes:
    aips = _AipsIOWriter()
    payload = (
        aips.string("StandardStMan")
        + aips.block_uint(offsets)
        + aips.block_uint([0] * len(offsets))
    )
    return aips.object("SSM", 2, payload, with_magic=True)


def _serialize_column_desc(
    name: str,
    value: Any,
    aips: _AipsIOWriter,
    nrows: int,
) -> bytes:
    arr = np.asarray(value)
    is_scalar = _is_scalar_value(arr, nrows)
    column_type = _column_type_string(arr, is_scalar=is_scalar)
    data_type = _column_data_type(arr, is_scalar=is_scalar)
    ndim_value = 0 if is_scalar else arr.ndim - (1 if arr.ndim > 1 else 0)
    payload = aips.uInt(1) + aips.string(column_type)
    payload += aips.uInt(1)
    payload += aips.string(name)
    payload += aips.string("")
    payload += aips.string("StandardStMan")
    payload += aips.string("StandardStMan")
    payload += aips.int(data_type)
    payload += aips.int(_column_options(arr, is_scalar))
    payload += aips.uInt(ndim_value)
    if not is_scalar:
        payload += aips.iposition(list(arr.shape[1:] if arr.ndim > 1 else arr.shape))
    payload += aips.uInt(_string_max_length(arr))
    payload += _serialize_table_record({}, aips)
    if is_scalar:
        payload += aips.uInt(1)
        payload += _serialize_scalar_value(_scalar_default(arr), aips)
    else:
        payload += aips.uInt(1)
        payload += aips.bool(False)
    return payload


def _serialize_table_record(record: dict[str, Any], aips: _AipsIOWriter) -> bytes:
    desc_payload = aips.uInt(len(record))
    values_payload = b""
    for key, value in record.items():
        arr = np.asarray(value) if not isinstance(value, dict) else None
        desc_payload += aips.string(str(key))
        desc_payload += aips.int(_record_data_type(value))
        if isinstance(value, dict):
            desc_payload += _serialize_record_desc(value, aips)
        elif _record_is_array(value):
            desc_payload += aips.iposition([-1])
        desc_payload += aips.string("")
        values_payload += _serialize_record_value(value, aips)
    desc = aips.object("RecordDesc", 2, desc_payload)
    payload = desc + aips.int(1) + values_payload
    return aips.object("TableRecord", 1, payload)


def _serialize_record_desc(record: dict[str, Any], aips: _AipsIOWriter) -> bytes:
    payload = aips.uInt(len(record))
    for key, value in record.items():
        payload += aips.string(str(key))
        payload += aips.int(_record_data_type(value))
        if isinstance(value, dict):
            payload += _serialize_record_desc(value, aips)
        elif _record_is_array(value):
            payload += aips.iposition([-1])
        payload += aips.string("")
    return aips.object("RecordDesc", 2, payload)


def _record_is_array(value: Any) -> bool:
    return isinstance(value, np.ndarray) or isinstance(value, (list, tuple))


def _record_data_type(value: Any) -> int:
    if isinstance(value, dict):
        return 25  # TpRecord
    arr = np.asarray(value)
    if arr.ndim > 0:
        return _column_data_type(arr, is_scalar=False, for_record=True)
    return _column_data_type(arr, is_scalar=True)


def _serialize_record_value(value: Any, aips: _AipsIOWriter) -> bytes:
    if isinstance(value, dict):
        return _serialize_table_record(value, aips)
    arr = np.asarray(value)
    if arr.ndim > 0:
        return aips.array(arr)
    return _serialize_scalar_value(arr.item(), aips)


def _serialize_scalar_value(value: Any, aips: _AipsIOWriter) -> bytes:
    if isinstance(value, (bool, np.bool_)):
        return aips.bool(bool(value))
    if isinstance(value, np.uint8):
        return bytes([int(value)])
    if isinstance(value, np.int16):
        return struct.pack(">h", int(value))
    if isinstance(value, np.uint16):
        return struct.pack(">H", int(value))
    if isinstance(value, np.int32):
        return aips.int(int(value))
    if isinstance(value, np.uint32):
        return aips.uInt(int(value))
    if isinstance(value, np.int64):
        return aips.int64(int(value))
    if isinstance(value, int):
        return aips.int(int(value))
    if isinstance(value, np.float32):
        return aips.float(float(value))
    if isinstance(value, float):
        return aips.double(float(value))
    if isinstance(value, np.float64):
        return aips.double(float(value))
    if isinstance(value, np.complex64):
        return aips.complex64(complex(value))
    if isinstance(value, complex):
        return aips.complex128(complex(value))
    if isinstance(value, np.complex128):
        return aips.complex128(complex(value))
    if isinstance(value, (str, np.str_)):
        return aips.string(str(value))
    raise TypeError(f"Unsupported scalar value for AipsIO serialization: {type(value)!r}")


def _build_standard_stman_file(table: MeasurementSetTable, bucket_size: int = 384) -> bytes:
    aips = _LocalStManWriter()
    column_names = list(table.columns.keys())
    values = [table.columns[name] for name in column_names]
    rows_per_bucket = _standard_stman_rows_per_bucket()
    row_stride = _row_stride_bytes(values, table.nrows)
    if row_stride > 0:
        bucket_size = max(bucket_size, row_stride * rows_per_bucket)
    offsets = _compute_bucket_offsets(values, rows_per_bucket, table.nrows)
    total_storage = 0
    if values:
        total_storage = offsets[-1] + _column_storage_size(values[-1], table.nrows) * rows_per_bucket
    n_data_buckets = math.ceil(total_storage / bucket_size) if table.nrows > 0 and total_storage > 0 else 0
    index_object = _build_ssm_index(
        n_buckets=max(1, n_data_buckets - 1) if table.nrows > 0 else 0,
        rows_per_bucket=rows_per_bucket,
        n_columns=len(column_names),
        last_rows=[max(table.nrows - 1, 0)],
        aips=aips,
        map_keys=list(range(1, max(1, n_data_buckets - 1) + 1)) if table.nrows > 0 else [],
        map_values=[1] * max(1, n_data_buckets - 1) if table.nrows > 0 else [],
    )
    bucket_size = max(bucket_size, 8 + len(index_object))
    data_buckets = _build_data_buckets(
        table=table,
        column_names=column_names,
        bucket_size=bucket_size,
        rows_per_bucket=rows_per_bucket,
        offsets=offsets,
    )
    index_object = _build_ssm_index(
        n_buckets=max(1, n_data_buckets - 1) if table.nrows > 0 else 0,
        rows_per_bucket=rows_per_bucket,
        n_columns=len(column_names),
        last_rows=[max(table.nrows - 1, 0)],
        aips=aips,
        map_keys=list(range(1, max(1, n_data_buckets - 1) + 1)) if table.nrows > 0 else [],
        map_values=[1] * max(1, n_data_buckets - 1) if table.nrows > 0 else [],
    )
    trailing_index_object = _build_ssm_index(
        n_buckets=max(1, n_data_buckets - 1) if table.nrows > 0 else 0,
        rows_per_bucket=rows_per_bucket,
        n_columns=len(column_names),
        last_rows=[max(table.nrows - 1, 0)],
        aips=aips,
        map_keys=[0] if table.nrows > 0 else [],
        map_values=[max(1, n_data_buckets - 1)] if table.nrows > 0 else [],
    )
    index_bucket = _build_index_bucket(
        index_object=index_object,
        bucket_size=bucket_size,
        index_offset=0,
        trailing_index_object=trailing_index_object,
    )
    total_buckets = 1 + len(data_buckets)
    header_payload = (
        aips.bool(False)
        + aips.uInt(bucket_size)
        + aips.uInt(total_buckets)
        + aips.uInt(total_buckets)
        + aips.uInt(0)
        + aips.int(-1)
        + aips.uInt(1)
        + aips.int(0)
        + aips.uInt(8)
        + aips.int(-1)
        + aips.uInt(len(index_object))
        + aips.uInt(1)
    )
    header = aips.object("StandardStMan", 3, header_payload, with_magic=True)
    if len(header) > 512:
        raise ValueError("StandardStMan header exceeds 512 bytes")
    header_block = header + bytes(512 - len(header))
    payload = header_block + index_bucket + b"".join(data_buckets)
    return payload


def _build_index_bucket(
    *,
    index_object: bytes,
    bucket_size: int,
    index_offset: int,
    trailing_index_object: bytes | None = None,
) -> bytes:
    bucket = bytearray(bucket_size)
    bucket[0:4] = struct.pack(">i", -1)
    bucket[4:8] = struct.pack(">i", -1)
    object_offset = 8 + index_offset
    bucket[object_offset : object_offset + len(index_object)] = index_object
    if trailing_index_object:
        trailer_offset = bucket_size // 2 + 4
        if (
            trailer_offset >= object_offset + len(index_object)
            and trailer_offset + len(trailing_index_object) <= bucket_size
        ):
            bucket[trailer_offset : trailer_offset + len(trailing_index_object)] = trailing_index_object
    return bytes(bucket)


def _build_ssm_index(
    *,
    n_buckets: int,
    rows_per_bucket: int,
    n_columns: int,
    last_rows: list[int],
    aips: _AipsIOWriter,
    map_keys: list[int] | None = None,
    map_values: list[int] | None = None,
) -> bytes:
    if map_keys is None or map_values is None:
        map_keys = []
        map_values = []
        if n_buckets > 0:
            map_keys = list(range(1, n_buckets + 1))
            map_values = [1] * n_buckets
    simple_ordered_map = aips.object(
        "SimpleOrderedMap",
        1,
        aips.int(0) + aips.uInt(0) + aips.uInt(1),
    )
    payload = (
        aips.uInt(n_buckets)
        + aips.uInt(rows_per_bucket)
        + aips.int(n_columns)
        + simple_ordered_map
        + aips.block_uint(map_keys)
        + aips.block_uint(map_values)
    )
    return aips.object("SSMIndex", 1, payload, with_magic=True)


def _build_data_buckets(
    table: MeasurementSetTable,
    column_names: list[str],
    bucket_size: int,
    rows_per_bucket: int,
    offsets: list[int],
) -> list[bytes]:
    if table.nrows <= 0 or not column_names:
        return []
    total_storage = offsets[-1] + _column_storage_size(
        table.columns[column_names[-1]],
        table.nrows,
    ) * rows_per_bucket
    storage = bytearray(total_storage)
    for name, offset in zip(column_names, offsets, strict=True):
        value = table.columns[name]
        encoded = _encode_column_data(
            value,
            table.nrows,
            rows_per_bucket,
        )
        storage[offset : offset + len(encoded)] = encoded
    buckets: list[bytes] = []
    for start in range(0, len(storage), bucket_size):
        chunk = bytes(storage[start : start + bucket_size])
        if len(chunk) < bucket_size:
            chunk = chunk + bytes(bucket_size - len(chunk))
        buckets.append(chunk)
    return buckets


def _encode_column_data(value: Any, nrows: int, rows_per_bucket: int) -> bytes:
    arr = np.asarray(value)
    row_count = min(nrows, rows_per_bucket)
    if _is_scalar_value(arr, nrows):
        rows = np.repeat(arr.reshape(1), row_count, axis=0) if arr.ndim == 0 else arr[:row_count]
        payload = b"".join(_encode_scalar_storage_for_stman(v, arr) for v in rows.reshape(-1))
        pad_size = _column_storage_size(arr, nrows) * rows_per_bucket - len(payload)
        if pad_size > 0:
            payload += bytes(pad_size)
        return payload
    rows = arr[:row_count]
    payload = b""
    for row in rows:
        payload += _encode_fixed_array_storage(np.asarray(row))
    pad_size = _column_storage_size(arr, nrows) * rows_per_bucket - len(payload)
    if pad_size > 0:
        payload += bytes(pad_size)
    return payload


def _encode_scalar_storage_for_stman(value: Any, arr: np.ndarray) -> bytes:
    if arr.dtype.kind in {"U", "S"}:
        slot_size = _string_storage_slot_size(arr)
        data_size = slot_size - 4
        raw = str(value).encode("ascii", errors="replace")[:data_size]
        return raw.ljust(data_size, b"\x00") + struct.pack("<I", len(raw))
    return _encode_scalar_storage(value, arr)


def _encode_scalar_storage(value: Any, arr: np.ndarray) -> bytes:
    dtype = arr.dtype
    if dtype == np.bool_:
        return b"\x01" if bool(value) else b"\x00"
    if dtype == np.uint8:
        return struct.pack("<B", int(value))
    if dtype == np.int16:
        return struct.pack("<h", int(value))
    if dtype == np.uint16:
        return struct.pack("<H", int(value))
    if dtype == np.int32:
        return struct.pack("<i", int(value))
    if dtype == np.uint32:
        return struct.pack("<I", int(value))
    if dtype == np.int64:
        return struct.pack("<q", int(value))
    if dtype == np.float32:
        return struct.pack("<f", float(value))
    if dtype == np.float64:
        return struct.pack("<d", float(value))
    if dtype == np.complex64:
        return struct.pack("<ff", float(np.real(value)), float(np.imag(value)))
    if dtype == np.complex128:
        return struct.pack("<dd", float(np.real(value)), float(np.imag(value)))
    if dtype.kind in {"U", "S"}:
        maxlen = _string_max_length(arr)
        encoded = str(value).encode("ascii", errors="replace")[:maxlen]
        if len(encoded) < maxlen:
            encoded = encoded + b"\x00" + bytes(max(0, maxlen - len(encoded) - 1))
        return encoded.ljust(maxlen, b"\x00")
    raise TypeError(f"Unsupported scalar storage dtype: {dtype}")


def _encode_fixed_array_storage(arr: np.ndarray) -> bytes:
    flat = np.asarray(arr).reshape(-1, order="F")
    if flat.dtype == np.bool_:
        return np.packbits(flat.astype(np.uint8), bitorder="little").tobytes()
    if flat.dtype.kind in {"U", "S"}:
        maxlen = _string_max_length(flat)
        return b"".join(_encode_scalar_storage(item, np.asarray(flat)) for item in flat)
    if flat.dtype == np.uint8:
        return flat.astype("<u1", copy=False).tobytes()
    if flat.dtype == np.int16:
        return flat.astype("<i2", copy=False).tobytes()
    if flat.dtype == np.uint16:
        return flat.astype("<u2", copy=False).tobytes()
    if flat.dtype == np.int32:
        return flat.astype("<i4", copy=False).tobytes()
    if flat.dtype == np.uint32:
        return flat.astype("<u4", copy=False).tobytes()
    if flat.dtype == np.int64:
        return flat.astype("<i8", copy=False).tobytes()
    if flat.dtype == np.float32:
        return flat.astype("<f4", copy=False).tobytes()
    if flat.dtype == np.float64:
        return flat.astype("<f8", copy=False).tobytes()
    if flat.dtype == np.complex64:
        return flat.astype("<c8", copy=False).tobytes()
    if flat.dtype == np.complex128:
        return flat.astype("<c16", copy=False).tobytes()
    raise TypeError(f"Unsupported fixed-array storage dtype: {flat.dtype}")


def _compute_bucket_offsets(values: list[Any], rows_per_bucket: int, nrows: int) -> list[int]:
    offsets: list[int] = []
    current = 0
    for value in values:
        offsets.append(current)
        current += _column_storage_size(value, nrows) * rows_per_bucket
    return offsets


def _column_storage_size(value: Any, nrows: int | None) -> int:
    arr = np.asarray(value)
    if _is_scalar_value(arr, nrows if nrows is not None else (arr.shape[0] if arr.ndim > 0 else 1)):
        if arr.dtype.kind in {"U", "S"}:
            return _string_storage_slot_size(arr)
        return _dtype_itemsize(arr.dtype)
    shape = arr.shape[1:] if nrows is not None and arr.ndim > 1 else arr.shape
    nelem = int(np.prod(shape)) if shape else 1
    if arr.dtype == np.bool_:
        return max(1, math.ceil(nelem / 8))
    if arr.dtype.kind in {"U", "S"}:
        return nelem * _string_storage_slot_size(arr)
    return nelem * _dtype_itemsize(arr.dtype)


def _row_stride_bytes(values: list[Any], nrows: int) -> int:
    return sum(_column_storage_size(value, nrows) for value in values)


def _standard_stman_rows_per_bucket() -> int:
    return 32


def _column_type_string(arr: np.ndarray, *, is_scalar: bool) -> str:
    suffix = _column_type_suffix(arr)
    type_name = f"{suffix:<8}"
    if is_scalar:
        return f"ScalarColumnDesc<{type_name}"
    return f"ArrayColumnDesc<{type_name}"


def _column_type_suffix(arr: np.ndarray) -> str:
    if arr.dtype == np.bool_:
        return "Bool"
    if arr.dtype == np.uint8:
        return "uChar"
    if arr.dtype == np.int16:
        return "Short"
    if arr.dtype == np.uint16:
        return "uShort"
    if arr.dtype == np.int32:
        return "Int"
    if arr.dtype == np.uint32:
        return "uInt"
    if arr.dtype == np.float32:
        return "float"
    if arr.dtype == np.float64:
        return "double"
    if arr.dtype == np.complex64:
        return "Complex"
    if arr.dtype == np.complex128:
        return "DComplex"
    if arr.dtype.kind in {"U", "S"}:
        return "String"
    if arr.dtype == np.int64:
        return "Int64"
    raise TypeError(f"Unsupported column dtype: {arr.dtype}")


def _column_data_type(arr: np.ndarray, *, is_scalar: bool, for_record: bool = False) -> int:
    if arr.dtype == np.bool_:
        kind = "bool"
    else:
        kind = "str" if arr.dtype.kind in {"U", "S"} else arr.dtype.str[1:]
    base = DATA_TYPE_CODES[kind]
    if for_record and not is_scalar:
        return base + 13 if kind not in {"str", "i8"} else (24 if kind == "str" else 30)
    return base


def _column_options(arr: np.ndarray, is_scalar: bool) -> int:
    if is_scalar:
        return 0
    return 1 | 4


def _string_max_length(arr: np.ndarray) -> int:
    if arr.dtype.kind not in {"U", "S"}:
        return 0
    flat = arr.reshape(-1).tolist()
    return max(1, max(len(str(item).encode("ascii", errors="replace")) for item in flat))


def _string_storage_slot_size(arr: np.ndarray) -> int:
    maxlen = _string_max_length(arr)
    data_size = max(8, ((maxlen + 7) // 8) * 8)
    return data_size + 4


def _dtype_itemsize(dtype: np.dtype) -> int:
    if dtype == np.bool_:
        return 1
    return int(np.dtype(dtype).itemsize)


def _scalar_default(arr: np.ndarray) -> Any:
    dtype = arr.dtype
    if dtype == np.bool_:
        return False
    if dtype == np.uint8:
        return np.uint8(0)
    if dtype == np.int16:
        return np.int16(0)
    if dtype == np.uint16:
        return np.uint16(0)
    if dtype == np.int32:
        return np.int32(0)
    if dtype == np.uint32:
        return np.uint32(0)
    if dtype == np.int64:
        return np.int64(0)
    if dtype == np.float32:
        return np.float32(0.0)
    if dtype == np.float64:
        return np.float64(0.0)
    if dtype == np.complex64:
        return np.complex64(0.0 + 0.0j)
    if dtype == np.complex128:
        return np.complex128(0.0 + 0.0j)
    if dtype.kind in {"U", "S"}:
        return ""
    raise TypeError(f"Unsupported scalar default dtype: {dtype}")


def _is_scalar_value(arr: np.ndarray, nrows: int) -> bool:
    if arr.ndim == 0:
        return True
    if arr.ndim == 1 and arr.dtype.kind in {"U", "S"} and nrows > 1:
        return True
    return arr.ndim == 1 and nrows > 1 and arr.dtype.kind not in {"U", "S"}
