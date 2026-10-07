"""Recover seekable ZIP local records when a timed-out writer lost its directory.

Copy complete local records verbatim to a NEW archive and regenerate only its
central directory. The source is never changed. Incomplete final files are
omitted. ZIPs with streaming data descriptors need a different recovery method.
"""
import argparse
import json
import shutil
import struct
import zipfile
from pathlib import Path


def scan_local_records(source):
    source = Path(source)
    size = source.stat().st_size
    records, offset, stop = [], 0, 'end_of_file'
    with source.open('rb') as handle:
        while offset < size:
            handle.seek(offset)
            header = handle.read(30)
            if len(header) < 30:
                stop = 'truncated_header'
                break
            signature, version, flags, method, dos_time, dos_date, crc, packed, unpacked, name_len, extra_len = struct.unpack('<4s5H3I2H', header)
            if signature != b'PK\x03\x04':
                stop = 'central_directory_or_nonlocal_record'
                break
            name_raw, extra = handle.read(name_len), handle.read(extra_len)
            if len(name_raw) != name_len or len(extra) != extra_len:
                stop = 'truncated_name_or_extra'
                break
            name = name_raw.decode('utf-8' if flags & 0x800 else 'cp437')
            if flags & (1 | 8):
                stop = 'encrypted_or_streaming_descriptor_file: ' + name
                break
            if packed == 0xffffffff or unpacked == 0xffffffff:
                cursor, zip64 = 0, None
                while cursor + 4 <= len(extra):
                    tag, length = struct.unpack_from('<HH', extra, cursor)
                    payload = extra[cursor + 4:cursor + 4 + length]
                    if tag == 1:
                        zip64 = payload
                    cursor += 4 + length
                if zip64 is None:
                    raise ValueError('ZIP64 sizes missing: ' + name)
                cursor = 0
                if unpacked == 0xffffffff:
                    unpacked = struct.unpack_from('<Q', zip64, cursor)[0]
                    cursor += 8
                if packed == 0xffffffff:
                    packed = struct.unpack_from('<Q', zip64, cursor)[0]
            if (method == 8 and packed == 0) or offset + 30 + name_len + extra_len + packed > size:
                stop = 'incomplete_file: ' + name
                break
            info = zipfile.ZipInfo(name, (1980 + (dos_date >> 9), (dos_date >> 5) & 15,
                                         dos_date & 31, dos_time >> 11,
                                         (dos_time >> 5) & 63, (dos_time & 31) * 2))
            info.extract_version = version
            info.flag_bits = flags
            info.compress_type = method
            info.CRC, info.compress_size, info.file_size = crc, packed, unpacked
            info.header_offset = offset
            info.external_attr = (0o100644 << 16) if not name.endswith('/') else (0o40755 << 16) | 16
            # File payload and local extra remain byte-identical. Central extras
            # are generated anew by zipfile, including ZIP64 offsets if needed.
            info.extra = b''
            records.append(info)
            offset += 30 + name_len + extra_len + packed
    if not records:
        raise ValueError('No complete ZIP local records were recoverable.')
    if len({i.filename for i in records}) != len(records):
        raise ValueError('Duplicate local record paths require manual inspection.')
    return records, offset, stop


def recover_zip(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination or destination.exists():
        raise ValueError('Use a new destination; the original archive must remain unchanged.')
    records, boundary, stop = scan_local_records(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if boundary > shutil.disk_usage(destination.parent).free:
        raise RuntimeError('Not enough disk space to copy the recoverable archive.')
    with source.open('rb') as handle, destination.open('xb') as output:
        remaining = boundary
        while remaining:
            block = handle.read(min(8 * 1024 * 1024, remaining))
            if not block:
                raise EOFError('Source changed while recovery was running.')
            output.write(block)
            remaining -= len(block)
    # Append a directory for the existing raw local records; no recompression.
    with zipfile.ZipFile(destination, 'a', allowZip64=True) as archive:
        archive.filelist = records
        archive.NameToInfo = {info.filename: info for info in records}
        archive.start_dir = boundary
        archive._didModify = True
    with zipfile.ZipFile(destination) as archive:
        if len(archive.infolist()) != len(records):
            raise RuntimeError('Recovered central directory has an unexpected record count.')
    return dict(source=str(source), destination=str(destination), complete_records=len(records),
                copied_local_record_bytes=boundary, omitted_tail_bytes=source.stat().st_size - boundary,
                scan_stop_reason=stop, source_modified=False, crc_validation='not_yet_run')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source')
    parser.add_argument('destination')
    args = parser.parse_args()
    print(json.dumps(recover_zip(args.source, args.destination), ensure_ascii=False, indent=2))
