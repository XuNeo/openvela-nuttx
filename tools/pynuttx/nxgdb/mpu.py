############################################################################
# tools/pynuttx/nxgdb/mpu.py
#
# SPDX-License-Identifier: Apache-2.0
#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.  The
# ASF licenses this file to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance with the
# License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the
# License for the specific language governing permissions and limitations
# under the License.
#
############################################################################

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List, Optional, Tuple

import gdb

from . import autocompeletion, utils

MPU_TYPE = 0xE000ED90
MPU_CTRL = 0xE000ED94
MPU_RNR = 0xE000ED98
MPU_RBAR = 0xE000ED9C
MPU_RASR = 0xE000EDA0
MPU_RLAR = 0xE000EDA0
MPU_RBAR_A1 = 0xE000EDA4
MPU_RLAR_A1 = 0xE000EDA8
MPU_RASR_A1 = 0xE000EDA8
MPU_RBAR_A2 = 0xE000EDAC
MPU_RLAR_A2 = 0xE000EDB0
MPU_RASR_A2 = 0xE000EDB0
MPU_RBAR_A3 = 0xE000EDB4
MPU_RLAR_A3 = 0xE000EDB8
MPU_RASR_A3 = 0xE000EDB8
MPU_MAIR0 = 0xE000EDC0
MPU_MAIR1 = 0xE000EDC4

ADDRESS_MAX = 0xFFFFFFFF


@dataclass(frozen=True)
class Permissions:
    priv_read: bool
    priv_write: bool
    user_read: bool
    user_write: bool

    def allows(self, access: str, mode: str, xn: bool) -> bool:
        if access == "exec":
            return not xn
        if mode == "user":
            return self.user_read if access == "read" else self.user_write
        return self.priv_read if access == "read" else self.priv_write

    def priv_text(self, xn: bool) -> str:
        return _perm_text(self.priv_read, self.priv_write, xn)

    def user_text(self, xn: bool) -> str:
        return _perm_text(self.user_read, self.user_write, xn)


@dataclass(frozen=True)
class MPURegion:
    number: int
    enabled: bool
    start: int
    end: int
    size: int
    permissions: Permissions
    xn: bool
    attr: str
    raw_base: int
    raw_limit: int

    def contains(self, start: int, end: int) -> bool:
        return self.enabled and self.start <= start and end <= self.end


class UnsupportedArchitectureError(gdb.GdbError):
    pass


class MPUArchitecture:
    name = ""
    label = ""
    target_variants: Tuple[str, ...] = ()
    region_regs: Tuple[Tuple[int, ...], ...] = ()
    ap_map = {}

    def matches_target(self) -> bool:
        return any(utils.is_target_arch(variant) for variant in self.target_variants)

    def read_extra_state(self):
        return {}

    def read_regions(self, rnr: int, dregion: int, **extra_state) -> Tuple[List[MPURegion], str]:
        return self._scan_regions(
            rnr,
            dregion,
            lambda number, *values: self.decode_region(number, *values, **extra_state),
        )

    def permissions(self, bits: int) -> Permissions:
        return Permissions(*self.ap_map.get(bits, (False, False, False, False)))

    def decode_region(self, number: int, *values, **extra_state) -> MPURegion:
        raise NotImplementedError

    def _read_alias_regions(self, rnr: int, dregion: int, decode_region) -> List[MPURegion]:
        regions: List[MPURegion] = []
        for offset, reg_addrs in enumerate(self.region_regs):
            number = _region_number(rnr, offset, dregion)
            values = [utils.read_uint(addr) for addr in reg_addrs]
            regions.append(decode_region(number, *values))
        return regions

    def _scan_regions(self, rnr: int, dregion: int, decode_region) -> Tuple[List[MPURegion], str]:
        if dregion == 0:
            return [], "alias"

        regions: List[MPURegion] = []
        wrote_rnr = False
        try:
            for base in range(0, dregion, 4):
                utils.write_uint(MPU_RNR, base)
                wrote_rnr = True
                for offset, reg_addrs in enumerate(self.region_regs):
                    number = base + offset
                    if number >= dregion:
                        break
                    values = [utils.read_uint(addr) for addr in reg_addrs]
                    regions.append(decode_region(number, *values))
        except (gdb.GdbError, gdb.error):
            if wrote_rnr:
                try:
                    utils.write_uint(MPU_RNR, rnr)
                except (gdb.GdbError, gdb.error):
                    pass
            return self._read_alias_regions(rnr, dregion, decode_region), "alias"

        if wrote_rnr:
            try:
                utils.write_uint(MPU_RNR, rnr)
            except (gdb.GdbError, gdb.error):
                pass

        return regions, "full"


class ARMv8MArchitecture(MPUArchitecture):
    name = "armv8-m"
    label = "ARMv8-M"
    target_variants = ("armv8-m", "armv8.1-m")
    region_regs = (
        (MPU_RBAR, MPU_RLAR),
        (MPU_RBAR_A1, MPU_RLAR_A1),
        (MPU_RBAR_A2, MPU_RLAR_A2),
        (MPU_RBAR_A3, MPU_RLAR_A3),
    )
    ap_map = {
        0b00: (True, True, False, False),
        0b01: (True, True, True, True),
        0b10: (True, False, False, False),
        0b11: (True, False, True, False),
    }
    attr_names = {
        0x00: "Device-nGnRnE",
        0x04: "Device-nGnRE",
        0x44: "Normal NC",
        0x77: "Normal WB",
        0xFF: "Normal WB",
    }

    def read_extra_state(self):
        return {
            "mair0": utils.read_uint(MPU_MAIR0),
            "mair1": utils.read_uint(MPU_MAIR1),
        }

    def decode_region(
        self, number: int, rbar: int, rlar: int, mair0: int, mair1: int
    ) -> MPURegion:
        enabled = bool(rlar & 0x1)
        start = rbar & 0xFFFFFFE0
        end = (rlar & 0xFFFFFFE0) | 0x1F
        size = end - start + 1 if end >= start else 0
        attr_index = (rlar >> 1) & 0xF
        attr = self._mair_attr(mair0, mair1, attr_index)
        return MPURegion(
            number=number,
            enabled=enabled and size > 0,
            start=start,
            end=end,
            size=size,
            permissions=self.permissions((rbar >> 1) & 0x3),
            xn=bool(rbar & 0x1),
            attr=self._attr_text(attr),
            raw_base=rbar,
            raw_limit=rlar,
        )

    def _mair_attr(self, mair0: int, mair1: int, index: int) -> int:
        if index < 4:
            return (mair0 >> (index * 8)) & 0xFF
        return (mair1 >> ((index - 4) * 8)) & 0xFF

    def _attr_text(self, mair_byte: int) -> str:
        return self.attr_names.get(mair_byte, f"mair=0x{mair_byte:02x}")


class ARMv7MArchitecture(MPUArchitecture):
    name = "armv7-m"
    label = "ARMv7-M"
    target_variants = ("armv7-m", "armv7e-m")
    region_regs = (
        (MPU_RBAR, MPU_RASR),
        (MPU_RBAR_A1, MPU_RASR_A1),
        (MPU_RBAR_A2, MPU_RASR_A2),
        (MPU_RBAR_A3, MPU_RASR_A3),
    )
    ap_map = {
        0b000: (False, False, False, False),
        0b001: (True, True, False, False),
        0b010: (True, True, True, False),
        0b011: (True, True, True, True),
        0b101: (True, False, False, False),
        0b110: (True, False, True, False),
    }

    def decode_region(self, number: int, rbar: int, rasr: int, **_extra_state) -> MPURegion:
        size_field = (rasr >> 1) & 0x1F
        size = 1 << (size_field + 1) if size_field >= 4 else 0
        start = (rbar & 0xFFFFFFE0) & ~(size - 1) if size else (rbar & 0xFFFFFFE0)
        end = start + size - 1 if size else start
        tex = (rasr >> 19) & 0x7
        shareable = (rasr >> 18) & 0x1
        cacheable = (rasr >> 17) & 0x1
        bufferable = (rasr >> 16) & 0x1
        return MPURegion(
            number=number,
            enabled=bool(rasr & 0x1) and size > 0,
            start=start,
            end=end,
            size=size,
            permissions=self.permissions((rasr >> 24) & 0x7),
            xn=bool((rasr >> 28) & 0x1),
            attr=self._attr_text(tex, shareable, cacheable, bufferable),
            raw_base=rbar,
            raw_limit=rasr,
        )

    def _attr_text(self, tex: int, shareable: int, cacheable: int, bufferable: int) -> str:
        if tex == 0 and cacheable == 0 and bufferable == 0:
            return "Strongly-ordered"
        if tex == 0 and cacheable == 0 and bufferable == 1:
            return "Device"
        if tex == 0 and cacheable == 1 and bufferable == 0:
            return "Normal WT"
        if tex == 0 and cacheable == 1 and bufferable == 1:
            return "Normal WB"
        if tex == 1 and cacheable == 0 and bufferable == 0:
            return "Normal NC"
        return f"tex={tex} s={shareable} c={cacheable} b={bufferable}"


ARCHITECTURES = (
    ARMv8MArchitecture(),
    ARMv7MArchitecture(),
)


@dataclass(frozen=True)
class MPUState:
    arch: MPUArchitecture
    type: int
    ctrl: int
    rnr: int
    dregion: int
    regions: List[MPURegion]
    mair0: Optional[int] = None
    mair1: Optional[int] = None
    scan_mode: str = "alias"

    @classmethod
    def from_target(cls) -> "MPUState":
        arch = _detect_architecture()
        mpu_type = utils.read_uint(MPU_TYPE)
        ctrl = utils.read_uint(MPU_CTRL)
        rnr = utils.read_uint(MPU_RNR)
        dregion = (mpu_type >> 8) & 0xFF
        extra_state = arch.read_extra_state()
        regions, scan_mode = arch.read_regions(rnr, dregion, **extra_state)
        return cls(
            arch=arch,
            type=mpu_type,
            ctrl=ctrl,
            rnr=rnr,
            dregion=dregion,
            regions=regions,
            scan_mode=scan_mode,
            **extra_state,
        )

    @property
    def enabled(self) -> bool:
        return bool(self.ctrl & 0x1)

    @property
    def privdefena(self) -> bool:
        return bool((self.ctrl >> 2) & 0x1)

    def enabled_regions(self) -> List[MPURegion]:
        return [region for region in self.regions if region.enabled]


def _detect_architecture() -> MPUArchitecture:
    try:
        for arch in ARCHITECTURES:
            if arch.matches_target():
                return arch
    except (gdb.error, AttributeError) as exc:
        raise UnsupportedArchitectureError(f"unable to determine target architecture: {exc}")

    raise UnsupportedArchitectureError("mpu only supports Cortex-M ARMv8-M and ARMv7-M targets")


def _region_number(rnr: int, offset: int, dregion: int) -> int:
    if dregion == 0:
        return offset
    return ((rnr & ~0x3) + offset) % dregion


def _perm_text(read: bool, write: bool, xn: bool) -> str:
    chars = ["R" if read else "-", "W" if write else "-", "-" if xn else "X"]
    return "".join(chars)


def format_dump(state: MPUState) -> str:
    regions = sorted(
        (region for region in state.regions if region.enabled),
        key=lambda item: (item.start, item.number),
    )
    if state.enabled:
        count = len(regions)
        region_text = (
            f"{count} visible region{'s' if count != 1 else ''}"
            if state.scan_mode == "alias"
            else f"{count} region{'s' if count != 1 else ''}"
        )
        head = (
            f"{state.arch.label} MPU: enabled, PRIVDEFENA "
            f"{'on' if state.privdefena else 'off'}, {region_text}"
        )
    else:
        head = f"{state.arch.label} MPU: disabled"

    lines = [head]
    if state.enabled and state.scan_mode == "alias":
        lines.append(f"View: alias window only (RNR={state.rnr})")

    lines.extend(
        [
            "",
            "ID  EN  START        END          SIZE        PRIV  USER  XN  ATTR",
        ]
    )

    if not state.enabled:
        lines.append("--  N   MPU disabled")
    elif not regions:
        if state.scan_mode == "alias":
            lines.append("--  N   no enabled region in readable MPU alias window")
        else:
            lines.append("--  N   no enabled region configured")
    else:
        for region in regions:
            lines.append(_format_region(region))

    return "\n".join(lines) + "\n"


def _format_region(region: MPURegion) -> str:
    priv = region.permissions.priv_text(region.xn)
    user = region.permissions.user_text(region.xn)
    xn = "Y" if region.xn else "N"
    return (
        f"{region.number:02d}  Y   {region.start:#010x}   "
        f"{region.end:#010x}   {region.size:#010x}  "
        f"{priv:<3}   {user:<3}   {xn}   {region.attr}"
    )


def check_access(
    state: MPUState, address: int, size: int, access: str, mode: str
) -> Tuple[str, str]:
    if not state.enabled:
        return "ALLOW", "MPU disabled"

    end = address + size - 1
    region = None
    for candidate in state.regions:
        if candidate.contains(address, end) and (
            region is None or candidate.number > region.number
        ):
            region = candidate

    if region is None:
        return "DENY", f"{address:#010x}..{end:#010x}: no matching enabled region"

    if region.permissions.allows(access, mode, region.xn):
        return "ALLOW", f"{address:#010x}..{end:#010x}: region {region.number}"
    return "DENY", f"{address:#010x}..{end:#010x}: denied by region {region.number}"


@autocompeletion.complete
class MPUDump(gdb.Command):
    """Dump current MPU configuration.

    Live targets may switch MPU_RNR to scan all regions.
    Read-only/core targets fall back to the visible alias window.
    """

    def get_argparser(self):
        return argparse.ArgumentParser(prog="mpu dump", add_help=False)

    def parse_args(self, arg):
        try:
            return self.parser.parse_args(gdb.string_to_argv(arg))
        except SystemExit:
            return

    def __init__(self):
        super().__init__("mpu dump", gdb.COMMAND_USER)
        self.parser = self.get_argparser()

    @utils.dont_repeat_decorator
    def invoke(self, arg: str, from_tty: bool) -> None:
        if self.parse_args(arg) is None:
            return
        gdb.write(format_dump(MPUState.from_target()))


@autocompeletion.complete
class MPUCheck(gdb.Command):
    """Check an address range against the decoded MPU state.

    Usage: mpu check <addr> [size] [read|write|exec] [priv|user]
    The underlying state collection follows the same full-scan-or-alias
    fallback path as mpu dump.
    """

    def get_argparser(self):
        parser = argparse.ArgumentParser(
            prog="mpu check",
            usage="mpu check <addr> [size] [read|write|exec] [priv|user]",
            add_help=False,
        )
        parser.add_argument("address")
        parser.add_argument("extra", nargs="*")
        return parser

    def parse_args(self, arg):
        try:
            return self.parser.parse_args(gdb.string_to_argv(arg))
        except SystemExit:
            return

    def __init__(self):
        super().__init__("mpu check", gdb.COMMAND_USER)
        self.parser = self.get_argparser()

    @utils.dont_repeat_decorator
    def invoke(self, arg: str, from_tty: bool) -> None:
        if not (parsed := self.parse_args(arg)):
            return

        address = utils.parse_arg(parsed.address)
        if address is None:
            raise gdb.GdbError(f"invalid address: {parsed.address}")
        address = int(address)
        if address < 0 or address > ADDRESS_MAX:
            raise gdb.GdbError(f"address out of 32-bit range: {parsed.address}")

        size = 1
        access = "read"
        mode = "priv"
        for token in parsed.extra:
            if token in {"read", "write", "exec"}:
                access = token
            elif token in {"priv", "user"}:
                mode = token
            elif size == 1:
                size_value = utils.parse_arg(token)
                if size_value is None:
                    raise gdb.GdbError(f"invalid size: {token}")
                size = int(size_value)
                if size < 0 or size > ADDRESS_MAX:
                    raise gdb.GdbError(f"size out of 32-bit range: {token}")
                if size == 0:
                    raise gdb.GdbError("size must be greater than zero")
            else:
                raise gdb.GdbError(f"unknown argument: {token}")

        end = address + size - 1
        if end > ADDRESS_MAX or end < address:
            raise gdb.GdbError(
                f"address range overflows 32-bit address space: {address:#010x} + {size:#x}"
            )
        state = MPUState.from_target()
        verdict, reason = check_access(state, address, size, access, mode)
        gdb.write(
            f"{verdict}: addr={address:#010x} size={size:#x} "
            f"access={access} mode={mode} - {reason}\n"
        )
