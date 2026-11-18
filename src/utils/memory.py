"""Zero-dependency cross-platform memory diagnostics for Krishi Sahayyak.

Provides real-time process RSS (Resident Set Size) and peak memory tracking
for Windows (development) and Linux (Render container deployment) using
only standard library modules (ctypes, resource, /proc/self/status).
"""
from __future__ import annotations

import logging
import os
import sys

logger = logging.getLogger("src.utils.memory")


def get_process_memory_mb() -> float:
    """Return current process resident set size (RSS) in megabytes."""
    try:
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes

            class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            pmc = PROCESS_MEMORY_COUNTERS()
            pmc.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
            kernel32 = ctypes.windll.kernel32
            psapi = ctypes.windll.psapi
            kernel32.GetCurrentProcess.restype = wintypes.HANDLE
            psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESS_MEMORY_COUNTERS), wintypes.DWORD]
            psapi.GetProcessMemoryInfo.restype = wintypes.BOOL

            handle = kernel32.GetCurrentProcess()
            if psapi.GetProcessMemoryInfo(handle, ctypes.byref(pmc), pmc.cb):
                return round(pmc.WorkingSetSize / (1024.0 * 1024.0), 2)
            return 0.0

        # Linux (Render container environment)
        try:
            with open("/proc/self/status", "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        # Value in kB
                        parts = line.split()
                        if len(parts) >= 2:
                            return round(float(parts[1]) / 1024.0, 2)
        except (OSError, ValueError):
            pass

        # Fallback for POSIX systems
        import resource
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # On Linux ru_maxrss is in kB, on macOS in bytes
        if sys.platform == "darwin":
            return round(usage / (1024.0 * 1024.0), 2)
        return round(usage / 1024.0, 2)
    except Exception as exc:
        logger.debug("Could not read process memory: %s", exc)
        return 0.0


def get_peak_memory_mb() -> float:
    """Return peak process memory usage in megabytes."""
    try:
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes

            class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            pmc = PROCESS_MEMORY_COUNTERS()
            pmc.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
            kernel32 = ctypes.windll.kernel32
            psapi = ctypes.windll.psapi
            kernel32.GetCurrentProcess.restype = wintypes.HANDLE
            psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESS_MEMORY_COUNTERS), wintypes.DWORD]
            psapi.GetProcessMemoryInfo.restype = wintypes.BOOL

            handle = kernel32.GetCurrentProcess()
            if psapi.GetProcessMemoryInfo(handle, ctypes.byref(pmc), pmc.cb):
                return round(pmc.PeakWorkingSetSize / (1024.0 * 1024.0), 2)
            return 0.0

        try:
            with open("/proc/self/status", "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("VmHWM:"):
                        parts = line.split()
                        if len(parts) >= 2:
                            return round(float(parts[1]) / 1024.0, 2)
        except (OSError, ValueError):
            pass

        import resource
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            return round(usage / (1024.0 * 1024.0), 2)
        return round(usage / 1024.0, 2)
    except Exception as exc:
        logger.debug("Could not read peak memory: %s", exc)
        return 0.0


def log_memory(tag: str, log: logging.Logger | None = None) -> float:
    """Log current process memory footprint with a diagnostic tag."""
    target_log = log or logger
    current_mb = get_process_memory_mb()
    peak_mb = get_peak_memory_mb()
    target_log.info("[Memory] %s: Current RSS = %.1f MB (Peak = %.1f MB)", tag, current_mb, peak_mb)
    return current_mb
