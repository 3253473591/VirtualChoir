from __future__ import annotations

from pathlib import Path


class Colors:
    BG_PRIMARY      = "#ffffff"
    BG_SECONDARY    = "#f6f8fa"
    BG_TERTIARY     = "#eaeef2"
    BORDER          = "#d1d9e0"
    BORDER_STRONG   = "#b7bec5"
    TEXT_PRIMARY    = "#1f2328"
    TEXT_SECONDARY  = "#656d76"
    TEXT_DISABLED   = "#8c959f"
    ACCENT          = "#0969da"
    ACCENT_HOVER    = "#0550ae"
    ACCENT_LIGHT    = "#ddf4ff"
    SUCCESS         = "#1a7f37"
    SUCCESS_LIGHT   = "#dafbe1"
    WARNING         = "#9a6700"
    WARNING_LIGHT   = "#fff8c5"
    DANGER          = "#cf222e"
    DANGER_LIGHT    = "#ffebe9"
    MIC_AMBER       = "#f59e0b"
    MIC_AMBER_DARK  = "#92400e"
    SINGER_FILL     = "#0969da"
    SINGER_DISABLED = "#8c959f"
    GRID_MAJOR      = "#d1d9e0"
    GRID_MINOR      = "#eaeef2"
    ROOM_BORDER     = "#24292f"
    SNAP_GUIDE      = "#1a7f37"


class Fonts:
    FAMILY         = '"Microsoft YaHei UI", "Segoe UI", "PingFang SC", sans-serif'
    MONO           = '"SF Mono", "Consolas", monospace'
    TITLE_SIZE     = 14
    PANEL_SIZE     = 13
    BODY_SIZE      = 12
    SMALL_SIZE     = 11
    TINY_SIZE      = 10


class Spacing:
    XS  = 4
    SM  = 8
    MD  = 12
    LG  = 16
    XL  = 24


class Radii:
    SM = 4
    MD = 6
    LG = 8


DEFAULT_PROJECT_DIR = Path.cwd() / "project"
