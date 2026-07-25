"""Backward-compatible imports for the structured Qt UI package.

New code may import components from :mod:`virtual_choir.ui` directly.
"""

from .ui.dialogs import (
    AIConversationDialog,
    AISettingsDialog,
    AISuggestionDialog,
    DuplicateTrackDialog,
    MidiAssignmentDialog,
    NaturalizationDialog,
)
from .ui.feedback import TaskOverlay, Toast
from .ui.main_window import MainWindow
from .ui.room import MicItem, RoomView, SingerItem, _grid_nodes, _snap
from .ui.theme import Colors, DEFAULT_PROJECT_DIR, Fonts, Radii, Spacing
from .ui.tracks import BatchTrackDialog, CollapsibleCard, TrackCard, TrackPanel
from .ui.widgets import ParameterSpinBox, PreviewSlider
from .ui.workers import AIAnalysisWorker, AIChatWorker, DuplicateWorker, RenderWorker
