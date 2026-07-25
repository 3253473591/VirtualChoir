from __future__ import annotations

import ctypes
import sys
from ctypes import wintypes

from PySide6.QtCore import QSettings

from .ai import AIConfig


_ORGANIZATION = "VirtualChoir"
_APPLICATION = "VirtualChoir"
_CREDENTIAL_TARGET = "VirtualChoir/AI/APIKey"
_CRED_TYPE_GENERIC = 1
_CRED_PERSIST_LOCAL_MACHINE = 2


class _Credential(ctypes.Structure):
    _fields_ = [
        ("Flags", wintypes.DWORD),
        ("Type", wintypes.DWORD),
        ("TargetName", wintypes.LPWSTR),
        ("Comment", wintypes.LPWSTR),
        ("LastWritten", wintypes.FILETIME),
        ("CredentialBlobSize", wintypes.DWORD),
        ("CredentialBlob", ctypes.POINTER(ctypes.c_byte)),
        ("Persist", wintypes.DWORD),
        ("AttributeCount", wintypes.DWORD),
        ("Attributes", ctypes.c_void_p),
        ("TargetAlias", wintypes.LPWSTR),
        ("UserName", wintypes.LPWSTR),
    ]


def _settings() -> QSettings:
    return QSettings(_ORGANIZATION, _APPLICATION)


def _read_api_key() -> str:
    if sys.platform != "win32":
        return ""
    credential = ctypes.POINTER(_Credential)()
    advapi = ctypes.WinDLL("Advapi32.dll", use_last_error=True)
    read = advapi.CredReadW
    read.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.POINTER(ctypes.POINTER(_Credential))]
    read.restype = wintypes.BOOL
    if not read(_CREDENTIAL_TARGET, _CRED_TYPE_GENERIC, 0, ctypes.byref(credential)):
        return ""
    try:
        data = ctypes.string_at(credential.contents.CredentialBlob, credential.contents.CredentialBlobSize)
        return data.decode("utf-16-le")
    finally:
        advapi.CredFree(credential)


def _write_api_key(api_key: str) -> bool:
    if sys.platform != "win32":
        return False
    encoded = api_key.encode("utf-16-le")
    buffer = (ctypes.c_byte * len(encoded)).from_buffer_copy(encoded)
    credential = _Credential()
    credential.Type = _CRED_TYPE_GENERIC
    credential.TargetName = _CREDENTIAL_TARGET
    credential.CredentialBlobSize = len(encoded)
    credential.CredentialBlob = ctypes.cast(buffer, ctypes.POINTER(ctypes.c_byte))
    credential.Persist = _CRED_PERSIST_LOCAL_MACHINE
    credential.UserName = "VirtualChoir"
    advapi = ctypes.WinDLL("Advapi32.dll", use_last_error=True)
    write = advapi.CredWriteW
    write.argtypes = [ctypes.POINTER(_Credential), wintypes.DWORD]
    write.restype = wintypes.BOOL
    return bool(write(ctypes.byref(credential), 0))


def load_ai_config() -> AIConfig | None:
    settings = _settings()
    provider = settings.value("ai/provider", "gemini_native_api", type=str)
    base_url = settings.value("ai/base_url", "https://generativelanguage.googleapis.com/v1beta", type=str)
    model = settings.value("ai/model", "", type=str)
    api_key = _read_api_key()
    if not api_key and not model and provider == "gemini_native_api" and base_url == "https://generativelanguage.googleapis.com/v1beta":
        return None
    return AIConfig(provider, base_url, api_key, model)


def save_ai_config(config: AIConfig) -> bool:
    """Persist non-secret settings and store the API key in Windows Credential Manager."""
    settings = _settings()
    settings.setValue("ai/provider", config.provider)
    settings.setValue("ai/base_url", config.base_url)
    settings.setValue("ai/model", config.model)
    settings.sync()
    return _write_api_key(config.api_key)
