from __future__ import annotations

from dataclasses import dataclass
from typing import Any


MESSAGES = {
    "PROJECT_NOT_FOUND": "工程文件不存在或路径不可读", "PROJECT_PARSE_ERROR": "工程 JSON 无法解析",
    "PROJECT_SCHEMA_ERROR": "工程字段或类型不符合 schema", "PROJECT_VERSION_UNSUPPORTED": "工程 schema_version 不受支持",
    "AUDIO_NOT_FOUND": "音频文件不存在", "AUDIO_PERMISSION_DENIED": "音频文件无读取权限",
    "AUDIO_FORMAT_UNSUPPORTED": "不是支持的 WAV 参数", "AUDIO_EMPTY": "音频没有有效样本",
    "AUDIO_NON_FINITE": "音频包含 NaN 或 Infinity", "TRACK_LIMIT_EXCEEDED": "轨道数量超过 256",
    "DUPLICATE_TRACK_ID": "track_id 重复", "INVALID_COORDINATE": "坐标不是有限数或超出房间边界",
    "INVALID_ROOM": "房间参数不合法", "INVALID_MICROPHONE": "麦克风参数不合法",
    "AI_RESPONSE_PARSE_ERROR": "AI 返回内容不是合法 JSON", "AI_RESPONSE_SCHEMA_ERROR": "AI 返回 JSON 不符合建议格式",
    "AI_TRACK_MISMATCH": "AI 返回的 track_id 与本地轨道不一致", "AI_VALUE_OUT_OF_RANGE": "AI 建议值超出安全边界",
    "AI_PROVIDER_CONFIG_INVALID": "AI 接入配置不完整或地址无效", "AI_MODEL_LIST_FETCH_FAILED": "动态模型列表拉取失败",
    "AI_MODEL_UNSUPPORTED": "所选模型不支持音频输入或内容生成", "AI_AUDIO_PREPROCESS_FAILED": "发送给 Gemini 的音频预处理失败",
    "AI_AUTH_FAILED": "AI API 密钥无效或无权限", "AI_REQUEST_FAILED": "AI 分析请求失败",
    "RENDER_CANCELLED": "渲染被用户取消", "RENDER_DEPENDENCY_MISSING": "渲染依赖不可用",
    "RENDER_FAILED": "渲染过程失败", "AUDIO_CLIP_RISK": "输出峰值可能超过 0 dBFS",
    "DISK_SPACE_LOW": "输出磁盘空间不足", "OUTPUT_WRITE_FAILED": "输出文件无法写入",
    "NATURALIZATION_INPUT_MISSING": "随机偏移缺少带歌词 MIDI",
    "NATURALIZATION_LANGUAGE_MISSING": "随机偏移配置无效",
    "NATURALIZATION_MIDI_INVALID": "MIDI 歌词无法解析或没有有效歌词事件",
    "NATURALIZATION_TRACK_MISMATCH": "MIDI 分配包含不存在的轨道",
    "NATURALIZATION_ALIGNMENT_FAILED": "MIDI 歌词与音频无法可靠对齐",
    "NATURALIZATION_PROCESS_FAILED": "随机偏移处理失败，已回退原始音频",
    "PLAYBACK_DEVICE_ERROR": "播放设备不可用", "UNSAVED_CHANGES": "工程存在未保存修改",
}


@dataclass
class ChoirError(Exception):
    code: str
    detail: Any = None
    recoverable: bool = True
    action: str | None = None

    @property
    def message(self) -> str:
        return MESSAGES.get(self.code, self.code)

    def __str__(self) -> str:
        suffix = f"：{self.detail}" if self.detail else ""
        return f"{self.code} - {self.message}{suffix}"
