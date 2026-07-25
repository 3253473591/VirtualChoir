from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from .errors import ChoirError
from .models import (MicrophoneInput, Position, ProjectConfig, RoomConfig,
                     TrackConfig, _number, snap_to_grid)


@dataclass
class AIConfig:
    provider: str
    base_url: str
    api_key: str
    model: str = ""

    def validate(self, require_model: bool = False) -> None:
        if self.provider not in {"gemini_native_api", "aggregator_openai_compatible"} or not self.base_url.startswith(("http://", "https://")) or not self.api_key:
            raise ChoirError("AI_PROVIDER_CONFIG_INVALID")
        if require_model and not self.model: raise ChoirError("AI_PROVIDER_CONFIG_INVALID", "请选择模型")


@dataclass(frozen=True)
class AIAudioInput:
    track_id: str
    note: str
    wav_bytes: bytes


class AIClient:
    def __init__(self, config: AIConfig, timeout_s: float = 120): self.config, self.timeout_s = config, timeout_s

    @property
    def url(self) -> str: return self.config.base_url.rstrip("/")

    def fetch_models(self) -> tuple[list[str], str]:
        self.config.validate()
        try:
            if self.config.provider == "gemini_native_api":
                response = requests.get(f"{self.url}/models", params={"key": self.config.api_key}, timeout=self.timeout_s)
                body = response.json(); response.raise_for_status()
                models = [m["name"].removeprefix("models/") for m in body.get("models", []) if "generateContent" in m.get("supportedGenerationMethods", [])]
            else:
                response = requests.get(f"{self.url}/models", headers={"Authorization": f"Bearer {self.config.api_key}"}, timeout=self.timeout_s)
                body = response.json(); response.raise_for_status(); models = [m["id"] for m in body.get("data", []) if isinstance(m, dict) and m.get("id")]
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code in (401, 403): raise ChoirError("AI_AUTH_FAILED") from exc
            raise ChoirError("AI_MODEL_LIST_FETCH_FAILED", str(exc)) from exc
        except (requests.RequestException, ValueError, KeyError) as exc: raise ChoirError("AI_MODEL_LIST_FETCH_FAILED", str(exc)) from exc
        if not models: raise ChoirError("AI_MODEL_LIST_FETCH_FAILED", "未发现可用模型")
        return sorted(set(models)), datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    def analyze_audio_json(self, audio_inputs: list[AIAudioInput], project: ProjectConfig) -> dict[str, Any]:
        self.config.validate(require_model=True)
        expected = [(track.track_id, track.file_name) for track in project.tracks if track.enabled]
        actual = [(item.track_id, item.note) for item in audio_inputs]
        if not audio_inputs or actual != expected or not all(isinstance(item.wav_bytes, bytes) and item.wav_bytes for item in audio_inputs):
            raise ChoirError("AI_REQUEST_FAILED", "AI 音频列表必须与全部启用音轨按顺序完全一致")
        prompt = _prompt(project)
        try:
            if self.config.provider == "gemini_native_api":
                parts: list[dict[str, Any]] = [{"text": prompt}]
                for index, item in enumerate(audio_inputs, start=1):
                    parts.extend([
                        {"text": _audio_label(index, item)},
                        {"inline_data": {"mime_type": "audio/wav", "data": base64.b64encode(item.wav_bytes).decode("ascii")}},
                    ])
                payload = {"contents": [{"parts": parts}], "generationConfig": {"responseMimeType": "application/json"}}
                response = requests.post(f"{self.url}/models/{self.config.model}:generateContent", params={"key": self.config.api_key}, json=payload, timeout=self.timeout_s)
                body = response.json(); response.raise_for_status(); content = body["candidates"][0]["content"]["parts"][0]["text"]
            else:
                message_content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
                for index, item in enumerate(audio_inputs, start=1):
                    message_content.extend([
                        {"type": "text", "text": _audio_label(index, item)},
                        {"type": "input_audio", "input_audio": {"data": base64.b64encode(item.wav_bytes).decode("ascii"), "format": "wav"}},
                    ])
                payload = {"model": self.config.model, "messages": [{"role": "user", "content": message_content}], "response_format": {"type": "json_object"}}
                response = requests.post(f"{self.url}/chat/completions", headers={"Authorization": f"Bearer {self.config.api_key}"}, json=payload, timeout=self.timeout_s)
                body = response.json(); response.raise_for_status(); content = body["choices"][0]["message"]["content"]
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code in (401, 403): raise ChoirError("AI_AUTH_FAILED") from exc
            raise ChoirError("AI_REQUEST_FAILED", str(exc)) from exc
        except (requests.RequestException, ValueError, KeyError, IndexError) as exc: raise ChoirError("AI_REQUEST_FAILED", str(exc)) from exc
        return validate_ai_response(content, project)

    def customize_json(self, messages: list[dict[str, str]], project: ProjectConfig) -> dict[str, Any]:
        """Turn multi-turn user feedback into validated, directly applicable options."""
        self.config.validate(require_model=True)
        if not messages or not all(
            isinstance(item, dict) and set(item) == {"role", "content"}
            and item["role"] in {"user", "assistant"}
            and isinstance(item["content"], str) and item["content"].strip()
            for item in messages
        ):
            raise ChoirError("AI_REQUEST_FAILED", "对话历史格式无效")
        instruction = _customization_prompt(project)
        try:
            if self.config.provider == "gemini_native_api":
                contents = [{"role": "user", "parts": [{"text": instruction}]}]
                contents.extend({
                    "role": "model" if item["role"] == "assistant" else "user",
                    "parts": [{"text": item["content"]}],
                } for item in messages)
                payload = {
                    "contents": contents,
                    "generationConfig": {"responseMimeType": "application/json"},
                }
                response = requests.post(
                    f"{self.url}/models/{self.config.model}:generateContent",
                    params={"key": self.config.api_key}, json=payload, timeout=self.timeout_s,
                )
                body = response.json(); response.raise_for_status()
                content = body["candidates"][0]["content"]["parts"][0]["text"]
            else:
                payload = {
                    "model": self.config.model,
                    "messages": [{"role": "system", "content": instruction}] + messages,
                    "response_format": {"type": "json_object"},
                }
                response = requests.post(
                    f"{self.url}/chat/completions",
                    headers={"Authorization": f"Bearer {self.config.api_key}"},
                    json=payload, timeout=self.timeout_s,
                )
                body = response.json(); response.raise_for_status()
                content = body["choices"][0]["message"]["content"]
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code in (401, 403):
                raise ChoirError("AI_AUTH_FAILED") from exc
            raise ChoirError("AI_REQUEST_FAILED", str(exc)) from exc
        except (requests.RequestException, ValueError, KeyError, IndexError) as exc:
            raise ChoirError("AI_REQUEST_FAILED", str(exc)) from exc
        return validate_ai_chat_response(content, project)


def validate_ai_response(raw: str | dict[str, Any], project: ProjectConfig) -> dict[str, Any]:
    if isinstance(raw, str):
        if raw.strip().startswith("```"): raise ChoirError("AI_RESPONSE_PARSE_ERROR", "不允许 Markdown 围栏")
        try: data = json.loads(raw)
        except json.JSONDecodeError as exc: raise ChoirError("AI_RESPONSE_PARSE_ERROR", str(exc)) from exc
    else: data = raw
    if not isinstance(data, dict):
        raise ChoirError("AI_RESPONSE_SCHEMA_ERROR")
    # Accept the old single-suggestion payload for backward compatibility, but
    # normalize it to the current multi-option contract before presenting it.
    if set(data) == {"recommendations"}:
        recommendations = data["recommendations"]
        if not isinstance(recommendations, list) or not 1 <= len(recommendations) <= 5:
            raise ChoirError("AI_RESPONSE_SCHEMA_ERROR", "recommendations")
        return {"recommendations": [_validate_recommendation(item, project) for item in recommendations]}
    return {"recommendations": [_validate_legacy_suggestion(data, project)]}


def validate_ai_chat_response(raw: str | dict[str, Any], project: ProjectConfig) -> dict[str, Any]:
    if isinstance(raw, str):
        if raw.strip().startswith("```"):
            raise ChoirError("AI_RESPONSE_PARSE_ERROR", "不允许 Markdown 围栏")
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ChoirError("AI_RESPONSE_PARSE_ERROR", str(exc)) from exc
    else:
        data = raw
    if not isinstance(data, dict) or set(data) != {"message", "recommendations"}:
        raise ChoirError("AI_RESPONSE_SCHEMA_ERROR", "定制对话根字段")
    if not isinstance(data["message"], str) or not data["message"].strip():
        raise ChoirError("AI_RESPONSE_SCHEMA_ERROR", "message")
    validated = validate_ai_response({"recommendations": data["recommendations"]}, project)
    return {"message": data["message"].strip(), **validated}


def _validate_recommendation(data: Any, project: ProjectConfig) -> dict[str, Any]:
    required = {"id", "name", "description", "room", "microphone", "singers"}
    if not isinstance(data, dict) or set(data) != required:
        raise ChoirError("AI_RESPONSE_SCHEMA_ERROR", "recommendation")
    if not all(isinstance(data[key], str) and data[key].strip() for key in ("id", "name", "description")):
        raise ChoirError("AI_RESPONSE_SCHEMA_ERROR", "recommendation metadata")
    suggestion = _validate_values(data, project)
    return {"id": data["id"], "name": data["name"], "description": data["description"], **suggestion}


def _validate_legacy_suggestion(data: Any, project: ProjectConfig) -> dict[str, Any]:
    required = {"schema_version", "project_id", "room", "microphone", "singers", "analysis", "confidence", "warnings"}
    if not isinstance(data, dict) or set(data) != required or data.get("schema_version") != 1:
        raise ChoirError("AI_RESPONSE_SCHEMA_ERROR")
    if not isinstance(data["project_id"], str) or not data["project_id"]:
        raise ChoirError("AI_RESPONSE_SCHEMA_ERROR", "project_id")
    if not isinstance(data["analysis"], dict) or set(data["analysis"]) != {"overall_character", "recommended_scene", "reason"} or not all(isinstance(value, str) for value in data["analysis"].values()):
        raise ChoirError("AI_RESPONSE_SCHEMA_ERROR", "analysis")
    _number(data["confidence"], "confidence", 0, 1)
    if not isinstance(data["warnings"], list) or not all(isinstance(value, str) for value in data["warnings"]):
        raise ChoirError("AI_RESPONSE_SCHEMA_ERROR", "warnings")
    legacy_values = dict(data)
    local_notes = {track.track_id: track.file_name for track in project.tracks}
    legacy_values["singers"] = [
        {**singer, "note": local_notes.get(singer.get("track_id"), "")}
        for singer in data["singers"]
    ]
    suggestion = _validate_values(legacy_values, project)
    return {
        "id": "ai_recommendation",
        "name": data["analysis"]["recommended_scene"],
        "description": data["analysis"]["reason"],
        **suggestion,
    }


def _validate_values(data: dict[str, Any], project: ProjectConfig) -> dict[str, Any]:
    try:
        room = data["room"]
        legacy_room_keys = {"rt60_s", "reverb_gain_db"}
        adjustable_room_keys = {"length_m", "width_m", "height_m", "rt60_s", "reverb_gain_db"}
        if set(room) == legacy_room_keys:
            room_config = RoomConfig(
                length_m=project.room.length_m,
                width_m=project.room.width_m,
                height_m=project.room.height_m,
                rt60_s=room["rt60_s"],
                reverb_gain_db=room["reverb_gain_db"],
                bus_gain_db=project.room.bus_gain_db,
                grid_step_m=project.room.grid_step_m,
            )
        elif set(room) == adjustable_room_keys:
            room_config = RoomConfig(
                length_m=room["length_m"], width_m=room["width_m"], height_m=room["height_m"],
                rt60_s=room["rt60_s"], reverb_gain_db=room["reverb_gain_db"],
                bus_gain_db=project.room.bus_gain_db,
                grid_step_m=min(project.room.grid_step_m, room["length_m"], room["width_m"]),
            )
        else:
            raise ValueError("room fields")
        room_config.validate()
        room = {
            "length_m": room_config.length_m,
            "width_m": room_config.width_m,
            "height_m": room_config.height_m,
            "rt60_s": room_config.rt60_s,
            "reverb_gain_db": room_config.reverb_gain_db,
        }
        mic = data["microphone"]
        if set(mic) != {"count", "spacing_m", "height_m"}: raise ValueError("microphone 不得包含坐标")
        MicrophoneInput(**mic).validate(room_config)
        if not isinstance(data["singers"], list): raise ValueError("singers")
        local_notes = {track.track_id: track.file_name for track in project.tracks}
        ids = set()
        singers = []
        for singer in data["singers"]:
            if set(singer) != {"track_id", "note", "position", "gain_db"}: raise ValueError("singer")
            if not isinstance(singer["note"], str) or singer["note"] != local_notes.get(singer["track_id"]):
                raise ChoirError("AI_TRACK_MISMATCH", f"{singer.get('track_id')}.note 与本地文件名不一致")
            position = Position.from_dict(singer["position"])
            temp = TrackConfig(singer["track_id"], "AI", position, singer["gain_db"])
            try:
                # Check the original value before normalization.  An AI value
                # outside the room must never be pulled back to an edge node.
                temp.validate(room_config)
            except ChoirError as exc:
                if exc.code == "INVALID_COORDINATE":
                    raise ChoirError(
                        "AI_VALUE_OUT_OF_RANGE",
                        f"AI returned {singer['track_id']}.position.{exc.detail} outside room bounds; proposal was rejected.",
                    ) from exc
                raise
            # X/Y are a floor-plan coordinate system.  AI responses are
            # intentionally normalized here, rather than trusted as-is, so
            # applying a proposal has the same grid-snap contract as dragging
            # a singer in the room view.  Z is not a floor-plan coordinate and
            # must remain untouched.
            position.x_m = snap_to_grid(position.x_m, room_config.width_m, room_config.grid_step_m)
            position.y_m = snap_to_grid(position.y_m, room_config.length_m, room_config.grid_step_m)
            temp = TrackConfig(singer["track_id"], "AI", position, singer["gain_db"])
            try:
                temp.validate(room_config)
            except ChoirError as exc:
                if exc.code == "INVALID_COORDINATE":
                    raise ChoirError(
                        "AI_VALUE_OUT_OF_RANGE",
                        f"AI 返回 {singer['track_id']}.position.{exc.detail} 超出当前房间边界，已拒绝应用。",
                    ) from exc
                raise
            ids.add(temp.track_id)
            singers.append({
                "track_id": temp.track_id,
                "note": singer["note"],
                "position": {"x_m": position.x_m, "y_m": position.y_m, "z_m": position.z_m},
                "gain_db": temp.gain_db,
            })
        if ids != {track.track_id for track in project.tracks} or len(ids) != len(data["singers"]): raise ChoirError("AI_TRACK_MISMATCH")
    except ChoirError: raise
    except (TypeError, ValueError, KeyError) as exc: raise ChoirError("AI_RESPONSE_SCHEMA_ERROR", str(exc)) from exc
    return {"room": room, "microphone": mic, "singers": singers}


def _analysis_context(project: ProjectConfig) -> dict[str, Any]:
    """Return the complete current spatial state supplied alongside the audio."""
    return {
        "room": {
            "length_m": project.room.length_m,
            "width_m": project.room.width_m,
            "height_m": project.room.height_m,
            "rt60_s": project.room.rt60_s,
            "reverb_gain_db": project.room.reverb_gain_db,
            "grid_step_m": project.room.grid_step_m,
            "x_step_m": project.room.grid_step_m,
            "y_step_m": project.room.grid_step_m,
        },
        "microphone": {
            "count": project.microphone.count,
            "spacing_m": project.microphone.spacing_m,
            "height_m": project.microphone.height_m,
        },
        "singers": [
            {
                "track_id": track.track_id,
                "note": track.file_name,
                "position": {
                    "x_m": track.position.x_m,
                    "y_m": track.position.y_m,
                    "z_m": track.position.z_m,
                },
                "gain_db": track.gain_db,
                "enabled": track.enabled,
            }
            for track in project.tracks
        ],
    }


def _audio_label(index: int, item: AIAudioInput) -> str:
    return f"音频附件 {index}：track_id={item.track_id}，note={item.note}。"


def _variation_context(project: ProjectConfig) -> str:
    notes = []
    for track in project.tracks:
        if track.parent_source:
            notes.append(
                f"{track.file_name} 是基于 {Path(track.parent_source).name} 复制并轻微差异化得到的副本。"
            )
    return "\n".join(notes) if notes else "当前没有差异化副本。"


def _prompt(project: ProjectConfig) -> str:
    tracks = ", ".join(track.track_id for track in project.tracks)
    context = json.dumps(_analysis_context(project), ensure_ascii=False, separators=(",", ":"))
    variation_context = _variation_context(project)
    return f"""只返回无 Markdown 围栏的合法 JSON。全部附件均为未空间渲染的干声；若工程启用了随机起音偏移，附件已应用该偏移。附件不包含房间、麦克风、位置或增益渲染。音频已转换为 44100Hz/16-bit/单声道；每段最多 10 秒，仅截取有声音频，不足时会拼接多个有声段。每个附件前的标签给出对应 track_id 和只读文件名 note。差异化关系：{variation_context}。请综合分析全部附件，并基于以下当前工程数据提出 3 套可选空间方案：{context}
根对象只能包含 recommendations：{{"recommendations":[{{"id":"stable_id","name":"方案名","description":"简短说明","room":{{"length_m":number,"width_m":number,"height_m":number,"rt60_s":number,"reverb_gain_db":number}},"microphone":{{"count":integer,"spacing_m":number,"height_m":number}},"singers":[{{"track_id":"...","note":"原文件名.wav","position":{{"x_m":number,"y_m":number,"z_m":number}},"gain_db":number}}]}}]}}。
每套方案必须且只能包含 id、name、description、room、microphone、singers。轨道 ID 必须恰好为 [{tracks}]，每个 singers.note 必须逐字等于当前工程中该 track_id 的 note；note 是只读文件名，严禁改写。
可调整房间长宽高、RT60、混响增益、麦克风数量/间距/高度以及歌手位置/增益。范围：length_m、width_m 在 (0,100]，height_m 在 (0,30]，rt60_s 在 [0.2,2.0] 秒，reverb_gain_db 在 [-30,0] dB，歌手 gain_db 在 [-60,12] dB。房间尺寸与 RT60 相互独立，不得因扩大房间而自动延长 RT60。
麦克风 count 必须是 2 至 6 的整数，spacing_m 表示相邻麦克风间距且必须满足 0.2 < spacing_m < min(3,width_m/(count-1))，麦克风 height_m 必须在 (0,房间 height_m)。阵列整体中心固定在 width_m/2，全部麦克风 Y=0；从 0 开始的第 i 个麦克风 X=width_m/2+(i-(count-1)/2)*spacing_m。坐标由程序自动计算，microphone 只能返回 count、spacing_m、height_m，严禁返回任何坐标。
歌手坐标必须按各方案返回的房间尺寸满足 0<=x_m<=width_m、0<=y_m<=length_m、0<=z_m<=height_m。x_m、y_m 使用 {project.room.grid_step_m}m 网格且房间边界也是有效节点；z_m 不吸附网格。grid_step_m、x_step_m、y_step_m 仅为输入约束，不得返回。所有数值必须是 JSON number，输出前逐一核对边界。"""


def _customization_prompt(project: ProjectConfig) -> str:
    tracks = ", ".join(track.track_id for track in project.tracks)
    context = json.dumps(_analysis_context(project), ensure_ascii=False, separators=(",", ":"))
    return f"""你是虚拟合唱空间混音助手。用户会用多轮对话描述听感和目标，例如右声道能量偏高、房间过小、混响太长。请先用简洁中文说明判断和修改依据，再输出 1 至 3 套可直接应用的定制方案。用户反馈是本轮调整依据；不要声称你在本轮重新听取了音频，本轮不会发送音频。
当前工程数据：{context}
只返回无 Markdown 围栏的合法 JSON，根对象必须且只能包含 message 和 recommendations：{{"message":"对用户的中文回复","recommendations":[{{"id":"stable_id","name":"方案名","description":"具体改动说明","room":{{"length_m":number,"width_m":number,"height_m":number,"rt60_s":number,"reverb_gain_db":number}},"microphone":{{"count":integer,"spacing_m":number,"height_m":number}},"singers":[{{"track_id":"...","note":"原文件名.wav","position":{{"x_m":number,"y_m":number,"z_m":number}},"gain_db":number}}]}}]}}。
轨道 ID 必须恰好为 [{tracks}]，每套方案必须包含全部轨道；每个 note 必须逐字保持为当前工程文件名，严禁改写。房间范围：length_m、width_m 在 (0,100]，height_m 在 (0,30]，rt60_s 在 [0.2,2.0]，reverb_gain_db 在 [-30,0]。麦克风 count 是 2 至 6 的整数，spacing_m 表示相邻间距并满足 0.2 < spacing_m < min(3,width_m/(count-1))，麦克风 height_m 在 (0,房间 height_m)。阵列整体中心为 width_m/2、全部 Y=0，第 i 个麦克风 X=width_m/2+(i-(count-1)/2)*spacing_m；程序自动计算坐标，严禁返回坐标。歌手 gain_db 在 [-60,12]，位置必须在该方案房间边界内。x_m、y_m 使用 {project.room.grid_step_m}m 网格，边界也可作为节点；z_m 不吸附网格。房间尺寸与 RT60 相互独立。"""
