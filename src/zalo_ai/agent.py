"""Tạo hội thoại ReAct và thực thi tool cho một mẫu LifeMate."""

from __future__ import annotations

import json
import re
from typing import Any, Callable


ASK_TOOL = "sys.ask_missing_info"


def build_tools_spec(
    installed_apps: dict[str, Any], catalog_tools: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    """Tạo schema OpenAI chỉ cho tool thuộc ứng dụng đã cài."""
    allowed_tools = []
    seen = set()
    for app_info in installed_apps.values():
        for tool_name in app_info.get("tools", []):
            if tool_name in catalog_tools and tool_name not in seen:
                seen.add(tool_name)
                allowed_tools.append(_tool_spec(catalog_tools[tool_name]))
    if ASK_TOOL in catalog_tools and ASK_TOOL not in seen:
        allowed_tools.append(_tool_spec(catalog_tools[ASK_TOOL]))
    return allowed_tools


def _tool_spec(tool: dict[str, Any]) -> dict[str, Any]:
    """Chuyển một mục catalog thành schema function của OpenAI."""
    return {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "parameters": tool.get("parameters", {"type": "object", "properties": {}}),
        },
    }


def clean_arguments(args: Any, schema_properties: dict[str, Any]) -> dict[str, Any]:
    """Đưa kiểu tham số model trả về về dạng schema chấp nhận."""
    if not isinstance(args, dict):
        raise TypeError("Tool arguments phải là một JSON object.")
    cleaned = dict(args)
    for key, value in cleaned.items():
        prop = schema_properties.get(key, {})
        if prop.get("type") == "integer" and isinstance(value, str) and value.isdigit():
            cleaned[key] = int(value)
        if prop.get("format") == "date-time" and isinstance(value, str):
            value = value.replace(" ", "T")
            if not re.search(r"(Z|[+-]\d{2}:?\d{2})$", value):
                value += "+07:00"
            cleaned[key] = value
    return cleaned


def build_system_prompt(context: dict[str, Any]) -> str:
    """Prompt dùng chung cho inference và SFT."""
    return f"""Bạn là LifeMate - trợ lý AI cá nhân tại Việt Nam có nhiệm vụ thực hiện hành động dựa trên yêu cầu người dùng.
Thông tin bối cảnh hiện tại:
- Thời gian hiện tại: {context.get('current_time')}
- Múi giờ: {context.get('timezone', 'Asia/Ho_Chi_Minh')}
- Vị trí hiện tại: {context.get('location', {})}
- Hồ sơ người dùng: {json.dumps(context.get('user_profile', {}), ensure_ascii=False)}

Nguyên tắc thực thi bắt buộc:
1. TUYỆT ĐỐI KHÔNG TỰ BỊA ID (contact_id, event_id, device_id). BẮT BUỘC gọi tool tra cứu (read/search) trước rồi mới gọi tool ghi (create/update/delete).
2. Nếu câu lệnh thiếu tham số bắt buộc không thể tự tra cứu, BẮT BUỘC gọi tool 'sys.ask_missing_info' với 'tool_name' và danh sách 'params' còn thiếu.
3. Nếu người dùng đổi ý ('à thôi', 'không cần nữa') hoặc yêu cầu ngoài phạm vi, KHÔNG gọi bất kỳ tool thay đổi dữ liệu nào.
4. Mọi tham số thời gian (start, end, time) phải tính toán theo chuẩn ISO-8601 dựa trên thời gian hiện tại."""


def run_agent(
    item: dict[str, Any],
    env: Any,
    *,
    catalog_tools: dict[str, dict[str, Any]],
    eval_ids: set[str] | None,
    transcribe: Callable[[str], tuple[str, list[str]]],
    query_model: Callable[[list[dict[str, Any]], list[dict[str, Any]]], dict[str, Any] | None],
) -> None:
    """Thực hiện ASR, tạo ngữ cảnh hội thoại rồi chạy vòng lặp tool."""
    context = item.get("context", {})
    tools = build_tools_spec(context.get("installed_apps", {}), catalog_tools)
    system_prompt = build_system_prompt(context)
    messages = [{"role": "system", "content": system_prompt}]

    # Một mẫu được giải mã theo batch gồm các lượt user trước đó và lượt cuối.
    current_request, history_requests = transcribe(item["id"])
    user_turn_index = 0
    for turn in item.get("history", []):
        role = turn.get("role")
        if role == "user":
            asr_text = history_requests[user_turn_index] if user_turn_index < len(history_requests) else ""
            user_turn_index += 1
            text = asr_text if eval_ids is not None else (turn.get("audio_text") or asr_text)
            messages.append({"role": "user", "content": text or ""})
        elif role == "assistant":
            assistant_message = {"role": "assistant", "content": turn.get("text", "")}
            if turn.get("tool_calls"):
                assistant_message["tool_calls"] = [
                    {
                        "id": f"call_hist_{index}",
                        "type": "function",
                        "function": {
                            "name": call["name"],
                            "arguments": json.dumps(call["arguments"], ensure_ascii=False),
                        },
                    }
                    for index, call in enumerate(turn["tool_calls"])
                ]
            messages.append(assistant_message)
            for index, (_, result) in enumerate(zip(
                turn.get("tool_calls", []), turn.get("tool_results", [])
            )):
                result_text = json.dumps(result, ensure_ascii=False)
                if len(result_text) > 2000:
                    # Giữ lịch sử tool gọn để prompt không tăng quá nhanh.
                    result_text = result_text[:2000] + "...(truncated)"
                messages.append({
                    "role": "tool",
                    "tool_call_id": f"call_hist_{index}",
                    "content": result_text,
                })

    # Train dùng transcript nhãn theo mặc định. Khi có danh sách ID, chạy ASR.
    latest_text = (
        current_request if eval_ids is not None
        else item.get("audio_text") or current_request
    )
    messages.append({"role": "user", "content": latest_text or ""})

    # Giới hạn số vòng tool để một mẫu không mắc kẹt trong hội thoại quá dài.
    previous_calls = None
    for step in range(6):
        response = query_model(messages, tools)
        if not response or not response.get("choices"):
            break
        assistant_message = response["choices"][0].get("message", {})
        messages.append(assistant_message)
        tool_calls = assistant_message.get("tool_calls", [])
        # Không thực thi lại một nhóm tool call nếu model trả lại nguyên nhóm cũ.
        if not tool_calls or tool_calls == previous_calls:
            break
        previous_calls = tool_calls

        for tool_call in tool_calls:
            function = tool_call.get("function", {})
            name = function.get("name", "")
            raw_args = function.get("arguments", {})
            call_id = tool_call.get("id", f"call_{step}")
            if isinstance(raw_args, str):
                raw_args = json.loads(raw_args)

            tool = catalog_tools.get(name, {})
            properties = (tool.get("parameters") or {}).get("properties") or {}
            args = clean_arguments(raw_args, properties)

            # validate chỉ kiểm tra schema, tránh ghi call không hợp lệ vào log ToolEnv.
            check = env.validate(name, args)
            if not check.get("ok"):
                messages.append({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": json.dumps({"ok": False, "error": check.get("error")}, ensure_ascii=False),
                })
                continue

            result = env.call(name, args)
            messages.append({
                "role": "tool",
                "tool_call_id": call_id,
                "content": json.dumps(result, ensure_ascii=False),
            })
            if name == ASK_TOOL:
                return
