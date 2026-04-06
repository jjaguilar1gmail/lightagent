from __future__ import annotations

import json
from typing import Any, TypeVar

from langchain_core.messages import AIMessage, BaseMessage
from pydantic import BaseModel, ValidationError

SchemaT = TypeVar("SchemaT", bound=BaseModel)


def schema_instruction(schema: type[SchemaT]) -> str:
    json_schema = json.dumps(schema.model_json_schema(), indent=2)
    return (
        "Return exactly one valid JSON object and no surrounding commentary. "
        "The JSON must conform to this schema:\n"
        f"{json_schema}"
    )


def validate_structured_output(raw: Any, schema: type[SchemaT]) -> SchemaT:
    if isinstance(raw, schema):
        return raw
    if isinstance(raw, BaseMessage):
        return _validate_text_output(_message_text(raw), schema)
    if isinstance(raw, BaseModel):
        return schema.model_validate(raw.model_dump())
    if isinstance(raw, dict):
        return schema.model_validate(raw)
    if isinstance(raw, str):
        return _validate_text_output(raw, schema)
    return schema.model_validate(raw)


def _validate_text_output(text: str, schema: type[SchemaT]) -> SchemaT:
    candidate_texts = _candidate_json_texts(text)
    last_error: Exception | None = None
    for candidate in candidate_texts:
        try:
            return schema.model_validate_json(candidate)
        except ValidationError as exc:
            last_error = exc
            try:
                return schema.model_validate(json.loads(candidate))
            except Exception as inner_exc:
                last_error = inner_exc

    if last_error is not None:
        raise last_error
    raise ValidationError.from_exception_data(schema.__name__, [])


def _candidate_json_texts(text: str) -> list[str]:
    stripped = text.strip()
    candidates: list[str] = []
    if stripped:
        candidates.append(stripped)

    if stripped.startswith("```"):
        fence_lines = stripped.splitlines()
        if len(fence_lines) >= 3 and fence_lines[-1].strip() == "```":
            inner = "\n".join(fence_lines[1:-1]).strip()
            if inner:
                if inner.lower().startswith("json\n"):
                    inner = inner[5:]
                candidates.append(inner.strip())

    extracted = _extract_first_json_object(stripped)
    if extracted:
        candidates.append(extracted)

    deduped: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate not in seen:
            seen.add(candidate)
            deduped.append(candidate)
    return deduped


def _message_text(message: BaseMessage) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    return str(content)


def _extract_first_json_object(text: str) -> str | None:
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]

    return None