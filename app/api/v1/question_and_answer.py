import asyncio
import json
import os
import re
from dataclasses import dataclass, field
from functools import partial

from fastapi import APIRouter, Depends, HTTPException
from openai import OpenAI

from app.api.schemas.auth import UserResponse
from app.api.schemas.knowledge_base import QuestionRequest, RetrieveChunksResponse
from app.api.v1.knowledge_base import get_knowledge_base_service
from app.core.config import configs
from app.core.dependencies import get_current_user, get_openclaw_or_current_user_id
from app.services.knowledge_base_service import (
    KnowledgeBaseService,
)
from logs.logging_config import logger

router = APIRouter()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

NO_ANSWER_MESSAGES = {
    "en": (
        "Based on the provided reference materials, I couldn't find enough "
        "information to answer this question."
    ),
    "ja": "提供された参考資料に基づくと、この質問に答えるための十分な情報が見つかりませんでした。",
    "vi": "Tôi không có thông tin để trả lời câu hỏi này",
}
LANGUAGE_LABELS = {
    "en": "English",
    "ja": "Japanese",
    "vi": "Vietnamese",
}
LANGUAGE_HINT_PATTERNS = (
    ("ja", re.compile(r"[\u3040-\u30ff\u31f0-\u31ff]")),
)
VIETNAMESE_CHAR_PATTERN = re.compile(
    r"[ăâđêôơưáàảãạắằẳẵặấầẩẫậéèẻẽẹếềểễệ"
    r"íìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ]"
)
LATIN_LANGUAGE_HINTS = {
    "vi": (
        "thủ",
        "đô",
        "của",
        "là",
        "gì",
        "khong",
        "không",
        "tai lieu",
        "tài liệu",
        "cau hoi",
        "câu hỏi",
        "giup",
        "giúp",
    ),
    "en": ("what", "who", "when", "where", "why", "how", "please", "question"),
}
ANSWER_SCRIPT_PATTERNS = {
    "ja": re.compile(r"[\u3040-\u30ff\u31f0-\u31ff]"),
}


@dataclass
class ParsedModelResponse:
    answer: str
    has_answer: bool | None = None
    source_ids: list[str] = field(default_factory=list)
    language: str | None = None


def _format_conversation_history(conversation_history) -> str:
    lines = []
    for message in conversation_history:
        role = message.role.strip().capitalize()
        content = message.content.strip()
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _detect_question_language(question: str) -> str:
    for language, pattern in LANGUAGE_HINT_PATTERNS:
        if pattern.search(question):
            return language

    if VIETNAMESE_CHAR_PATTERN.search(question.casefold()):
        return "vi"

    lowered_question = f" {question.casefold()} "
    for language, hints in LATIN_LANGUAGE_HINTS.items():
        if any(f" {hint} " in lowered_question for hint in hints):
            return language

    return "en"


def _localized_no_answer_message(question: str) -> str:
    return NO_ANSWER_MESSAGES.get(
        _detect_question_language(question),
        NO_ANSWER_MESSAGES["en"],
    )


def _language_label(language_code: str) -> str:
    return LANGUAGE_LABELS.get(language_code, "English")


def _build_system_prompt(question_language: str, question_language_label: str) -> str:
    return f"""
    Bạn là một trợ lý hỏi đáp AI thông minh và tận tâm. Nhiệm vụ của bạn là trả lời các câu hỏi của người dùng MỘT CÁCH CHÍNH XÁC VÀ ĐẦY ĐỦ nhất có thể, DỰA HOÀN TOÀN vào thông tin tham khảo được cung cấp.

    Ngôn ngữ của câu hỏi hiện tại đã được xác định là {question_language_label} ({question_language}).

    ## Hướng dẫn Trả lời:
    1. **ĐÚNG TRỌNG TÂM:** Chỉ trả lời dựa trên *Context* được trích xuất. Tránh suy diễn, thêm thông tin bên ngoài, hoặc trả lời mơ hồ.
    2. **ĐÚNG NGÔN NGỮ:** Trường `answer` BẮT BUỘC phải dùng đúng ngôn ngữ của câu hỏi hiện tại là {question_language_label} ({question_language}), kể cả khi tài liệu tham khảo đang ở ngôn ngữ khác.
    3. **THÂN THIỆN & RÕ RÀNG:** Sử dụng giọng điệu chuyên nghiệp, thân thiện, và dễ hiểu.
    4. **XỬ LÝ THIẾU THÔNG TIN:** Nếu Context KHÔNG CÓ thông tin cần thiết để trả lời câu hỏi, hãy dùng answer theo CÙNG NGÔN NGỮ với câu hỏi hiện tại và giữ cùng ý nghĩa với câu: **"{NO_ANSWER_MESSAGES["vi"]}"**
    5. **BẮT BUỘC JSON:** Chỉ trả về JSON hợp lệ, không thêm markdown/code fence, theo đúng schema:
        {{
          "answer": "string",
          "has_answer": true,
          "source": ["S1", "S2"],
          "language": "{question_language}"
        }}
    6. Khi có đủ thông tin để trả lời, đặt `has_answer: true`, `source` chỉ chứa các mã nguồn có trong context (S1..Sn), và `language: "{question_language}"`.
    7. Khi KHÔNG đủ thông tin để trả lời, đặt `has_answer: false`, `source: []`, `language: "{question_language}"`, và `answer` là câu từ chối theo cùng ngôn ngữ với câu hỏi.
    """


def _build_question_prompt(
    question: str,
    context: str,
    history_text: str,
    question_language: str,
    question_language_label: str,
) -> str:
    history_section = ""
    if history_text:
        history_section = f"""
    Lịch sử hội thoại trước đó (chỉ dùng để hiểu ngữ cảnh của câu hỏi hiện tại, không dùng làm nguồn thông tin thay cho tài liệu):
    {history_text}

    """

    return f"""
    {history_section}Ngôn ngữ câu hỏi hiện tại: {question_language_label} ({question_language}).
    Hãy trả lời bằng chính ngôn ngữ này trong trường `answer`.

    Dưới đây là thông tin tài liệu tham khảo:
    {context}

    Dựa trên thông tin trên, trả lời câu hỏi hiện tại: {question}
    """


def _build_language_repair_messages(question: str, question_language: str, raw_content: str) -> list[dict[str, str]]:
    localized_no_answer = _localized_no_answer_message(question)
    system_prompt = f"""
    Bạn là trợ lý sửa JSON đầu ra.
    Hãy giữ nguyên ý nghĩa, `has_answer`, và `source`, nhưng đảm bảo `answer` dùng đúng ngôn ngữ `{question_language}`.
    Chỉ trả về JSON hợp lệ, không thêm markdown/code fence, theo schema:
    {{
      "answer": "string",
      "has_answer": true,
      "source": ["S1", "S2"],
      "language": "{question_language}"
    }}
    Nếu `has_answer` là false thì `answer` phải là: "{localized_no_answer}"
    """
    user_prompt = f"""
    Câu hỏi gốc: {question}
    JSON hiện tại:
    {raw_content}
    """
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _create_completion(messages: list[dict[str, str]]) -> str:
    response = client.chat.completions.create(
        model=configs.openai_qna_model,
        messages=messages,
    )
    return (response.choices[0].message.content or "").strip()


def _parse_model_response(raw_content: str) -> ParsedModelResponse:
    parsed_response = ParsedModelResponse(answer=raw_content)
    try:
        parsed_json = json.loads(raw_content)
    except json.JSONDecodeError:
        logger.warning("LLM did not return valid JSON. Falling back to raw content.")
        return parsed_response

    parsed_response.answer = str(parsed_json.get("answer", "")).strip() or parsed_response.answer
    parsed_response.has_answer = _coerce_has_answer(parsed_json.get("has_answer"))

    src = parsed_json.get("source", [])
    if isinstance(src, list):
        parsed_response.source_ids = [str(item).strip() for item in src if str(item).strip()]

    language = str(parsed_json.get("language", "")).strip().casefold()
    if language:
        parsed_response.language = language

    return parsed_response


def _should_repair_language(question_language: str, parsed_response: ParsedModelResponse) -> bool:
    if parsed_response.has_answer is False or not parsed_response.answer:
        return False
    if parsed_response.language and parsed_response.language != question_language:
        return True

    script_pattern = ANSWER_SCRIPT_PATTERNS.get(question_language)
    if script_pattern is None:
        return False
    return script_pattern.search(parsed_response.answer) is None


def _coerce_has_answer(value) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized_value = value.strip().casefold()
        if normalized_value in {"true", "1", "yes"}:
            return True
        if normalized_value in {"false", "0", "no"}:
            return False
    return None


@router.post("/question_and_answer")
async def question_and_answer(
    request: QuestionRequest,
    service: KnowledgeBaseService = Depends(get_knowledge_base_service),
    current_user: UserResponse = Depends(get_current_user),
):
    try:
        qa_result = await asyncio.to_thread(
            partial(
                service.question_and_answer,
                request.question,
                request.conversation_history,
                user_id=current_user.id,
            ),
        )
    except ValueError as exc:
        logger.warning("Knowledge base question answering failed with a validation error: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    metadata = qa_result["metadata_list"]

    source_metadata = []
    source_index = {}
    context_blocks = []
    for idx, item in enumerate(metadata, start=1):
        source_id = f"S{idx}"
        source_item = {
            "source_id": source_id,
            "page": item.get("page_label", "-"),
            "file_name": item.get("file_name", "-"),
            "window": item.get("window", "-"),
        }
        source_metadata.append(source_item)
        source_index[source_id] = source_item
        context_blocks.append(
            f"[{source_id}] Page {source_item['page']}, File {source_item['file_name']}\n{source_item['window']}"
        )

    context = "\n\n".join(context_blocks)

    question_language = _detect_question_language(request.question)
    question_language_label = _language_label(question_language)
    system_prompt = _build_system_prompt(question_language, question_language_label)
    history_text = _format_conversation_history(request.conversation_history)
    prompt = _build_question_prompt(
        request.question,
        context,
        history_text,
        question_language,
        question_language_label,
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    raw_content = _create_completion(messages)
    parsed_response = _parse_model_response(raw_content)

    if _should_repair_language(question_language, parsed_response):
        repaired_raw_content = _create_completion(
            _build_language_repair_messages(
                request.question,
                question_language,
                raw_content,
            )
        )
        repaired_response = _parse_model_response(repaired_raw_content)
        if repaired_response.answer:
            raw_content = repaired_raw_content
            parsed_response = repaired_response

    parsed_answer = parsed_response.answer
    parsed_has_answer = parsed_response.has_answer
    parsed_source_ids = parsed_response.source_ids
    if parsed_has_answer is False:
        parsed_answer = _localized_no_answer_message(request.question)
        parsed_source_ids = []

    resolved_sources = []
    if parsed_has_answer is False:
        resolved_sources = []
    elif parsed_source_ids:
        for source_id in parsed_source_ids:
            if source_id in source_index:
                source_item = source_index[source_id]
                resolved_sources.append(
                    {
                        "source_id": source_id,
                        "page": source_item["page"],
                        "file_name": source_item["file_name"],
                    }
                )
    else:
        # Fallback: expose all retrieved sources when model does not provide source ids
        resolved_sources = [
            {"source_id": item["source_id"], "page": item["page"], "file_name": item["file_name"]}
            for item in source_metadata
        ]

    return {
        "answer": parsed_answer,
        "source": resolved_sources,
        "source_ids": parsed_source_ids,
        "metadata": source_metadata,
        "raw_model_output": raw_content,
    }


@router.post("/retrieve-chunks", response_model=RetrieveChunksResponse)
async def retrieve_chunks(
    request: QuestionRequest,
    service: KnowledgeBaseService = Depends(get_knowledge_base_service),
    user_id: str = Depends(get_openclaw_or_current_user_id),
) -> RetrieveChunksResponse:
    try:
        result = await asyncio.to_thread(
            partial(
                service.retrieve_chunks,
                request.question,
                request.conversation_history,
                user_id=user_id,
            ),
        )
    except ValueError as exc:
        logger.warning("Knowledge base chunk retrieval failed with a validation error: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return RetrieveChunksResponse.model_validate(result)
