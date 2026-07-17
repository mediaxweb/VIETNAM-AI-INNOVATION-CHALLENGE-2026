from app.api.schemas.knowledge_base import ConversationHistoryMessage
from app.api.v1 import question_and_answer


def test_detect_question_language_supports_english_vietnamese_and_japanese():
    assert question_and_answer._detect_question_language("What is the deadline?") == "en"
    assert question_and_answer._detect_question_language("Thời hạn là gì?") == "vi"
    assert question_and_answer._detect_question_language("締め切りはいつですか？") == "ja"


def test_format_conversation_history_joins_non_empty_messages():
    history = [
        ConversationHistoryMessage(role="user", content="First question"),
        ConversationHistoryMessage(role="assistant", content="First answer"),
    ]

    assert question_and_answer._format_conversation_history(history) == "User: First question\nAssistant: First answer"


def test_parse_model_response_extracts_answer_sources_and_language():
    parsed = question_and_answer._parse_model_response(
        '{"answer":"Use section 2","has_answer":true,"source":["S1","S2"],"language":"en"}'
    )

    assert parsed.answer == "Use section 2"
    assert parsed.has_answer is True
    assert parsed.source_ids == ["S1", "S2"]
    assert parsed.language == "en"

