from fastapi import APIRouter

from app.api.v1 import auth
from app.api.v1 import knowledge_base
from app.api.v1 import loan_agent
from app.api.v1 import question_and_answer

api_router = APIRouter()

api_router.include_router(
    auth.router, prefix="/auth", tags=["Auth"]
)
api_router.include_router(
    knowledge_base.router, prefix="/knowledge-base", tags=["Knowledge Base"]
)
api_router.include_router(
    loan_agent.router, prefix="/loan"
)
api_router.include_router(
    question_and_answer.router, prefix="/qna", tags=["Question and Answer"]
)
