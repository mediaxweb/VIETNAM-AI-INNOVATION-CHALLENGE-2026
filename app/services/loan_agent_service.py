from __future__ import annotations

import re
import unicodedata
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bson import ObjectId
from bson.errors import InvalidId
from fastapi import UploadFile

from app.api.schemas.loan_agent import (
    CaseStatusResponse,
    CaseStatusUpdateRequest,
    ChecklistCreateRequest,
    ChecklistResponse,
    CheckResultResponse,
    CollateralResponse,
    ComplianceResultRequest,
    ComplianceResultResponse,
    CustomerCreateRequest,
    CustomerResponse,
    CustomerUpdateRequest,
    FinancialReportResponse,
    LoanLimitCalculationRequest,
    LoanLimitCalculationResponse,
    LoanProfileCreateRequest,
    LoanProfileResponse,
    ReportCreateRequest,
    ReportResponse,
    ReportsListResponse,
    TaskCreateRequest,
    TaskResponse,
    UploadedDocumentResponse,
)
from app.core.config import configs
from app.core.database import Database


UPLOAD_CHUNK_SIZE_BYTES = 1024 * 1024


class LoanAgentError(Exception):
    """Base error for loan-agent service failures."""


class LoanAgentNotFoundError(LoanAgentError):
    """Raised when a user-scoped loan-agent record cannot be found."""


class LoanAgentValidationError(LoanAgentError):
    """Raised when a loan-agent request is invalid."""


class LoanAgentService:
    """Service layer for the bank loan-agent MVP workflow."""

    DEFAULT_CHECKLIST_ITEMS = (
        {"title": "Customer profile is created", "owner_agent": "credit"},
        {"title": "Legal documents are uploaded", "owner_agent": "credit"},
        {"title": "Financials are checked", "owner_agent": "compliance"},
        {"title": "Collateral is checked", "owner_agent": "compliance"},
        {"title": "Loan limit is calculated", "owner_agent": "operations"},
        {"title": "Case report is created", "owner_agent": "operations"},
    )

    def __init__(self, database: Database = Database):
        self.database = database

    async def create_customer(self, *, user_id: str, payload: CustomerCreateRequest) -> CustomerResponse:
        now = self._now()
        document = {
            **payload.model_dump(mode="json", exclude_none=True),
            "user_id": self._normalize_user_id(user_id),
            "created_at": now,
            "updated_at": now,
        }
        result = await self._collection("loan_customers").insert_one(document)
        document["_id"] = result.inserted_id
        return CustomerResponse.model_validate(self._serialize_record(document))

    async def get_customer(self, *, user_id: str, customer_id: str) -> CustomerResponse:
        document = await self._require_customer(user_id=user_id, customer_id=customer_id)
        return CustomerResponse.model_validate(self._serialize_record(document))

    async def update_customer(
        self,
        *,
        user_id: str,
        customer_id: str,
        payload: CustomerUpdateRequest,
    ) -> CustomerResponse:
        updates = payload.model_dump(mode="json", exclude_unset=True)
        updates = {key: value for key, value in updates.items() if value is not None}
        if not updates:
            raise LoanAgentValidationError("At least one customer field is required.")

        now = self._now()
        updates["updated_at"] = now
        customer_object_id = self._object_id(customer_id, label="customer_id")
        result = await self._collection("loan_customers").update_one(
            {"_id": customer_object_id, "user_id": self._normalize_user_id(user_id)},
            {"$set": updates},
        )
        if result.matched_count == 0:
            raise LoanAgentNotFoundError("Customer was not found.")

        return await self.get_customer(user_id=user_id, customer_id=customer_id)

    async def create_loan_profile(
        self,
        *,
        user_id: str,
        payload: LoanProfileCreateRequest,
    ) -> LoanProfileResponse:
        normalized_user_id = self._normalize_user_id(user_id)
        await self._require_customer(user_id=normalized_user_id, customer_id=payload.customer_id)

        now = self._now()
        document = {
            **payload.model_dump(mode="json", exclude_none=True),
            "user_id": normalized_user_id,
            "status": "draft",
            "created_at": now,
            "updated_at": now,
        }
        result = await self._collection("loan_profiles").insert_one(document)
        document["_id"] = result.inserted_id
        return LoanProfileResponse.model_validate(self._serialize_record(document))

    async def get_loan_profile(self, *, user_id: str, loan_profile_id: str) -> LoanProfileResponse:
        document = await self._require_loan_profile(
            user_id=user_id,
            loan_profile_id=loan_profile_id,
        )
        return LoanProfileResponse.model_validate(self._serialize_record(document))

    async def upload_legal_doc(
        self,
        *,
        user_id: str,
        loan_profile_id: str,
        file: UploadFile,
        doc_type: str | None,
        description: str | None,
    ) -> UploadedDocumentResponse:
        await self._require_loan_profile(user_id=user_id, loan_profile_id=loan_profile_id)
        upload = await self._save_uploaded_file(
            file,
            user_id=user_id,
            loan_profile_id=loan_profile_id,
            category="legal_doc",
        )
        document = await self._insert_uploaded_document(
            collection_name="loan_legal_docs",
            user_id=user_id,
            loan_profile_id=loan_profile_id,
            category="legal_doc",
            upload=upload,
            extra={
                "doc_type": self._normalize_optional_text(doc_type),
                "description": self._normalize_optional_text(description),
                "status": "uploaded",
            },
        )
        return UploadedDocumentResponse.model_validate(self._serialize_record(document))

    async def check_legal_docs(
        self,
        *,
        user_id: str,
        loan_profile_id: str,
        required_doc_types: list[str],
        notes: str | None = None,
    ):
        await self._require_loan_profile(user_id=user_id, loan_profile_id=loan_profile_id)
        required = [item.strip().lower() for item in required_doc_types if item.strip()]
        documents = await self._find_by_profile(
            "loan_legal_docs",
            user_id=user_id,
            loan_profile_id=loan_profile_id,
        )
        uploaded_types = {
            str(document.get("doc_type") or "").strip().lower()
            for document in documents
            if document.get("doc_type")
        }
        missing = [doc_type for doc_type in required if doc_type not in uploaded_types]
        issues = [
            {
                "code": "missing_legal_doc",
                "message": f"Missing required legal document type: {doc_type}.",
                "severity": "critical",
            }
            for doc_type in missing
        ]
        if not documents:
            issues.append(
                {
                    "code": "no_legal_docs",
                    "message": "No legal documents have been uploaded.",
                    "severity": "critical",
                }
            )

        status = "passed" if not issues else "failed"
        score = 100.0 if not required else round(((len(required) - len(missing)) / len(required)) * 100, 2)
        if not documents:
            score = 0.0
        recommendation = (
            "Legal documents are sufficient for the MVP review."
            if status == "passed"
            else "Request the missing legal documents before compliance review."
        )
        return await self._record_check(
            user_id=user_id,
            loan_profile_id=loan_profile_id,
            check_type="legal_docs",
            status=status,
            score=score,
            issues=issues,
            recommendation=recommendation,
            notes=notes,
        )

    async def upload_financial_report(
        self,
        *,
        user_id: str,
        loan_profile_id: str,
        file: UploadFile,
        report_period: str | None,
        declared_revenue: float | None,
        declared_expense: float | None,
        declared_monthly_income: float | None,
    ) -> FinancialReportResponse:
        await self._require_loan_profile(user_id=user_id, loan_profile_id=loan_profile_id)
        upload = await self._save_uploaded_file(
            file,
            user_id=user_id,
            loan_profile_id=loan_profile_id,
            category="financial_report",
        )
        document = await self._insert_uploaded_document(
            collection_name="loan_financial_reports",
            user_id=user_id,
            loan_profile_id=loan_profile_id,
            category="financial_report",
            upload=upload,
            extra={
                "report_period": self._normalize_optional_text(report_period),
                "declared_revenue": declared_revenue,
                "declared_expense": declared_expense,
                "declared_monthly_income": declared_monthly_income,
                "status": "uploaded",
            },
        )
        return FinancialReportResponse.model_validate(self._serialize_record(document))

    async def upload_collateral(
        self,
        *,
        user_id: str,
        loan_profile_id: str,
        file: UploadFile,
        collateral_type: str | None,
        estimated_value: float | None,
        currency: str,
        description: str | None,
    ) -> CollateralResponse:
        await self._require_loan_profile(user_id=user_id, loan_profile_id=loan_profile_id)
        upload = await self._save_uploaded_file(
            file,
            user_id=user_id,
            loan_profile_id=loan_profile_id,
            category="collateral",
        )
        document = await self._insert_uploaded_document(
            collection_name="loan_collaterals",
            user_id=user_id,
            loan_profile_id=loan_profile_id,
            category="collateral",
            upload=upload,
            extra={
                "collateral_type": self._normalize_optional_text(collateral_type),
                "estimated_value": estimated_value,
                "currency": currency or "VND",
                "description": self._normalize_optional_text(description),
                "status": "uploaded",
            },
        )
        return CollateralResponse.model_validate(self._serialize_record(document))

    async def check_financials(self, *, user_id: str, loan_profile_id: str, notes: str | None = None):
        await self._require_loan_profile(user_id=user_id, loan_profile_id=loan_profile_id)
        reports = await self._find_by_profile(
            "loan_financial_reports",
            user_id=user_id,
            loan_profile_id=loan_profile_id,
        )
        issues: list[dict[str, Any]] = []
        if not reports:
            issues.append(
                {
                    "code": "no_financial_report",
                    "message": "No financial report has been uploaded.",
                    "severity": "critical",
                }
            )

        declared_income = self._latest_number(reports, "declared_monthly_income")
        if reports and declared_income is None:
            issues.append(
                {
                    "code": "missing_income",
                    "message": "Financial reports do not include declared monthly income.",
                    "severity": "warning",
                }
            )

        status = "passed" if not issues else "warning" if reports else "failed"
        score = 100.0 if status == "passed" else 70.0 if status == "warning" else 0.0
        recommendation = (
            "Financial information is ready for credit-rule checks."
            if status == "passed"
            else "Review or supplement financial information before final approval."
        )
        return await self._record_check(
            user_id=user_id,
            loan_profile_id=loan_profile_id,
            check_type="financials",
            status=status,
            score=score,
            issues=issues,
            recommendation=recommendation,
            notes=notes,
        )

    async def check_collateral(self, *, user_id: str, loan_profile_id: str, notes: str | None = None):
        profile = await self._require_loan_profile(user_id=user_id, loan_profile_id=loan_profile_id)
        collaterals = await self._find_by_profile(
            "loan_collaterals",
            user_id=user_id,
            loan_profile_id=loan_profile_id,
        )
        total_value = sum(float(item.get("estimated_value") or 0) for item in collaterals)
        requested_amount = float(profile.get("loan_amount") or 0)
        issues: list[dict[str, Any]] = []
        if not collaterals:
            issues.append(
                {
                    "code": "no_collateral",
                    "message": "No collateral has been uploaded.",
                    "severity": "critical",
                }
            )
        elif requested_amount and total_value < requested_amount:
            issues.append(
                {
                    "code": "low_collateral_value",
                    "message": "Collateral value is lower than requested loan amount.",
                    "severity": "warning",
                }
            )

        status = "passed" if not issues else "warning" if collaterals else "failed"
        score = 0.0 if requested_amount <= 0 else min(round((total_value / requested_amount) * 100, 2), 100.0)
        recommendation = (
            "Collateral is sufficient for the MVP review."
            if status == "passed"
            else "Review collateral value or request additional collateral."
        )
        return await self._record_check(
            user_id=user_id,
            loan_profile_id=loan_profile_id,
            check_type="collateral",
            status=status,
            score=score,
            issues=issues,
            recommendation=recommendation,
            notes=notes,
        )

    async def check_credit_rule(self, *, user_id: str, loan_profile_id: str, notes: str | None = None):
        profile = await self._require_loan_profile(user_id=user_id, loan_profile_id=loan_profile_id)
        legal_docs = await self._find_by_profile("loan_legal_docs", user_id=user_id, loan_profile_id=loan_profile_id)
        reports = await self._find_by_profile(
            "loan_financial_reports",
            user_id=user_id,
            loan_profile_id=loan_profile_id,
        )
        collaterals = await self._find_by_profile("loan_collaterals", user_id=user_id, loan_profile_id=loan_profile_id)

        issues: list[dict[str, Any]] = []
        if not legal_docs:
            issues.append({"code": "legal_docs_required", "message": "Legal documents are required.", "severity": "critical"})
        if not reports:
            issues.append({"code": "financials_required", "message": "Financial reports are required.", "severity": "critical"})
        if not collaterals:
            issues.append({"code": "collateral_required", "message": "Collateral is required.", "severity": "warning"})
        if int(profile.get("term_months") or 0) > 360:
            issues.append({"code": "term_too_long", "message": "Loan term exceeds 360 months.", "severity": "critical"})

        critical_count = sum(1 for issue in issues if issue["severity"] == "critical")
        status = "passed" if not issues else "failed" if critical_count else "warning"
        score = max(0.0, 100.0 - critical_count * 30.0 - (len(issues) - critical_count) * 15.0)
        recommendation = (
            "Credit-rule checklist passed."
            if status == "passed"
            else "Resolve credit-rule issues before approval."
        )
        return await self._record_check(
            user_id=user_id,
            loan_profile_id=loan_profile_id,
            check_type="credit_rule",
            status=status,
            score=score,
            issues=issues,
            recommendation=recommendation,
            notes=notes,
        )

    async def save_compliance_result(
        self,
        *,
        user_id: str,
        loan_profile_id: str,
        payload: ComplianceResultRequest,
    ) -> ComplianceResultResponse:
        await self._require_loan_profile(user_id=user_id, loan_profile_id=loan_profile_id)
        now = self._now()
        document = {
            **payload.model_dump(mode="json", exclude_none=True),
            "user_id": self._normalize_user_id(user_id),
            "loan_profile_id": loan_profile_id,
            "created_at": now,
        }
        result = await self._collection("loan_compliance_results").insert_one(document)
        document["_id"] = result.inserted_id
        await self._collection("loan_profiles").update_one(
            {"_id": self._object_id(loan_profile_id, label="loan_profile_id")},
            {
                "$set": {
                    "status": "compliance_completed",
                    "compliance_decision": payload.decision,
                    "updated_at": now,
                }
            },
        )
        return ComplianceResultResponse.model_validate(self._serialize_record(document))

    async def update_case_status(
        self,
        *,
        user_id: str,
        loan_profile_id: str,
        payload: CaseStatusUpdateRequest,
    ) -> CaseStatusResponse:
        await self._require_loan_profile(user_id=user_id, loan_profile_id=loan_profile_id)
        now = self._now()
        result = await self._collection("loan_profiles").update_one(
            {
                "_id": self._object_id(loan_profile_id, label="loan_profile_id"),
                "user_id": self._normalize_user_id(user_id),
            },
            {
                "$set": {
                    "status": payload.status,
                    "status_notes": payload.notes,
                    "updated_at": now,
                }
            },
        )
        if result.matched_count == 0:
            raise LoanAgentNotFoundError("Loan profile was not found.")
        return CaseStatusResponse(
            loan_profile_id=loan_profile_id,
            status=payload.status,
            notes=payload.notes,
            updated_at=now,
        )

    async def create_checklist(
        self,
        *,
        user_id: str,
        loan_profile_id: str,
        payload: ChecklistCreateRequest,
    ) -> ChecklistResponse:
        await self._require_loan_profile(user_id=user_id, loan_profile_id=loan_profile_id)
        now = self._now()
        source_items = payload.items or [
            {**item, "status": "pending"}
            for item in self.DEFAULT_CHECKLIST_ITEMS
        ]
        items = [
            {
                "id": uuid.uuid4().hex,
                **(item.model_dump(mode="json") if hasattr(item, "model_dump") else item),
            }
            for item in source_items
        ]
        document = {
            "user_id": self._normalize_user_id(user_id),
            "loan_profile_id": loan_profile_id,
            "items": items,
            "status": "open",
            "notes": payload.notes,
            "created_at": now,
            "updated_at": now,
        }
        result = await self._collection("loan_checklists").insert_one(document)
        document["_id"] = result.inserted_id
        return ChecklistResponse.model_validate(self._serialize_record(document))

    async def calculate_loan_limit(
        self,
        *,
        user_id: str,
        loan_profile_id: str,
        payload: LoanLimitCalculationRequest,
    ) -> LoanLimitCalculationResponse:
        profile = await self._require_loan_profile(user_id=user_id, loan_profile_id=loan_profile_id)
        reports = await self._find_by_profile(
            "loan_financial_reports",
            user_id=user_id,
            loan_profile_id=loan_profile_id,
        )
        collaterals = await self._find_by_profile("loan_collaterals", user_id=user_id, loan_profile_id=loan_profile_id)

        monthly_income = payload.monthly_income
        if monthly_income is None:
            monthly_income = self._latest_number(reports, "declared_monthly_income") or 0.0
        monthly_debt = payload.monthly_debt or 0.0
        collateral_value = payload.collateral_value
        if collateral_value is None:
            collateral_value = sum(float(item.get("estimated_value") or 0) for item in collaterals)

        term_months = int(profile.get("term_months") or 1)
        income_capacity = max((monthly_income * payload.debt_to_income_ratio - monthly_debt) * term_months, 0.0)
        collateral_capacity = collateral_value * payload.collateral_ltv_ratio if collateral_value else 0.0
        capacities = [value for value in (income_capacity, collateral_capacity) if value > 0]
        calculated_limit = min(capacities) if len(capacities) > 1 else capacities[0] if capacities else 0.0
        requested_amount = float(profile.get("loan_amount") or 0.0)
        recommended_limit = min(requested_amount, calculated_limit) if calculated_limit else 0.0
        now = self._now()
        document = {
            "user_id": self._normalize_user_id(user_id),
            "loan_profile_id": loan_profile_id,
            "requested_amount": requested_amount,
            "calculated_limit": round(calculated_limit, 2),
            "recommended_limit": round(recommended_limit, 2),
            "currency": profile.get("currency") or "VND",
            "assumptions": {
                "monthly_income": monthly_income,
                "monthly_debt": monthly_debt,
                "collateral_value": collateral_value,
                "debt_to_income_ratio": payload.debt_to_income_ratio,
                "collateral_ltv_ratio": payload.collateral_ltv_ratio,
                "term_months": term_months,
            },
            "created_at": now,
        }
        result = await self._collection("loan_limit_calculations").insert_one(document)
        document["_id"] = result.inserted_id
        return LoanLimitCalculationResponse.model_validate(self._serialize_record(document))

    async def create_task(
        self,
        *,
        user_id: str,
        loan_profile_id: str,
        payload: TaskCreateRequest,
    ) -> TaskResponse:
        await self._require_loan_profile(user_id=user_id, loan_profile_id=loan_profile_id)
        now = self._now()
        document = {
            **payload.model_dump(mode="json", exclude_none=True),
            "user_id": self._normalize_user_id(user_id),
            "loan_profile_id": loan_profile_id,
            "status": "open",
            "created_at": now,
            "updated_at": now,
        }
        result = await self._collection("loan_tasks").insert_one(document)
        document["_id"] = result.inserted_id
        return TaskResponse.model_validate(self._serialize_record(document))

    async def create_report(
        self,
        *,
        user_id: str,
        loan_profile_id: str,
        payload: ReportCreateRequest,
    ) -> ReportResponse:
        profile = await self._require_loan_profile(user_id=user_id, loan_profile_id=loan_profile_id)
        customer = await self._require_customer(user_id=user_id, customer_id=str(profile["customer_id"]))
        now = self._now()
        report_body = await self._build_report_body(
            user_id=user_id,
            loan_profile_id=loan_profile_id,
            profile=profile,
            customer=customer,
        )
        title = payload.title or f"Loan case report for {customer.get('full_name')}"
        summary = payload.summary or self._build_report_summary(profile, report_body)
        document = {
            "user_id": self._normalize_user_id(user_id),
            "loan_profile_id": loan_profile_id,
            "report_type": payload.report_type,
            "title": title,
            "summary": summary,
            "report_body": report_body,
            "created_at": now,
        }
        result = await self._collection("loan_reports").insert_one(document)
        document["_id"] = result.inserted_id
        return ReportResponse.model_validate(self._serialize_record(document))

    async def list_reports(self, *, user_id: str, loan_profile_id: str) -> ReportsListResponse:
        await self._require_loan_profile(user_id=user_id, loan_profile_id=loan_profile_id)
        documents = await self._find_by_profile(
            "loan_reports",
            user_id=user_id,
            loan_profile_id=loan_profile_id,
            sort_desc=True,
        )
        reports = [
            ReportResponse.model_validate(self._serialize_record(document))
            for document in documents
        ]
        return ReportsListResponse(total_count=len(reports), reports=reports)

    async def _insert_uploaded_document(
        self,
        *,
        collection_name: str,
        user_id: str,
        loan_profile_id: str,
        category: str,
        upload: dict[str, Any],
        extra: dict[str, Any],
    ) -> dict[str, Any]:
        now = self._now()
        document = {
            **upload,
            **extra,
            "user_id": self._normalize_user_id(user_id),
            "loan_profile_id": loan_profile_id,
            "category": category,
            "created_at": now,
        }
        result = await self._collection(collection_name).insert_one(document)
        document["_id"] = result.inserted_id
        return document

    async def _save_uploaded_file(
        self,
        file: UploadFile,
        *,
        user_id: str,
        loan_profile_id: str,
        category: str,
    ) -> dict[str, Any]:
        file_name = self._uploaded_file_name(file)
        target_dir = (
            Path(configs.resolved_loan_upload_dir)
            / self._safe_path_component(user_id)
            / self._safe_path_component(loan_profile_id)
            / self._safe_path_component(category)
        )
        target_dir.mkdir(parents=True, exist_ok=True)
        stored_name = f"{uuid.uuid4().hex}_{self._safe_file_name(file_name)}"
        target_path = target_dir / stored_name
        size_bytes = 0
        try:
            with target_path.open("wb") as output:
                while True:
                    chunk = await file.read(UPLOAD_CHUNK_SIZE_BYTES)
                    if not chunk:
                        break
                    size_bytes += len(chunk)
                    output.write(chunk)
            if size_bytes == 0:
                target_path.unlink(missing_ok=True)
                raise LoanAgentValidationError("Uploaded file cannot be empty.")
        except Exception:
            target_path.unlink(missing_ok=True)
            raise
        finally:
            await file.close()

        return {
            "file_name": file_name,
            "file_path": str(target_path),
            "content_type": file.content_type,
            "size_bytes": size_bytes,
        }

    async def _record_check(
        self,
        *,
        user_id: str,
        loan_profile_id: str,
        check_type: str,
        status: str,
        score: float,
        issues: list[dict[str, Any]],
        recommendation: str,
        notes: str | None,
    ):
        now = self._now()
        document = {
            "user_id": self._normalize_user_id(user_id),
            "loan_profile_id": loan_profile_id,
            "check_type": check_type,
            "status": status,
            "score": score,
            "issues": issues,
            "recommendation": recommendation,
            "notes": notes,
            "created_at": now,
        }
        result = await self._collection("loan_agent_checks").insert_one(document)
        document["_id"] = result.inserted_id
        return CheckResultResponse.model_validate(self._serialize_record(document))

    async def _build_report_body(
        self,
        *,
        user_id: str,
        loan_profile_id: str,
        profile: dict[str, Any],
        customer: dict[str, Any],
    ) -> dict[str, Any]:
        collection_counts = {}
        for name, collection_name in {
            "legal_docs": "loan_legal_docs",
            "financial_reports": "loan_financial_reports",
            "collaterals": "loan_collaterals",
            "checks": "loan_agent_checks",
            "tasks": "loan_tasks",
        }.items():
            collection_counts[name] = await self._collection(collection_name).count_documents(
                {
                    "user_id": self._normalize_user_id(user_id),
                    "loan_profile_id": loan_profile_id,
                }
            )

        latest_compliance = await self._latest_by_profile(
            "loan_compliance_results",
            user_id=user_id,
            loan_profile_id=loan_profile_id,
        )
        latest_limit = await self._latest_by_profile(
            "loan_limit_calculations",
            user_id=user_id,
            loan_profile_id=loan_profile_id,
        )
        return {
            "customer": self._serialize_record(customer),
            "loan_profile": self._serialize_record(profile),
            "counts": collection_counts,
            "latest_compliance_result": self._serialize_record(latest_compliance) if latest_compliance else None,
            "latest_limit_calculation": self._serialize_record(latest_limit) if latest_limit else None,
        }

    @staticmethod
    def _build_report_summary(profile: dict[str, Any], report_body: dict[str, Any]) -> str:
        counts = report_body.get("counts") or {}
        return (
            f"Loan profile status is {profile.get('status')}. "
            f"Documents: {counts.get('legal_docs', 0)} legal, "
            f"{counts.get('financial_reports', 0)} financial, "
            f"{counts.get('collaterals', 0)} collateral. "
            f"Checks completed: {counts.get('checks', 0)}."
        )

    async def _find_by_profile(
        self,
        collection_name: str,
        *,
        user_id: str,
        loan_profile_id: str,
        sort_desc: bool = False,
    ) -> list[dict[str, Any]]:
        cursor = self._collection(collection_name).find(
            {
                "user_id": self._normalize_user_id(user_id),
                "loan_profile_id": loan_profile_id,
            }
        )
        cursor = cursor.sort("created_at", -1 if sort_desc else 1)
        return await cursor.to_list(length=None)

    async def _latest_by_profile(
        self,
        collection_name: str,
        *,
        user_id: str,
        loan_profile_id: str,
    ) -> dict[str, Any] | None:
        cursor = self._collection(collection_name).find(
            {
                "user_id": self._normalize_user_id(user_id),
                "loan_profile_id": loan_profile_id,
            }
        ).sort("created_at", -1)
        documents = await cursor.to_list(length=1)
        return documents[0] if documents else None

    async def _require_customer(self, *, user_id: str, customer_id: str) -> dict[str, Any]:
        document = await self._collection("loan_customers").find_one(
            {
                "_id": self._object_id(customer_id, label="customer_id"),
                "user_id": self._normalize_user_id(user_id),
            }
        )
        if not document:
            raise LoanAgentNotFoundError("Customer was not found.")
        return document

    async def _require_loan_profile(self, *, user_id: str, loan_profile_id: str) -> dict[str, Any]:
        document = await self._collection("loan_profiles").find_one(
            {
                "_id": self._object_id(loan_profile_id, label="loan_profile_id"),
                "user_id": self._normalize_user_id(user_id),
            }
        )
        if not document:
            raise LoanAgentNotFoundError("Loan profile was not found.")
        return document

    def _collection(self, name: str):
        return self.database.get_loan_collection(name)

    @staticmethod
    def _object_id(value: str, *, label: str) -> ObjectId:
        try:
            return ObjectId(str(value))
        except (InvalidId, TypeError) as exc:
            raise LoanAgentValidationError(f"{label} must be a valid MongoDB ObjectId.") from exc

    @staticmethod
    def _now() -> datetime:
        return datetime.now(timezone.utc)

    @staticmethod
    def _normalize_user_id(user_id: str) -> str:
        normalized_user_id = str(user_id or "").strip()
        if not normalized_user_id:
            raise LoanAgentValidationError("user_id is required.")
        return normalized_user_id

    @staticmethod
    def _normalize_optional_text(value: str | None) -> str | None:
        normalized = str(value or "").strip()
        return normalized or None

    @classmethod
    def _uploaded_file_name(cls, file: UploadFile) -> str:
        file_name = (file.filename or "").replace("\\", "/").rsplit("/", 1)[-1].strip()
        if not file_name:
            raise LoanAgentValidationError("Uploaded file name is required.")
        return file_name

    @classmethod
    def _safe_file_name(cls, file_name: str) -> str:
        normalized_name = unicodedata.normalize("NFKD", file_name)
        ascii_name = "".join(char for char in normalized_name if not unicodedata.combining(char))
        safe_name = re.sub(r"[^a-zA-Z0-9._-]+", "_", ascii_name).strip("._-")
        return safe_name or "upload.bin"

    @classmethod
    def _safe_path_component(cls, value: str) -> str:
        return cls._safe_file_name(value).replace(".", "_")

    @staticmethod
    def _latest_number(documents: list[dict[str, Any]], key: str) -> float | None:
        for document in reversed(documents):
            value = document.get(key)
            if value is None:
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
        return None

    @staticmethod
    def _serialize_record(document: dict[str, Any]) -> dict[str, Any]:
        serialized: dict[str, Any] = {}
        for key, value in dict(document).items():
            if key == "_id":
                serialized["id"] = str(value)
            elif key == "user_id":
                continue
            elif isinstance(value, ObjectId):
                serialized[key] = str(value)
            else:
                serialized[key] = value
        return serialized
