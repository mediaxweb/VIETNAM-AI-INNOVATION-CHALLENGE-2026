import os
from urllib.parse import urlparse

import motor.motor_asyncio
from dotenv import load_dotenv
from pymongo import ASCENDING

load_dotenv()


USER_INDEXED_FILES_COLLECTION_NAME = "user_indexed_files"
LOAN_AGENT_COLLECTION_NAMES = (
    "loan_customers",
    "loan_profiles",
    "loan_legal_docs",
    "loan_financial_reports",
    "loan_collaterals",
    "loan_agent_checks",
    "loan_compliance_results",
    "loan_checklists",
    "loan_limit_calculations",
    "loan_tasks",
    "loan_reports",
)

KB_DOCUMENT_INDEX_STATUS_INDEXED = "indexed"
KB_DOCUMENT_INDEX_STATUS_DELETED = "deleted"
KB_DOCUMENT_INDEX_STATUS_FAILED = "failed"

KB_DOCUMENT_INDEX_STATUSES = frozenset(
    {
        KB_DOCUMENT_INDEX_STATUS_INDEXED,
        KB_DOCUMENT_INDEX_STATUS_DELETED,
        KB_DOCUMENT_INDEX_STATUS_FAILED,
    }
)

USER_INDEXED_FILES_UNIQUE_INDEX_NAME = "user_indexed_files_user_file_unique"
USER_INDEXED_FILES_USER_UPDATED_INDEX_NAME = "user_indexed_files_user_updated_at"
DEFAULT_DATABASE_NAME = "rag_brain"


class Database:
    client: motor.motor_asyncio.AsyncIOMotorClient = None

    @staticmethod
    def _mongo_uri() -> str:
        mongo_uri = (os.getenv("MONGO_URI") or "").strip()
        if not mongo_uri:
            raise ValueError("MONGO_URI environment variable is required.")
        return mongo_uri

    @staticmethod
    def _fallback_database_name() -> str:
        configured_name = (os.getenv("MONGO_DB_NAME") or "").strip()
        return configured_name or DEFAULT_DATABASE_NAME

    @classmethod
    def get_client(cls) -> motor.motor_asyncio.AsyncIOMotorClient:
        if cls.client is None:
            cls.client = motor.motor_asyncio.AsyncIOMotorClient(cls._mongo_uri())
        return cls.client

    @classmethod
    def get_database_name(cls) -> str:
        parsed_uri = urlparse(cls._mongo_uri())
        database_name = parsed_uri.path.lstrip("/").strip()
        return database_name or cls._fallback_database_name()

    @classmethod
    def get_database(cls):
        """Get database instance."""
        client = cls.get_client()
        return client[cls.get_database_name()]

    @classmethod
    def get_users_collection(cls):
        """Get the users collection used by the auth domain."""

        return cls.get_database()["users"]

    @classmethod
    def get_user_indexed_files_collection(cls):
        """Get the registry collection for user-scoped indexed files."""

        return cls.get_database()[USER_INDEXED_FILES_COLLECTION_NAME]

    @classmethod
    def get_loan_collection(cls, collection_name: str):
        """Get one collection used by the loan-agent domain."""

        if collection_name not in LOAN_AGENT_COLLECTION_NAMES:
            raise ValueError(f"Unknown loan-agent collection '{collection_name}'.")
        return cls.get_database()[collection_name]
    
    @classmethod
    async def close_connection(cls):
        """Close database connection."""
        if cls.client:
            cls.client.close()
            cls.client = None


async def ensure_user_indexed_files_indexes():
    """Create indexes required by the user-scoped indexed file registry."""

    collection = Database.get_user_indexed_files_collection()
    await collection.create_index(
        [("user_id", ASCENDING), ("file_key", ASCENDING)],
        unique=True,
        name=USER_INDEXED_FILES_UNIQUE_INDEX_NAME,
    )
    await collection.create_index(
        [("user_id", ASCENDING), ("updated_at", ASCENDING)],
        name=USER_INDEXED_FILES_USER_UPDATED_INDEX_NAME,
    )


async def ensure_loan_agent_indexes():
    """Create indexes required by user-scoped loan-agent collections."""

    for collection_name in LOAN_AGENT_COLLECTION_NAMES:
        collection = Database.get_loan_collection(collection_name)
        await collection.create_index(
            [("user_id", ASCENDING), ("created_at", ASCENDING)],
            name=f"{collection_name}_user_created_at",
        )

    loan_profiles = Database.get_loan_collection("loan_profiles")
    await loan_profiles.create_index(
        [("user_id", ASCENDING), ("customer_id", ASCENDING)],
        name="loan_profiles_user_customer",
    )

    for collection_name in (
        "loan_legal_docs",
        "loan_financial_reports",
        "loan_collaterals",
        "loan_agent_checks",
        "loan_compliance_results",
        "loan_checklists",
        "loan_limit_calculations",
        "loan_tasks",
        "loan_reports",
    ):
        collection = Database.get_loan_collection(collection_name)
        await collection.create_index(
            [("user_id", ASCENDING), ("loan_profile_id", ASCENDING)],
            name=f"{collection_name}_user_loan_profile",
        )


async def init_db():
    """Initialize and validate the MongoDB connection."""
    print(">>> INIT DB START")
    client = Database.get_client()
    await client.admin.command("ping")
    await Database.get_users_collection().create_index(
        [("email", ASCENDING)],
        unique=True,
        name="users_email_unique",
    )
    await ensure_user_indexed_files_indexes()
    await ensure_loan_agent_indexes()
    print(">>> INIT DB DONE")
