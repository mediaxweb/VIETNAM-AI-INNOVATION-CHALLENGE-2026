import os
from urllib.parse import urlparse

import motor.motor_asyncio
from dotenv import load_dotenv
from pymongo import ASCENDING

load_dotenv()


USER_INDEXED_FILES_COLLECTION_NAME = "user_indexed_files"

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
    print(">>> INIT DB DONE")
