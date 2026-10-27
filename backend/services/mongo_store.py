import os


class MongoDocumentStore:
    """Stores the same JSON document in MongoDB when MONGODB_URI is configured."""

    def __init__(self) -> None:
        self.collection = None
        uri = os.getenv("MONGODB_URI")
        if not uri:
            return
        try:
            from pymongo import MongoClient
            database = MongoClient(uri, serverSelectionTimeoutMS=1500)[os.getenv("MONGODB_DATABASE", "rag_explorer")]
            self.collection = database[os.getenv("MONGODB_COLLECTION", "documents")]
        except Exception:
            self.collection = None

    def save(self, document: dict) -> str:
        if self.collection is not None:
            try:
                self.collection.replace_one({"id": document["id"]}, document, upsert=True)
                return "mongodb"
            except Exception:
                self.collection = None
        return "memory"

    def delete(self, document_id: str) -> None:
        if self.collection is not None:
            try:
                self.collection.delete_one({"id": document_id})
            except Exception:
                self.collection = None
