from qdrant_client import QdrantClient
client = QdrantClient(":memory:")
print([m for m in dir(client) if "search" in m or "query" in m])
