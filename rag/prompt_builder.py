"""
Prompt Builder
Constructs prompts from retrieved documents for LLM generation.
"""

from typing import List, Dict, Any


class PromptBuilder:
    """Build prompts for RAG pipeline."""

    def __init__(self, max_context_length: int = 4096):
        """
        Initialize prompt builder.

        Args:
            max_context_length: Maximum context length in characters
        """
        self.max_context_length = max_context_length

    def build_prompt(self,
                    query: str,
                    retrieved_docs: List[Dict[str, Any]],
                    system_prompt: str = None) -> Dict[str, Any]:
        """
        Build prompt from query and retrieved documents.

        Args:
            query: User query
            retrieved_docs: List of retrieved document chunks
            system_prompt: Optional system prompt

        Returns:
            Dictionary with prompt, metadata, and token count
        """
        # Default system prompt
        if system_prompt is None:
            system_prompt = (
                "You are a helpful software engineering assistant. "
                "Answer the user's question based on the provided documentation. "
                "Be concise and accurate. If the answer is not in the documentation, "
                "say so."
            )

        # Build context from retrieved documents
        context_parts = []
        total_chars = 0

        for i, doc in enumerate(retrieved_docs, 1):
            doc_text = doc['text']
            doc_id = doc.get('document_id', 'unknown')
            doc_type = doc.get('document_type', 'text')

            # Format document
            doc_section = f"[Document {i}] ({doc_type}: {doc_id})\n{doc_text}\n"

            # Check if adding this document exceeds context length
            if total_chars + len(doc_section) > self.max_context_length:
                break

            context_parts.append(doc_section)
            total_chars += len(doc_section)

        context = "\n".join(context_parts)

        # Build final prompt
        prompt = f"""{system_prompt}

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:"""

        # Calculate token count (approximate: 1 token ≈ 4 characters)
        token_count = len(prompt) // 4

        result = {
            'prompt': prompt,
            'system_prompt': system_prompt,
            'context': context,
            'query': query,
            'num_docs_included': len(context_parts),
            'total_chars': len(prompt),
            'estimated_tokens': token_count
        }

        return result

    def build_chat_prompt(self,
                         query: str,
                         retrieved_docs: List[Dict[str, Any]],
                         system_prompt: str = None) -> Dict[str, Any]:
        """
        Build chat-style prompt (for chat models).

        Args:
            query: User query
            retrieved_docs: List of retrieved document chunks
            system_prompt: Optional system prompt

        Returns:
            Dictionary with messages array and metadata
        """
        if system_prompt is None:
            system_prompt = (
                "You are a helpful software engineering assistant. "
                "Answer questions based on the provided documentation."
            )

        # Build context
        context_parts = []
        total_chars = 0

        for i, doc in enumerate(retrieved_docs, 1):
            doc_text = doc['text']
            doc_id = doc.get('document_id', 'unknown')

            doc_section = f"[Doc {i}: {doc_id}]\n{doc_text}\n"

            if total_chars + len(doc_section) > self.max_context_length:
                break

            context_parts.append(doc_section)
            total_chars += len(doc_section)

        context = "\n".join(context_parts)

        # Build messages array
        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": f"CONTEXT:\n{context}\n\nQUESTION:\n{query}"
            }
        ]

        # Calculate token count
        total_text = system_prompt + context + query
        token_count = len(total_text) // 4

        result = {
            'messages': messages,
            'num_docs_included': len(context_parts),
            'estimated_tokens': token_count
        }

        return result

    def truncate_to_token_limit(self,
                               text: str,
                               max_tokens: int) -> str:
        """
        Truncate text to fit within token limit.

        Args:
            text: Text to truncate
            max_tokens: Maximum tokens allowed

        Returns:
            Truncated text
        """
        # Approximate: 1 token ≈ 4 characters
        max_chars = max_tokens * 4

        if len(text) <= max_chars:
            return text

        # Truncate and add ellipsis
        return text[:max_chars - 3] + "..."


if __name__ == '__main__':
    # Example usage
    builder = PromptBuilder(max_context_length=2048)

    # Sample retrieved documents
    docs = [
        {
            'document_id': 'auth_guide',
            'document_type': 'text',
            'text': 'The authentication system uses JWT tokens with 15-minute expiration.'
        },
        {
            'document_id': 'cache_manager',
            'document_type': 'code',
            'text': 'def validate_token(token: str) -> bool:\n    """Validate JWT token."""\n    ...'
        }
    ]

    query = "How does token validation work?"

    # Build prompt
    result = builder.build_prompt(query, docs)

    print("=== Generated Prompt ===")
    print(result['prompt'])
    print(f"\n=== Metadata ===")
    print(f"Documents included: {result['num_docs_included']}")
    print(f"Estimated tokens: {result['estimated_tokens']}")
