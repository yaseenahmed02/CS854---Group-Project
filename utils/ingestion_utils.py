import re
import os
from typing import List, Dict, Any, Optional
from transformers import AutoTokenizer

def sanitize_path_component(name: str) -> str:
    """
    Convert a string into a safe directory name.
    Replaces non-alphanumeric characters (except - and _) with _.
    
    Args:
        name: Input string (e.g., "Automattic/wp-calypso")
        
    Returns:
        Safe string (e.g., "Automattic_wp_calypso")
    """
    # Replace non-alphanumeric characters (including -) with _
    safe_name = re.sub(r'[^\w]', '_', name)
    return safe_name

class SemanticChunker:
    """
    Chunker designed for Jina-v2 (8192 tokens).
    Attempts to keep files intact, splitting only when necessary using tree-sitter.
    """
    
    def __init__(self, model_name: str = "jinaai/jina-embeddings-v2-base-code"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.max_tokens = 8192
        
        # Try to import tree_sitter_languages, but handle if missing
        try:
            from tree_sitter_languages import get_language, get_parser
            self.get_language = get_language
            self.get_parser = get_parser
            self.has_tree_sitter = True
        except ImportError:
            print("Warning: tree_sitter_languages not found. Semantic splitting will fallback to simple chunking.")
            self.has_tree_sitter = False

    def count_tokens(self, text: str) -> int:
        """Count tokens in text using the model's tokenizer."""
        return len(self.tokenizer.encode(text, truncation=False))

    def chunk_file(self, file_content: str, file_path: str) -> List[str]:
        """
        Chunk a file into parts that fit within max_tokens.
        
        Args:
            file_content: Content of the file
            file_path: Path to the file (used to determine language)
            
        Returns:
            List of text chunks
        """
        token_count = self.count_tokens(file_content)
        
        # Case 1: File fits in context window
        if token_count <= self.max_tokens:
            return [file_content]
            
        # Case 2: File is too large, need to split
        print(f"File {file_path} exceeds token limit ({token_count} > {self.max_tokens}). Splitting...")
        
        if self.has_tree_sitter:
            return self._split_with_tree_sitter(file_content, file_path)
        else:
            return self._split_naive(file_content)

    def _split_with_tree_sitter(self, content: str, file_path: str) -> List[str]:
        """Split code using AST parsing."""
        ext = os.path.splitext(file_path)[1].lower()
        lang_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.tsx': 'tsx',
            '.jsx': 'javascript',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.go': 'go',
            '.rs': 'rust'
        }
        
        language_name = lang_map.get(ext)
        if not language_name:
            return self._split_naive(content)
            
        try:
            language = self.get_language(language_name)
            parser = self.get_parser(language_name)
            tree = parser.parse(bytes(content, "utf8"))
            
            chunks = []
            current_chunk = ""
            
            # Traverse top-level nodes (classes, functions)
            cursor = tree.walk()
            if cursor.goto_first_child():
                while True:
                    node = cursor.node
                    node_text = content[node.start_byte:node.end_byte]
                    
                    # If a single node is too big, we might need to split it further (not implemented for now, fallback to naive)
                    if self.count_tokens(node_text) > self.max_tokens:
                        # If current chunk has content, save it
                        if current_chunk:
                            chunks.append(current_chunk)
                            current_chunk = ""
                        # Add the big node as its own chunk (or split naively if strictly enforced)
                        # For now, we'll just append it and warn, or split naively
                        chunks.extend(self._split_naive(node_text))
                    else:
                        # Check if adding this node exceeds limit
                        if self.count_tokens(current_chunk + "\n" + node_text) > self.max_tokens:
                            chunks.append(current_chunk)
                            current_chunk = node_text
                        else:
                            if current_chunk:
                                current_chunk += "\n" + node_text
                            else:
                                current_chunk = node_text
                    
                    if not cursor.goto_next_sibling():
                        break
                
                if current_chunk:
                    chunks.append(current_chunk)
                    
            return chunks if chunks else [content]
            
        except Exception as e:
            print(f"Tree-sitter parsing failed for {file_path}: {e}")
            return self._split_naive(content)

    def _split_naive(self, content: str) -> List[str]:
        """Fallback splitting by lines."""
        lines = content.splitlines()
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for line in lines:
            line_tokens = self.count_tokens(line)
            if current_tokens + line_tokens > self.max_tokens:
                chunks.append("\n".join(current_chunk))
                current_chunk = [line]
                current_tokens = line_tokens
            else:
                current_chunk.append(line)
                current_tokens += line_tokens
                
        if current_chunk:
            chunks.append("\n".join(current_chunk))
            
        return chunks
