import re

def extract_diff(llm_response: str) -> str:
    """
    Extract the content between markdown code blocks from the LLM response.
    
    Args:
        llm_response: The raw text response from the LLM.
        
    Returns:
        The extracted code content.
    """
    # Pattern to find code blocks: ```[language]\n(content)\n```
    # We want to capture the content.
    # This regex looks for ``` followed by optional language identifier, then content, then ```
    pattern = r"```(?:\w+)?\n(.*?)```"
    
    matches = re.findall(pattern, llm_response, re.DOTALL)
    
    if matches:
        # Return the longest match (assuming the main file content is the largest block)
        # or just the first one if we expect only one.
        # Given the prompt "Output the full, valid file content enclosed in markdown code blocks",
        # we should probably take the largest one or the last one?
        # Let's take the longest one to be safe against small snippets.
        return max(matches, key=len).strip()
    
    # Fallback: if no code blocks, return the whole response (risky but better than empty)
    # or try to strip generic text.
    return llm_response.strip()
