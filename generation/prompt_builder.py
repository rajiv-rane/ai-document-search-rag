from typing import List, Dict, Any

class PromptBuilder:
    """
    Constructs strict prompts for the LLM to ensure grounded answers.
    """
    
    SYSTEM_INSTRUCTION = (
        "You are a strict, factual AI assistant. You must adhere to the following rules:\n"
        "1. Answer the query using ONLY the provided context passages.\n"
        "2. If the answer cannot be found in the context, reply exactly with: \"I'm sorry, but the provided documents do not contain enough information to answer this question.\"\n"
        "3. For every factual claim, you must extract and append the filename and page number from the metadata of the retrieved chunk as an explicit inline citation."
    )

    def build_user_prompt(self, query: str, context_chunks: List[Dict[str, Any]]) -> str:
        """
        Combines retrieved chunks into a formatted prompt string.
        """
        context_text = ""
        for i, chunk in enumerate(context_chunks):
            metadata = chunk.get("metadata", {})
            file_name = metadata.get("file_name", "Unknown")
            page = metadata.get("page_number", "Unknown")
            
            context_text += f"\n--- Document Segment {i+1} (Source: {file_name}, Page: {page}) ---\n"
            context_text += chunk.get("text", "") + "\n"

        prompt = (
            f"### Context:\n{context_text}\n"
            f"### Question:\n{query}\n\n"
            f"### Answer:"
        )
        return prompt
