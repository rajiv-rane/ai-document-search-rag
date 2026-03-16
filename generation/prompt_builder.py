from typing import List, Dict, Any

class PromptBuilder:
    """
    Constructs strict prompts for the LLM to ensure grounded answers.
    """
    
    SYSTEM_INSTRUCTION = (
        "You are a helpful assistant that answers questions based ONLY on the provided context.\n"
        "If the answer is not in the context, say: 'The documents do not contain this information.'\n"
        "Cite the document name and page number for every fact you state."
    )

    def build_prompt(self, query: str, context_chunks: List[Dict[str, Any]]) -> str:
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
            f"{self.SYSTEM_INSTRUCTION}\n\n"
            f"### Context:\n{context_text}\n"
            f"### Question:\n{query}\n\n"
            f"### Answer:"
        )
        return prompt
