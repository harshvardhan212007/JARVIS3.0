"""
J.A.R.V.I.S. Conversation Memory
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Maintains rolling conversation history and builds the
LangChain message list for the LLM.

Responsibilities
────────────────
• Store conversation turns
• Return the last N conversation turns
• Build the complete message list for the LLM
• Clear memory when requested
"""
from langchain_core.messages import (
    BaseMessage,
    SystemMessage,
    HumanMessage,
    AIMessage,
)

class ConversationMemory:
    """Rolling in-memory conversation history."""

    def __init__(self, max_turns: int = 4):
        self.max_turns = max_turns
        self._history: list[tuple[HumanMessage, AIMessage]] = []

    def add(self, query: str, response: str) -> None:
        """Add one conversation turn."""
        self._history.append(
            (
                HumanMessage(content=query),
                AIMessage(content=response),
            )
        )

    def build_messages(
        self,
        system_prompt: str,
        context: str,
        query: str,
    ) -> list[BaseMessage]:
        """
        Build the complete LangChain message list.
        """

        messages: list[BaseMessage] = []

        # System prompt
        messages.append(
            SystemMessage(content=system_prompt)
        )

        # Previous conversation
        for human, ai in self._history[-self.max_turns:]:
            messages.append(human)
            messages.append(ai)

        # Current user query
        messages.append(
            HumanMessage(
                content=(
                    f"Context:\n{context}\n\n"
                    f"Query: {query}"
                )
            )
        )

        return messages
    def clear(self) -> None:
        """Erase all conversation history."""
        self._history.clear()

    def size(self) -> int:
        """Number of stored conversation turns."""
        return len(self._history)

    def is_empty(self) -> bool:
        """True if there is no stored conversation."""
        return len(self._history) == 0

    def get_history(self) -> list[tuple[HumanMessage, AIMessage]]:
        """Return a copy of the stored conversation."""
        return self._history.copy()

    def __len__(self) -> int:
        return len(self._history)