"""
BufferMemory wrapper.  For now it's in-process; in Day 4 we'll switch to
RedisJSON by replacing TWO lines here.
"""

from langchain.memory import ConversationBufferMemory

_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)


def get_memory():
    return _memory
