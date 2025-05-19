from typing import List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.chat_history import BaseChatMessageHistory
import psycopg2
from psycopg2.extras import RealDictCursor
import os

class PostgresChatMessageHistory(BaseChatMessageHistory):
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.conn = psycopg2.connect(
            dbname="chatbotdb",
            user="postgres",
            password=os.getenv("POSTGRES_PASSWORD"),
            host="localhost",
            port="5432"
        )

    def add_message(self, message: BaseMessage) -> None:
        role_map = {
            HumanMessage: "user",
            AIMessage: "assistant",
            SystemMessage: "system"
        }
        role = role_map.get(type(message), "system")

        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO message_store (session_id, message_type, content, role)
                VALUES (%s, %s, %s, %s)
            """, (
                self.session_id,
                'text',
                message.content,
                role
            ))
            self.conn.commit()

    def get_messages(self) -> List[BaseMessage]:
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT role, content FROM message_store
                WHERE session_id = %s
                ORDER BY created_at ASC
            """, (self.session_id,))
            rows = cur.fetchall()

        messages = []
        for row in rows:
            role = row["role"]
            content = row["content"]
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))
            elif role == "system":
                messages.append(SystemMessage(content=content))
        return messages

    @property
    def messages(self) -> List[BaseMessage]:
        return self.get_messages()

    def clear(self) -> None:
        with self.conn.cursor() as cur:
            cur.execute("DELETE FROM message_store WHERE session_id = %s", (self.session_id,))
            self.conn.commit()
