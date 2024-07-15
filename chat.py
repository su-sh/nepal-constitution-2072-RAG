import os
import re
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, SystemMessage
from langchain.output_parsers import ResponseSchema, StructuredOutputParser

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CHROMA_PATH = "chroma"

SYSTEM_TEMPLATE = """
You are an AI assistant that provides information about the Constitution of Nepal 2072.
Be knowledgeable, helpful, friendly, and accurate in providing information based on the document's content.

Important rules:
0. You can greet users and provide information about the Constitution of Nepal 2072.
1. Answer questions related to the Constitution of Nepal 2072.
2. If a question is not about the Constitution, politely refuse to answer.
3. Base your answers solely on the provided context.
4. If the context doesn't contain relevant information, say you don't have enough information to answer accurately.
5. Do not make up or infer information that's not in the context.
6. Your responses must be directly related to the Constitution of Nepal 2072.
"""

def get_embedding_function():
    """Return the OpenAI embedding function."""
    return OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)


def initialize_components():
    """Initialize and return memory, database, and model components."""
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    memory = ConversationBufferMemory(return_messages=True)
    model = ChatOpenAI(
        model_name="gpt-3.5-turbo-0125",
        openai_api_key=OPENAI_API_KEY,
        temperature=0.2
    )
    return memory, db, model


def get_context(db, query_text, k=5):
    """Retrieve context from the database based on the query."""
    results = db.similarity_search_with_score(query_text, k=k)
    return "\n\n---\n\n".join([doc.page_content for doc, _score in results])


def sanitize_input(text):
    """Sanitize input text by removing special characters."""
    return re.sub(r'[^a-zA-Z0-9\s\.\,\?\!]', '', text)


def generate_response(model, system_message, memory, context_text, query_text):
    response_schemas = [
        ResponseSchema(
            name="answer",
            description="The answer to the user's question based on the provided context.",
        ),
        ResponseSchema(
            name="confidence",
            description="A confidence score from 0 to 1 indicating how certain the AI is about the answer.",
        ),
    ]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

    conversation_history = memory.chat_memory.messages

    HUMAN_MESSAGE_CONTEXT = f"""
    Current context:
    {context_text}
    
    
    Question: {query_text}
    {output_parser.get_format_instructions()}
    

    Remember to follow the important rules outlined in your instructions.
    """
    messages = [
        system_message,
        *conversation_history,
        HumanMessage(content=HUMAN_MESSAGE_CONTEXT),
    ]
    response = model.invoke(messages)

    try:
        parsed_response = output_parser.parse(response.content)
        answer = parsed_response["answer"]
        confidence = float(parsed_response["confidence"])

        if confidence < 0.7:
            return "I lack sufficient confidence to answer. Could you rephrase or ask about a different aspect of Nepal's 2072 Constitution?"

        return answer
    except Exception as e:
        print(f"Error parsing response: {e}")
        return "I'm unable to provide a proper answer. Could you restate your question?"


def main():
    if not OPENAI_API_KEY:
        print("OpenAI API key is missing. Please provide the API key to continue.")
        return

    memory, db, model = initialize_components()
    system_message = SystemMessage(content=SYSTEM_TEMPLATE)

    print(
        "You can start chatting with the AI about the Constitution of Nepal 2072. Type 'exit' to stop the conversation."
    )
    while True:
        query_text = input("You: ").strip()
        if query_text.lower() == "exit":
            break

        sanitized_query = sanitize_input(query_text)

        context_text = get_context(db, sanitized_query)

        try:
            response_text = generate_response(
                model, system_message, memory, context_text, sanitized_query
            )
            print(f"AI: {response_text}")
            memory.chat_memory.add_user_message(sanitized_query)
            memory.chat_memory.add_ai_message(response_text)
        except Exception as e:
            print(f"An error occurred while generating the response: {e}")
            print(
                "AI: I apologize, but I encountered an error. Could you please try asking your question again?"
            )


if __name__ == "__main__":
    main()
