# Nepal Constitution 2072 | RAG Application

This is a basic Retrieval-Augmented Generation (RAG) application that provides information about the Constitution of Nepal 2072.

## Files

1. **`populate_database.py`**: Loads, processes, and stores document chunks into a Chroma vector database.
2. **`chat.py`**: Manages user interactions and generates responses using the RAG approach.

## Setup

1. Clone this repository.
2. Install the necessary dependencies:
3. Create a `.env` file in the project root and add your OpenAI API key: `OPENAI_API_KEY=your_api_key_here`

## Usage

1. **Populate the database**:

   ```
   python populate_database.py
   ```

   This script will process the PDF documents in the `data` directory and store them in a Chroma database.

2. **Run the chat application**:

   ```
   python chat.py
   ```

   This will initiate a chat session where you can ask questions about the Constitution of Nepal 2072.
