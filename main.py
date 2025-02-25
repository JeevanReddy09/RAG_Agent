import argparse
from vector_store import load_and_process_pdf, create_chroma_vector_store, create_pinecone_vector_store
from qa_agent import create_tools, create_zero_shot_agent

def main():
    parser = argparse.ArgumentParser(description="LangChain QA Agent")
    parser.add_argument("--pdf", type=str, help="Path to the syllabus PDF", required=True)
    parser.add_argument("--store", type=str, choices=["chroma", "pinecone"], default="pinecone", help="Vector store type")
    args = parser.parse_args()

    # Load PDF and process text
    chunks = load_and_process_pdf(args.pdf)

    # Choose Vector Store
    if args.store == "chroma":
        vector_store = create_chroma_vector_store(chunks)
    else:
        vector_store = create_pinecone_vector_store(chunks)

    # Initialize agent
    tools = create_tools(vector_store)
    agent = create_zero_shot_agent(tools)

    print("Agent is ready. Type your questions:")
    while True:
        user_question = input("\nAsk a question (or type 'exit' to quit): ").strip()
        if user_question.lower() == "exit":
            break

        try:
            answer = agent.run(user_question)
            print(f"\nAnswer: {answer}")

        except Exception as e:
            print(f"⚠️ Error: {e}")
            print("The system encountered an issue. Try rephrasing your question.")

if __name__ == "__main__":
    main()
