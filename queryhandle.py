from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import ChatPromptTemplate
from transformers import pipeline

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
you are a highly advanced AI model, that can make concise, but informative reports.
based on the given OBD-2 codes, create a proffessional report 

{context}

---

Answer the question based on the above context: {question}
"""


def main():
    # Create CLI.
    query_text=input("OBD codes: ")

    # Use Hugging Face Embeddings (free model)
    embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB
    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    if len(results) == 0 or results[0][1] < 0.7:
        print(f"Unable to find matching results.")
        return

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print(prompt)

    # Use a Hugging Face transformer pipeline (e.g., Falcon, Mistral, GPT2, etc.)
    hf_pipeline = pipeline(
        "text-generation",
        model="tiiuae/falcon-7b-instruct",  # or mistralai/Mistral-7B-Instruct-v0.1, etc.
        max_new_tokens=500,
        temperature=0.7,
        do_sample=True,
    )

    model = HuggingFacePipeline(pipeline=hf_pipeline)
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("source", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)


if __name__ == "__main__":
    main()
