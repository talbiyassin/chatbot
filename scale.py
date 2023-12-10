import streamlit as st
from PyPDF2 import PdfReader
from langchain.chains import RetrievalQA
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma
import time
import io


# Logo
logo = "logo1.png"
st.image(logo, caption='Your Logo', use_column_width=True)

# Set your OpenAI API key here
openai_api_key = "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"

# Create OpenAIEmbeddings instance with the API key
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

# Title and description
st.title("Retrieval-Augmented Chatbot using LangChain")
st.write("A chatbot that integrates information from a PDF file using LangChain and OpenAI GPT3.5-Turbo. technique TEST for Scale ")

# File upload
pdf_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if pdf_file:
    # Read the uploaded file
    pdf_bytes = pdf_file.read()

    # Create a file-like object from the bytes
    pdf_file_like = io.BytesIO(pdf_bytes)

    # PDF File Processing with LangChain
    st.subheader("Processing PDF with LangChain...")

    # Record start time
    start_time = time.time()

    # Process the PDF file using PyPDF2
    pdf_reader = PdfReader(pdf_file_like)

    # Extract text from each page
    page_texts = [page.extract_text() for page in pdf_reader.pages]

    # Print the first 10 words from each page just to test that he read the pdf
    for i, text in enumerate(page_texts):
        st.subheader(f"Page {i + 1}: First 10 words")
        text = text or ""  # Check if text is None, set it to an empty string
        words = text.split()[:10]
        st.write(" ".join(words))

    # Create the vector store to use as an index
    db = Chroma.from_texts(page_texts, embeddings)

    # Expose this index in a retrieval interface
    retriever = db.as_retriever(
        search_type="similarity", search_kwargs={"k": 2}
    )

    # Create a chain to answer questions
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(openai_api_key=openai_api_key),
        chain_type="map_reduce",
        retriever=retriever,
        return_source_documents=True,
        verbose=True,
    )

    # Record end time
    end_time = time.time()

    # Calculate the time taken
    time_taken = end_time - start_time
    st.success(f"PDF processed successfully! Time taken: {time_taken:.2f} seconds")

    # User Interface
    st.subheader("Chat with the Chatbot:")

    # User input
    user_query = st.text_area("Your message:", height=100)

    # Display chat history
    st.subheader("Chat History:")
    st.text("User: " + user_query)

    # Get Answer button
    if st.button("Send"):
        if user_query:
            # Use the chatbot to get the answer
            try:
                answers = list(qa.stream([user_query]))
                if answers:
                    answer = answers[0].get('answer', "Sorry, I couldn't find an answer.")
                    st.text("Chatbot: " + answer)
                else:
                    st.warning("No answer found.")
            except Exception as e:
                st.error(f"Error: {str(e)}")
                print(f"Error: {str(e)}")
        else:
            st.warning("Please enter a message.")

# Additional notes and instructions
st.write(
    "Note: This is a simplified chatbot interface. For more advanced features, consider integrating the REST API for user interactions."
)

# Testing and Evaluation
st.header("Testing and Evaluation:")
st.write(
    "Conduct tests using predefined queries to evaluate the chatbot's performance in terms of accuracy, relevance, and coherence."
)
st.write("Document the testing process and results.")
