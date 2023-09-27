# Bring in deps
import os 
from apikey import apikey 

import streamlit as st 
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain 
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper
import jinja2
import PyPDF2

os.environ['OPENAI_API_KEY'] = apikey

# App framework
st.title('🦜🔗 Research Article Summarizer')

# Accept PDF Uploads
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

# Prompt templates
title_template = PromptTemplate(
    input_variables = ['topic'], 
    template='Based on this research article {topic}, answer the following questions: Who are the authors of the article? What are their academic or research affiliations? Have these authors published other notable works in this field? Summarize the abstract in 2-3 sentences. What are the key objectives or research questions presented? What research methods or approaches did the authors use? Are there any potential biases in their method? What are the primary findings of the study? Are the results statistically significant? What implications do they draw from their findings?'
)

# Llms
llm = OpenAI(model_name="gpt-4-0613", temperature=0.9)
title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True, output_key='title')

# Refined chunking function
def refined_chunk_by_paragraphs(text, max_chars):
    """Break text into chunks based on paragraphs and a maximum character length."""
    paragraphs = text.replace('\n\n', '\n').split('\n')
    chunks = []
    current_chunk = ""
    
    for paragraph in paragraphs:
        if len(current_chunk) + len(paragraph) > max_chars:
            chunks.append(current_chunk.strip())
            current_chunk = ""
        
        current_chunk += paragraph + "\n\n"
    
    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

MAX_CHARS_PER_CHUNK = 4000  # Adjust this value as needed

# If a PDF is uploaded
if uploaded_file:
    # Extract text from the uploaded PDF
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    pdf_text = ""
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        pdf_text += page.extract_text()
    
    # Use the extracted text and break it into chunks
    chunks = refined_chunk_by_paragraphs(pdf_text, MAX_CHARS_PER_CHUNK)

    responses = []

    # Analyze each chunk incrementally
    for chunk in chunks:
        title = title_chain.run(chunk)
        responses.append(title)

    # Display the aggregated results
    for response in responses:
        st.write(response)
