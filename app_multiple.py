import os
import streamlit as st
from dotenv import load_dotenv
from PIL import Image
import pytesseract
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceEndpoint
from googletrans import Translator
from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph
import pdfplumber
import yt_dlp
import whisper
import shutil

def extract_docx_text(file):
    text = ""
    doc = Document(file)

    for element in doc.element.body:
        if element.tag.endswith("}p"):  # Paragraph
            para = Paragraph(element, doc)
            if para.text.strip():
                text += f"Paragraph: {para.text.strip()}\n"

        elif element.tag.endswith("}tbl"):  # Table
            table = Table(element, doc)
            text += "Table:\n"
            for row in table.rows:
                row_text = [cell.text.strip() for cell in row.cells]
                text += " | ".join(row_text) + "\n"
            text += "\n"

    return text

def extract_pdf_text(file):
    output = ""

    with pdfplumber.open(file) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            elements = [] 

            words = page.extract_words(use_text_flow=True)
            if words:
                lines = {}
                for word in words:
                    y = round(word["top"], 1)
                    if y not in lines:
                        lines[y] = []
                    lines[y].append(word)
                
                for y in lines:
                    line_words = sorted(lines[y], key=lambda w: w["x0"])
                    line_text = " ".join(w["text"] for w in line_words)
                    elements.append({
                        "type": "paragraph",
                        "y": y,
                        "content": line_text.strip()
                    })

            for table in page.extract_tables():
                if not table:
                    continue
                elements.append({
                    "type": "table",
                    "y": page.bbox[3] - page.bbox[1],
                    "content": table
                })

            elements.sort(key=lambda e: e["y"])

            output += f"\n--- Page {page_num} ---\n"
            for elem in elements:
                if elem["type"] == "paragraph":
                    output += f"Paragraph: {elem['content']}\n"
                elif elem["type"] == "table":
                    output += "Table:\n"
                    for row in elem["content"]:
                        cleaned_row = [cell.strip() if cell else "" for cell in row]
                        output += " | ".join(cleaned_row) + "\n"
                    output += "\n"

    return output


def get_uploaded_file_text(files):
    text = ""
    for file in files:
        filename = file.name

        text += f"\n--- Processing file: {filename}. and files data is follows ---\n"
        if file.type == "application/pdf":
            text += extract_pdf_text(file)

        elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            text += extract_docx_text(file)

        elif file.type.startswith("image/"):
            img = Image.open(file)
            pytesseract.pytesseract.tesseract_cmd = f"{os.environ['TESSERACT_CMD']}"
            text += pytesseract.image_to_string(img)
        
        text+= f"\n processing of {filename} done.\n"
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def translate_text(text, response_language):
    translator =Translator()
    translated_text = translator.translate(text, dest=response_language)
    return translated_text.text

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vetorestore):
    llm_model = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    llm = HuggingFaceEndpoint(
            repo_id=llm_model, 
            task="text-generation",
            temperature = 0.7,
            max_new_tokens =1024,
            top_k = 3,
            load_in_8bit = True,
        )
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vetorestore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question, target_language, response_language):
    translated_question = translate_text(user_question, target_language)
    response = st.session_state.conversation({'question': translated_question})
    st.session_state.chat_history = response['chat_history']
    
    response_container = st.container()
    with response_container:
        for i, messages in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                message(messages.content, is_user=True, key=str(i))
            else:
                message(translate_text(messages.content, response_language), key=str(i))

def download_video(videoURL):
    if os.path.exists("audio"):
        shutil.rmtree("audio")

    os.makedirs("audio", exist_ok=True)

    print("Downloading from:", videoURL)

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': 'audio/ytAudio.%(ext)s',  
        'postprocessors': [
            {
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
            },
        ],
        'quiet': False
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([videoURL])

def audio_transcribe(path):
    model = whisper.load_model("base")
    fullTranscript = ""

    path = 'audio/ytAudio.wav'
    if os.path.exists(path):
        print(f"Transcribing: {path}")
        result = model.transcribe(path)
        fullTranscript += result["text"] + "\n"
    else:
        print(f"Missing file: {path}")

    return fullTranscript.strip()

def youtube_process(videoURL):
    download_video(videoURL)
    inputFile = "audio/ytAudio.wav"
    summarizedText = audio_transcribe(inputFile)
    print(summarizedText)
    return summarizedText    

def main():
    load_dotenv()
    st.set_page_config(page_title="Query Vault", page_icon=":bot:")
    
    target_language = st.selectbox("Select Document language (Target)", ["en", "es", "fr", "de"], key="target_language")
    response_language = st.selectbox("Select Document language (Response)", ["en", "es", "fr", "de"], key="response_language")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "user_question" not in st.session_state:
        st.session_state.user_question = ""
    if "youtube_link" not in st.session_state:
        st.session_state.youtube_link = ""
    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None

    st.header("Query Vault :books:")

    with st.sidebar:
        st.subheader("Your documents")
        files = st.file_uploader("Upload your Docs here and click on 'Process'", accept_multiple_files=True)
        
        for file in files:
            print(f"{file.name} \n")

        if st.button("Process"):
            with st.spinner("Processing"):
                raw_text = get_uploaded_file_text(files)
                text_chunks = get_text_chunks(raw_text)
                vectorstore = get_vectorstore(text_chunks)
                st.session_state.conversation = get_conversation_chain(vectorstore)
                st.session_state.processComplete = True
       
        st.header("OR")
        
        st.subheader("Youtube Link")
        youtube_link = st.text_input("Insert video link and click on 'Process Link'")
        if st.button("Process Link"):
            with st.spinner("Processing"):
                raw_text = youtube_process(youtube_link)
                text_chunks = get_text_chunks(raw_text)
                vectorstore = get_vectorstore(text_chunks)
                st.session_state.conversation = get_conversation_chain(vectorstore)
                st.session_state.processComplete = True

    if st.session_state.processComplete == True:
        user_question = st.chat_input("Ask Question about your files.")
        if user_question:
            handle_userinput(user_question, target_language, response_language)

if __name__ == '__main__':
    main()
