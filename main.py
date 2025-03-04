from io import BytesIO
from dotenv import load_dotenv
import os
from utils import google_search,split_text_into_chunks,insert_embeddings_into_pinecone_database,query_vector_database,generate_embedding_for_user_resume,delete_vector_namespace
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import docx
import fitz 
load_dotenv()

CX = os.getenv("SEARCH_ENGINE_ID")
API_KEY = os.getenv("API_KEY")
PINECONE_API_KEY=os.getenv("PINECONE_API_KEY")
GEMINI_API_KEY=os.getenv("GEMINI_API_KEY")
app = FastAPI()

@app.get("/get/course")
def get_course(query):
# Example search query
    results = google_search(query, API_KEY, CX)
    content=[]
   
    if results:
        for item in results.get('items', []):
            title = item.get('title')
            link = item.get('link')
            snippet = item.get('snippet')
            content_structure={}
            
            content_structure["Course_Title"]=title
            content_structure["Course_Link"]=link
            content_structure["Course_Snippet"]= snippet
            
            content.append(content_structure)
        
    
    return JSONResponse(content,status_code=200) 
            





@app.post("/upload")
async def upload_file(user_id,file: UploadFile = File(...)):
    content = await file.read()  # Read the file content (this will return bytes)
    sentences=[]

    # Print file details for debugging
    print(f"File name: {file.filename}")
    print(f"File content type: {file.content_type}")
    print(f"File size: {file.size} bytes")
    
    
    if "pdf" == file.filename.split('.')[1]:
        pdf_document = fitz.open(stream=BytesIO(content), filetype="pdf")
        # Print the content of the file (if it's text, you can decode it)
        extracted_text = ""
        for page_num in range(pdf_document.page_count):
            page = pdf_document.load_page(page_num)
            extracted_text += page.get_text()
            
    elif "docx" == file.filename.split('.')[1]:
        docx_file = BytesIO(content)
        doc = docx.Document(docx_file)
        extracted_text = ""
        for para in doc.paragraphs:
            extracted_text += para.text + "\n"
            
    sentences = split_text_into_chunks(extracted_text,chunk_size=200)
    docs = generate_embedding_for_user_resume(data=sentences,user_id=file.filename)
    response= insert_embeddings_into_pinecone_database(doc=docs,api_key=PINECONE_API_KEY,name_space=user_id)
    
    return {"filename": file.filename,"response":str(response) }    




@app.get("/ask")
def ask_ai_about_resume(query,user_id):
    context = query_vector_database(query=query,api_key=PINECONE_API_KEY,name_space=user_id)
    from google import genai

    client = genai.Client(api_key=GEMINI_API_KEY)
    response = client.models.generate_content(
        model="gemini-2.0-flash", contents=f"""
        Answer this question using the context provided
        question: {query}
        context: {context}
        """
    )
    
    return response.text