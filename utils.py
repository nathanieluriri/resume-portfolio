import requests

def google_search(query, api_key, cx):
    url = f"https://www.googleapis.com/customsearch/v1?q={query}&key={api_key}&cx={cx}"
    
    response = requests.get(url)

    if response.status_code == 200:
        search_results = response.json() 
        return search_results
    else:
        print(f"Error: {response.status_code}")
        return None




def generate_embedding_for_user_resume(data,user_id):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("nomic-ai/nomic-embed-text-v1", trust_remote_code=True)


    def get_embedding(data, precision="float32"):
        return model.encode(data, precision=precision)


    from pinecone import Vector
    def create_docs_with_vector_embeddings(bson_float32, data):
        docs = []
        for i, (bson_f32_emb, text) in enumerate(zip(bson_float32, data)):
                doc =Vector(
                id=f"{i}",
                values= bson_f32_emb.tolist(),
                metadata={"text":text,"user_id":user_id},
                )
                docs.append(doc)
        return docs
    float32_embeddings = get_embedding(data, "float32")




    docs = create_docs_with_vector_embeddings(float32_embeddings,  data)
    return docs


def insert_embeddings_into_pinecone_database(doc,api_key,name_space):
    from pinecone import Pinecone
    pc = Pinecone(api_key=api_key)
    index_name = "resumes"
    index = pc.Index(index_name)
    upsert_response = index.upsert(namespace=name_space,vectors=doc)
    return upsert_response




def query_vector_database(query,api_key,name_space):
    from pinecone import Pinecone
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("nomic-ai/nomic-embed-text-v1", trust_remote_code=True)
    ret=[]
    pc = Pinecone(api_key=api_key)
    index_name = "resumes"


    index = pc.Index(index_name)
    
    # Define a function to generate embeddings in multiple precisions
    def get_embedding(data, precision="float32"):
        return model.encode(data, precision=precision)
    
    query_embedding = get_embedding(query, precision="float32")

    response = index.query(
        namespace=name_space,
        vector=query_embedding.tolist(),
        top_k=3,
        include_metadata=True
        )


    for doc in response['matches']:
        ret.append(doc['metadata']['text'])
    return ret


def delete_vector_namespace(name_space,api_key):
    from pinecone import Pinecone
    pc = Pinecone(api_key=api_key)
    index_name = "resumes"


    index = pc.Index(index_name)
    response = index.delete(delete_all=True,namespace=name_space)
    return response



def split_text_into_chunks(text, chunk_size=400):
    # Split the text into words using whitespace.
    words = text.split()

    # Group the words into chunks of size 'chunk_size'.
    chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

