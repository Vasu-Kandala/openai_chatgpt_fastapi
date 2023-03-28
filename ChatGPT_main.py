import os
import json
import openai
import PyPDF2
import aiofiles    
import requests
import tiktoken
import gunicorn
import numpy as np
import pandas as pd
from typing import List
from flask import Response
from gunicorn.app.wsgiapp import run
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile, Form
from openai.embeddings_utils import distances_from_embeddings, cosine_similarity


# fastapi object
app = FastAPI()
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
#To get the health status of an API
@app.get("/api/health")
async def health():
    ok = dict()
    ok["message"] = "API up and running"
    ok["status"] = 200
    return Response(json.dumps(ok), status=200, mimetype="POC on OpenAI's ChatGPT")

# create new folder
des_path = os.getcwd()+r"/input_data/"
DESTINATION = des_path.replace("\\", "/")
max_tokens = 500
shortened = []
#Reading the Openai_api_key
with open(os.getcwd()+'\OpenAI_API_key.txt', 'r') as file:
    openai.api_key = file.read()
# create the function that extracts the text from pdfs
def content_extract_from_Pdf(DESTINATION):
    text = ''
    for filename in os.listdir(DESTINATION):
        if filename.endswith('.pdf'):
            # Open the PDF file in read-binary mode
            with open(os.path.join(DESTINATION, filename), 'rb') as f:
                # Create a PDFReader object
                pdf_reader = PyPDF2.PdfReader(f)
                num_pages = len(pdf_reader.pages)
                # Loop through each page of the PDF and extract the text
                for page_num in range(num_pages):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text()
                #   Print the extracted text
    return text  
              
# Function to split the text into chunks of a maximum number of tokens
def split_into_many(text, max_tokens = max_tokens):
    # Split the text into sentences
    sentences = text.split('. ')
    tokenizer = tiktoken.get_encoding("cl100k_base")
    # Get the number of tokens for each sentence
    n_tokens = [len(tokenizer.encode(" " + sentence)) for sentence in sentences]    
    chunks = []
    tokens_so_far = 0
    chunk = []
    # Loop through the sentences and tokens joined together in a tuple
    for sentence, token in zip(sentences, n_tokens):
        # If the number of tokens so far plus the number of tokens in the current sentence is greater 
        # than the max number of tokens, then add the chunk to the list of chunks and reset
        # the chunk and tokens so far
        if tokens_so_far + token > max_tokens:
            chunks.append(". ".join(chunk) + ".")
            chunk = []
            tokens_so_far = 0
        # If the number of tokens in the current sentence is greater than the max number of 
        # tokens, go to the next sentence
        if token > max_tokens:
            continue
        # Otherwise, add the sentence to the chunk and add the number of tokens to the total
        chunk.append(sentence)
        tokens_so_far += token + 1
    return chunks

# The function that compares the question embeddings to the embeddings in the dataset.
def create_context(
    question, df, max_len=1800, size="ada"
):
    """
    Create a context for a question by finding the most similar context from the dataframe
    """
    # Get the embeddings for the question
    q_embeddings = openai.Embedding.create(input=question, engine='text-embedding-ada-002')['data'][0]['embedding']
    # Get the distances from the embeddings
    df['distances'] = distances_from_embeddings(q_embeddings, df['embeddings'].values, distance_metric='cosine')
    returns = []
    cur_len = 0
    # Sort by distance and add the text to the context until the context is too long
    for i, row in df.sort_values('distances', ascending=True).iterrows():        
        # Add the length of the text to the current length
        cur_len += row['n_tokens'] + 4        
        # If the context is too long, break
        if cur_len > max_len:
            break        
        # Else add it to the text that is being returned
        returns.append(row["text"])
    # Return the context
    return "\n\n###\n\n".join(returns)

# The function that returns response for the question    
def answer_question(
    df,
    question="",
    max_len=1800,
    size="ada",
    debug=False,
    max_tokens=150,
    stop_sequence=None
):
    """
    Answer a question based on the most similar context from the dataframe texts
    """
    context = create_context(
        question,
        df,
        max_len=max_len,
        size=size,
    )
    prompt = f"""Answer the question as truthfully as possible using the provided text, and if the answer is not contained within the text below, say "Unable to find the Answer in the provided data"

    Context:{context}

    Q:{question}
    A:"""
    # If debug, print the raw model response
    if debug:
        print("Context:\n" + context)
        print("\n\n")
    try:
        # Create a completions using the question and context
        response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        return response['choices'][0]["message"]["content"]
    except Exception as e:
        print(e)
        return ""


@app.post("/upload-files")
async def create_upload_files(files: List[UploadFile] = File(...), question: str = Form(...)):
    shortened = []   
    for f in os.listdir(DESTINATION):
        os.remove(os.path.join(DESTINATION, f))
    for file in files:
        destination_file_path = DESTINATION+r"/"+file.filename #output file path               
        async with aiofiles.open(destination_file_path, 'wb') as out_file:            
            while content := await file.read(1024):  # async read file chunk               
                await out_file.write(content)  # async write file chunk                         
    text = content_extract_from_Pdf(DESTINATION)       
    df=pd.DataFrame(['0',text]).T
    # Load the cl100k_base tokenizer which is designed to work with the ada-002 model
    tokenizer = tiktoken.get_encoding("cl100k_base")
    #df = pd.read_csv('processed/scraped.csv', index_col=0)
    df.columns = ['title', 'text']
    # Tokenize the text and save the number of tokens to a new column
    df['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(x)))
    # Loop through the dataframe
    for row in df.iterrows():
        # If the text is None, go to the next row
        if row[1]['text'] is None:
            continue
        # If the number of tokens is greater than the max number of tokens, split the text into chunks
        if row[1]['n_tokens'] > max_tokens:
            # Calling split_into_many function
            shortened += split_into_many(row[1]['text'])        
        # Otherwise, add the text to the list of shortened texts
        else:
            shortened.append( row[1]['text'] )      
    df = pd.DataFrame(shortened, columns = ['text'])
    df['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(x)))    
    df['embeddings'] = df.text.apply(lambda x: openai.Embedding.create(input=x, engine='text-embedding-ada-002')['data'][0]['embedding'])
    df.to_csv(DESTINATION+r'/generated_embeddings_500.csv') # , index = False
    df=pd.read_csv(DESTINATION+r'/generated_embeddings_500.csv', index_col=0)
    df['embeddings'] = df['embeddings'].apply(eval).apply(np.array)
    answer = answer_question(df,question, debug=False)    
    return {question:answer}
    
@app.post("/question")
async def QnA(question: str = Form(...)):
    df=pd.read_csv(os.getcwd()+r'/openAI_embeddings_500.csv', index_col=0)
    df['embeddings'] = df['embeddings'].apply(eval).apply(np.array)
    answer = answer_question(df,question, debug=False)
    return {question: answer}     


