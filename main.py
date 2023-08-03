print("Hello World!")
from fastapi import FastAPI

import os
# import pandas as pd
import requests
import textract
import sys
# from bs4 import BeautifulSoup
# import matplotlib.pyplot as plt
from pypdf import PdfReader
from transformers import GPT2TokenizerFast
# from langchain.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings import OpenAIEmbeddings
# from langchain.vectorstores import FAISS
# from langchain.chains.question_answering import load_qa_chain
# from langchain.llms import OpenAI
# from langchain.chains import ConversationalRetrievalChain
from data import pdf_urls

# ここにOpenAIから取得したキーを設定します。
# os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
print("Error no 1111")
            sys.exit()
app = FastAPI()

@app.get("/")
def index():

    # PDFのダウンロードと保存を行う関数
    def download_and_save_pdf(url, filename):
        response = requests.get(url)
        if response.status_code == 200:
            with open(filename, 'wb') as file:
                file.write(response.content)
        else:
            print("Error: Unable to download the PDF file. The URL might be incorrect. Status code:", response.status_code)
            sys.exit()

    # PDFの読み込みを行う関数
    def read_pdf(filename):
        with open(filename, 'rb') as file:
            page_contents = ''
            reader = PdfReader(filename)
            number_of_pages = len(reader.pages)
            page_contents = ""
            for page_number in range(number_of_pages):
                pages = reader.pages
                page_contents += pages[page_number].extract_text()

            return page_contents




    # 各PDFの全ページからデータを取得
    pages_contents = ''
    for i, url in enumerate(pdf_urls):
        # PDFをダウンロードして保存
        filename = f'sample_document{i+1}.pdf'
        download_and_save_pdf(url, filename)
        
        # PDFを読み込み
        pages_contents += read_pdf(filename)


    chunks = pages_contents
    print('URLのPDF群から情報を抽出しました。')
    print(chunks)

    text = chunks

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    def count_tokens(text: str) -> int:
        return len(tokenizer.encode(text))

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        # chunk_size = 512,
        chunk_size = 512,
        chunk_overlap  = 24,
        length_function = count_tokens,
    )

    chunks3 = text_splitter.create_documents([text])

    print('step 222')
    print(chunks3)


    # DB利用

    # Get embedding model
    embeddings = OpenAIEmbeddings()

    #  vector databaseの作成
    db = FAISS.from_documents(chunks3, embeddings)

    # query = "所有者とアクセス許可設定を元に戻すは何ページ目ですか？"
    # query = "ローカルユーザーの共有フォルダーへのアクセスを制限する方法を教えてください"
    # query = "違反行為に対する抑止力の強化に関して何が改正されましたか？"
    # query = "共有フォルダーのデータを誤って消去しないためにはどうすればいい？"
    # query = "所有者とアクセス許可設定を元に戻すは何ページ目ですか？"
    query = "Active Directoryドメインユーザーの共有フォルダーへのアクセスを制限するにはどうすればいいですか？"

    embedding_vector = embeddings.embed_query(query)
    docs_and_scores = db.similarity_search_by_vector(embedding_vector)

    print('これがデータベースの中身です。')
    print(docs_and_scores)


    chain = load_qa_chain(OpenAI(temperature=0.1,max_tokens=1000), chain_type="stuff")

    message = chain.run(input_documents=docs_and_scores, question=query)
    print(query)
    print('に対する回答は以下の通りです。')
    print(message)
    # sys.exit()
    return {"Hello": "World"}