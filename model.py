from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
# from langchain.llms import HuggingFacePipeline
# from langchain.llms import GooglePalm
# from langchain.chains import RetrievalQA
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

DB_FAISS_PATH = 'vectorstores/db_faiss'

question_template = """
Given the following conversation and a follow-up question, rephrase the follow-up question to be a standalone question. 
Don't try to change the follow-up question's key entities.
At the end of the standalone question, add this 'answer the standalone question.'

Chat History:
{chat_history}
Follow-Up Input: {question}
Standalone Question:
"""

input_variables = ["chat_history", "question"]
CUSTOM_QUESTION_PROMPT = PromptTemplate(template=question_template, input_variables=input_variables)


def load_llm():
    llm = CTransformers(
        model = "llama-2-7b-chat.ggmlv3.q2_K.bin",
        model_type='llama',
        max_new_tokens = 512,
        temperature = 0.1
    )
    return llm


# def load_llm():
#     llm = HuggingFacePipeline.from_model_id(
#         model_id="ngoantech/Llama-2-7b-vietnamese-20k",
#         task="text-generation",
#         model_kwargs={"do_sample": True,"temperature": 0.2, "max_length": 600},
#     )
#     return llm

memory = ConversationBufferMemory(memory_key="chat_history", output_key="answer", return_messages=True)

def retrieval_qa_chain(llm, db):
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        chain_type='stuff',
        retriever = db.as_retriever(search_kwargs={'k': 4}),
        # retriever = db.as_retriever(),
        condense_question_prompt=CUSTOM_QUESTION_PROMPT,
        return_source_documents = True,
        memory=memory,
        verbose=True,
    )

    return qa_chain


def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2"
                                          , model_kwargs = {'device': 'cpu'})

    db = FAISS.load_local(DB_FAISS_PATH, embeddings)
    llm = load_llm()
    qa = retrieval_qa_chain(llm, db)

    return qa



def final_result(query):
    qa_result = qa_bot()
    response = qa_result({'question': query})

    return response


# Khởi tạo chatbot
print("Hi, Welcome to LawBot. What is your query?")
while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        break
    response = final_result(user_input)

    print("LawBot:", response['answer'])
    print(response['chat_history'])
    print(response['source_documents'])