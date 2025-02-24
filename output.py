from langchain_core.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.llms import CTransformers 
import chainlit as cl
#ACTIVATE VIRTUAL ENVIRONMENT
DB_CHROMA_PATH="vectorstores/db_chroma"

custom_prompt_template="""Use the following pieces of info to answer the user's question. If you don't know the answer, please just say that you don't know the answer, don't try to make up an answer.


Context:{context}
Question: {question}

Only returns the helpful answer below and nothing else.
Helpful answer:
"""

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vetor stores
    """

    prompt = PromptTemplate(template=custom_prompt_template, input_variables=['context','question'])

    return prompt

def load_llm():
    llm= CTransformers(
        model="llama-2-7b-chat.ggmlv3.q8_0.bin",
        model_type ="llama",
        max_new_tokens=512,
        temperature=0.5
    )
    return llm
 
def retrieval_qa_chain(llm,prompt,db):
    qa_chain=RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={'k':2}),
        return_source_documents= True,
        chain_type_kwargs={'prompt': prompt,"document_variable_name": "context"}
    )

    return qa_chain

def qa_bot():
     embeddings= HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                      model_kwargs={'device': 'cpu'})
     db = Chroma(persist_directory=DB_CHROMA_PATH, embedding_function=embeddings)
     llm=load_llm()
     qa_prompt=set_custom_prompt()
     qa=retrieval_qa_chain(llm,qa_prompt,db)

     return qa

def final_result(query):
    qa_result=qa_bot()
    response=qa_result({'query': query})
    return response

## chainlit decoration##
@cl.on_chat_start
async def start():
    chain=qa_bot()
    if chain is None:
        raise ValueError("qa_bot() returned None. Check initialization.")
    msg=cl.Message(content="Starting the bot.....")
    await msg.send()
    msg.content="Hi, Welcome to the med Bot. What's your question?"
    await msg.update()
    cl.user_session.set("chain",chain)

@cl.on_message
async def main(message):
    chain = cl.user_session.get("chain")      
    if chain is None:
        await cl.Message(content="Error: Chatbot is not initialized. Please restart.").send()
        return
    cb=cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True,answer_prefix_tokens=["FINAL","ANSWER"]
    )
    cb.answer_reached=True
    res=await chain.ainvoke({"query": message.content},callbacks=[cb])
    answer=res["result"]
    sources=res["source_documents"]

    if sources:
        answer+= f"\nSources:" + str(sources)
    else:
        answer+= f"\nNo sources found"

    await cl.Message(content=answer).send()
