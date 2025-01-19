import argparse
from langchain.document_loaders import JSONLoader
from langchain.text_splitter import TokenTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# 创建 argparse 解析器
def parse_args():
    parser = argparse.ArgumentParser(description="RAG Chatbot")
    parser.add_argument('--json_file', type=str, required=True, help='路径到 JSON 数据文件')
    parser.add_argument('--model_path', type=str, required=True, help='模型路径')
    return parser.parse_args()

def metadata_func(record: dict, metadata: dict) -> dict:
    pub_date = record.get("pub_date", {})
    metadata["year"] = pub_date.get('year')
    metadata["month"] = pub_date.get('month')
    metadata["day"] = pub_date.get('day')
    metadata["title"] = record.get("article_title")
    return metadata

def rag_chatbot(user_query: str, show_refs: bool = True, top_k: int = 2) -> str:
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": top_k})
    
    # 每次调用时都构建一次Chain，若频繁调用可考虑在外部创建并缓存
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )
    
    result = chain({"query": user_query})
    answer = result["result"].strip()
    response = f"**RAG回答**: {answer}"
    if show_refs:
        docs = result["source_documents"]
        titles = [f"- {doc.metadata.get('title', '未知来源')}" for doc in docs]
        unique_titles = list(set(titles))
        refs = "\n".join(unique_titles)
        response += f"\n\n-Reference:\n{refs}"
    return response

def llm_chatbot(user_query: str) -> str:
    return llm_only_chain.run(user_query)

def main():
    # 解析命令行参数
    args = parse_args()

    # 加载 JSON 数据文件
    loader = JSONLoader(
        file_path=args.json_file,
        jq_schema=".[]",
        content_key="article_abstract",
        metadata_func=metadata_func
    )

    data = loader.load()
    print(f"一共加载了 {len(data)} 篇文章。")

    # 文本切分
    text_splitter = TokenTextSplitter(chunk_size=128, chunk_overlap=50)
    chunks = text_splitter.split_documents(data)
    print(f"原始 {len(data)} 篇文章，切分后得到 {len(chunks)} 个文本片段。")

    # 向量数据库构建
    embeddings_model = "intfloat/e5-large-unsupervised"
    embeddings = HuggingFaceEmbeddings(
        model_name=embeddings_model,
        model_kwargs={'device': 'cuda'},       
        encode_kwargs={'normalize_embeddings': False}
    )

    db = FAISS.from_documents(chunks, embeddings)
    print("向量索引构建完成！")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        load_in_8bit=False,    
        device_map="auto"      
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512
    )
    llm = HuggingFacePipeline(
        pipeline=pipe,
        model_kwargs={"temperature": 0.5, "max_length": 1024}
    )

    print("模型加载完成！")

    PROMPT_TEMPLATE = """Answer the question based **only** on the following context:
    [BEGIN CONTEXT]
    {context}
    [END CONTEXT]
    You are allowed to rephrase the answer based on the context. 
    Question: {question}
    """
    prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)

    PROMPT_TEMPLATE_LLM_ONLY = """Answer the given question only.
    Question: {question}
    Answer:
    """
    llm_only_prompt = PromptTemplate.from_template(PROMPT_TEMPLATE_LLM_ONLY)
    llm_only_chain = LLMChain(llm=llm, prompt=llm_only_prompt)

    print("\n=== 命令行对话机器人就绪 ===")
    print("你可以输入任意问题，或者输入 exit/quit 退出程序。")
    print("在每次提问后，会询问你是否使用 RAG（检索+生成）或仅使用 LLM。")
    print("------------------------------------------------------------")
    
    while True:
        user_input = input("\n用户: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("再见！")
            break
        
        mode = input("使用 RAG 回答 (输入 'rag')，或仅使用 LLM 回答 (输入 'llm')？[默认rag]: ").strip()
        if mode.lower() == "llm":
            answer = llm_chatbot(user_input)
            print(f"[仅LLM] 回答: {answer}")
        else:
            # 默认为 RAG
            answer = rag_chatbot(user_input, show_refs=True, top_k=2)
            print(answer)

if __name__ == "__main__":
    main()
