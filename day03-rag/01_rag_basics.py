"""
Day 3 - RAG 入门
把文档变成Agent能搜索的知识库
"""
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os
import pathlib
load_dotenv()

# DeepSeek不提供Embedding模型，这里用替代方案
# 方案1：如果你有OpenAI key，直接用OpenAI Embedding
# 方案2：用免费的本地Embedding（我们先用这个）

# ========== 1. 加载文档 ==========
print("== 1. 加载文档 ==")

script_dir = pathlib.Path(__file__).parent
loader = TextLoader(str(script_dir / "test_docs/product_manual.md"), encoding="utf-8")
docs = loader.load()
print(f"加载了 {len(docs)} 个文档，总字数: {len(docs[0].page_content)}")

# ========== 2. 切分文档 ==========
print("\n== 2. 切分文档 ==")

# 为什么要切分？因为LLM的上下文窗口有限，你不能把整篇文档塞进去
# 而且切成小块后，检索更精准——只找最相关的几块
splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,  # 每块最多300字符
    chunk_overlap=50,  # 块之间重叠50字符，防止切断句子
)
chunks = splitter.split_documents(docs)
print(f"切分成 {len(chunks)} 个文本块")
for i, chunk in enumerate(chunks):
    print(f"  块{i}: {chunk.page_content[:60]}...")

# ========== 3. Embedding + 存入向量数据库 ==========
print("\n== 3. 存入向量数据库 ==")

# Embedding：把文本变成一串数字（向量），语义相近的文本向量距离近
# 对标Java：相当于给每段文本算一个"语义指纹"

# 用OpenAI的Embedding（需要OpenAI key）
# 如果你没有OpenAI key，换成下面的免费方案
try:
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        # 如果你用的是OpenAI key就不用配base_url
        # 如果想省钱用国内代理，可以配base_url
    )
except:
    # 备用方案：用HuggingFace免费模型
    from langchain_community.embeddings import HuggingFaceEmbeddings

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 存入ChromaDB（向量数据库）
# 对标Java：相当于把数据存入Elasticsearch，但搜的是语义而不是关键词
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=str(script_dir / "chroma_db") # 持久化到本地
)
print(f"已存入 {len(chunks)} 个文本块到ChromaDB")

# ========== 4. 检索测试 ==========
print("\n== 4. 检索测试 ==")

# 语义搜索：不是关键词匹配，而是理解"意思"
queries = ["怎么退货", "会员有什么好处", "充电器多少钱"]

for query in queries:
    results = vectorstore.similarity_search(query, k=2)  # 找最相关的2个块
    print(f"\n搜索: '{query}'")
    for i, doc in enumerate(results):
        print(f"  结果{i + 1}: {doc.page_content[:80]}...")

# ========== 5. 完整RAG链 ==========
print("\n\n== 5. 完整RAG问答 ==")

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

# 创建检索器
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# RAG的Prompt模板
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", """你是智选商城的客服。根据以下参考资料回答用户问题。
规则：
1. 只基于参考资料回答，不要编造信息
2. 如果参考资料中没有相关信息，明确说"抱歉，我没有找到相关信息"
3. 回答简洁专业

参考资料：
{context}"""),
    ("user", "{question}")
])


def rag_answer(question: str) -> str:
    """完整的RAG流程：检索 → 拼接 → LLM回答"""
    # 第一步：检索相关文档
    docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in docs])

    # 第二步：拼接到Prompt里，让LLM基于文档回答
    chain = rag_prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})

    return answer, docs


# 测试
test_questions = [
    "退换货政策是什么？",
    "钻石会员有哪些权益？",
    "手机壳保修多久？",
    "你们卖电脑吗？",  # 文档里没有的信息
    "怎么开发票？",
]

for q in test_questions:
    answer, sources = rag_answer(q)
    print(f"\n问: {q}")
    print(f"答: {answer}")
    print(f"参考来源: {[s.page_content[:40] + '...' for s in sources]}")

print()
print("=" * 50)
print("RAG的核心流程：")
print("1. 文档 → 切分 → Embedding → 存入向量数据库（离线做一次）")
print("2. 用户提问 → Embedding → 从向量库搜相关段落（实时检索）")
print("3. 相关段落 + 用户问题 → 塞进Prompt → LLM基于真实内容回答")
print()
print("对比没有RAG的Agent：")
print("  没RAG：LLM凭训练数据回答，可能编造信息")
print("  有RAG：LLM基于你提供的文档回答，有据可查")
print("=" * 50)
