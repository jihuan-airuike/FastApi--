# 加载环境变量库，读取 .env 文件中的密钥配置
from dotenv import load_dotenv
# FastAPI 核心库，用于构建异步Web接口；Request接收请求对象；UploadFile、File用于文件上传
from fastapi import FastAPI, Request, UploadFile, File
# Jinja2模板，用于渲染前端html页面
from fastapi.templating import Jinja2Templates
# openai SDK，兼容智谱等OpenAI协议接口；BaseModel用于请求体pydantic数据模型校验
from openai import OpenAI, BaseModel
# uvicorn ASGI服务运行器，启动FastAPI服务
import uvicorn
# chromadb 本地向量数据库，用于RAG向量存储与语义检索
import chromadb
# uuid 生成唯一id，作为向量文档的唯一标识
import uuid
# sqlalchemy会话对象，操作数据库会话
from sqlalchemy.orm import Session
# 导入数据库模型与数据库引擎
from db_models import QuestionLog, engine
# 异步redis客户端，用于缓存问答结果
import redis.asyncio as redis
# LangChain文本分割器，对原始文档做分块处理
from langchain_text_splitters import RecursiveCharacterTextSplitter
# os模块，读取环境变量
import os

# 创建FastAPI应用实例
app = FastAPI(
    title='doctor',
    description='内部知识问答系统',
    version='1.0',
)

# 创建Redis连接池，复用连接，提升性能
redis_pool = redis.ConnectionPool(
    host='localhost',       # redis服务地址
    port=6379,              # redis端口
    db=0,                   # 使用第0号数据库
    decode_responses=True,  # 自动将bytes解码为字符串
    max_connections=10,     # 最大连接数
)
# 获取redis异步客户端实例
redis_client = redis.Redis(connection_pool=redis_pool)

# 加载项目根目录下 .env 环境变量文件，存放API密钥等敏感信息
load_dotenv()

# 初始化智谱AI大模型客户端，兼容OpenAI接口格式
openai_zhipu = OpenAI(
    api_key=os.getenv("ZHIPU_API_KEY"),          # 从环境变量读取智谱密钥
    base_url='https://open.bigmodel.cn/api/paas/v4', # 智谱API地址
)

# Pydantic请求模型，/answer接口接收post请求体参数校验
class Question(BaseModel):
    question: str       # 用户提问内容
    model: str = 'glm-4-flash' # 指定调用的大模型，默认glm‑4‑flash


# 初始化持久化Chroma向量数据库，数据保存在 ./persistent.db 目录
client = chromadb.PersistentClient(path='./persistent.db')
# 获取/创建向量集合collects，文档向量全部存放在该集合
collects = client.get_or_create_collection('collects')

# 初始化文本分割器：文档切分，用于上传文档后拆分成小块文本
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,                     # 每个文本块最大字符数
    chunk_overlap=80,                   # 块之间重叠字符，保证上下文连续性
    separators=['\n\n','\n','。','，']  # 中文优先分隔符，按换行、句号、逗号切分
)

# 配置Jinja2模板目录，templates文件夹存放html页面
templates = Jinja2Templates(directory='templates')


# 首页路由，返回首页html页面
@app.get('/')
def home(request: Request):
    return templates.TemplateResponse(request,'home.html')


# 问答页面路由，返回问答交互html页面
@app.get('/question')
def question(request: Request):
    return templates.TemplateResponse(request,'question.html')


# 文档上传接口：接收文本文件，切分后向量化存入Chroma向量库
@app.post('/upload')
async def upload(file: UploadFile = File(...)):
    # 读取上传文件二进制内容
    file_01 = await file.read()
    # 二进制转utf‑8文本，遇到无法解析字符直接忽略
    text = file_01.decode('utf-8', 'ignore')
    # 使用分割器把完整文档切分为多个文本块
    texts = splitter.split_text(text)
    uids = []
    # 遍历文本块，生成每个块的唯一UUID标识
    for i in texts:
        uid = str(uuid.uuid4())
        uids.append(uid)
    # 将文本块写入向量数据库，内部自动执行Embedding向量化
    collects.add(ids=uids, documents=texts)
    return 'yes'


# 问答核心接口：优先走Redis缓存，无缓存则向量检索+调用大模型生成回答
@app.post('/answer')
async def answer(text: Question):
    redis_result = None
    # 尝试从Redis读取该问题的缓存结果
    try:
        redis_result = await redis_client.get(text.question)
    except Exception:
        redis_result = None # redis异常时降级，直接调用大模型

    # 如果缓存命中，直接返回缓存结果
    if redis_result:
        result_2 = redis_result
        source = 'redis缓存'
    else:
        # 向量数据库执行语义检索，取出相似度最高2条文档片段
        result = collects.query(query_texts=text.question, n_results=2)
        # 将检索到的文档片段拼接作为参考上下文
        result_1 = '\n'.join(result['documents'][0])
        # 构造RAG提示词，约束模型只能基于参考资料回答，减少幻觉
        prompt = f"""
        参考下面资料回答用户问题，不要编造资料以外的内容。
        参考资料：
        {result_1}
        用户问题：{text.question}
        """
        # 请求智谱大模型接口获取回答
        resp = openai_zhipu.chat.completions.create(
            model=text.model,
            messages=[{"content": prompt, "role": "user"}],
            temperature=0.1 # temperature越低，生成结果越确定，创造性越低
        )
        # 提取大模型返回内容
        result_2 = resp.choices[0].message.content
        # 将问答结果写入Redis缓存，有效期60秒
        await redis_client.setex(text.question, 60, result_2)
        source = 'AI大模型生成'

    # 数据库记录问答日志：问题、回答入库
    db = Session(engine)
    db.add(QuestionLog(question=text.question, result=result_2))
    db.commit() # 提交事务保存数据
    db.close()  # 关闭数据库会话

    # 返回接口JSON数据
    return {"code": 200, "question": text.question, "result": result_2, "source": source}


# ASGI服务启动入口
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8080)

