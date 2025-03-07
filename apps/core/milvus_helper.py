"""
Milvus 向量数据库操作助手

功能描述：
1. 向量模型管理
   - 使用 BAAI/bge-m3 模型进行文本向量化
   - 采用单例模式管理模型实例

2. Milvus集合管理
   - 创建和初始化向量数据库集合
   - 集合字段：
     * id: 主键，自增长整数
     * text: 测试用例文本，最大长度10000字符
     * embedding: 1024维向量，存储文本的向量表示

3. Excel数据处理
   - 读取测试用例Excel文件
   - 提取关键字段并组织成结构化文本
   - 支持的字段：用例名称、ID、前置条件、所属模块、步骤描述、预期结果等
   - 数据验证：确保必要字段（用例名称、步骤描述、预期结果）不为空

注意事项：
- 需要本地运行Milvus服务（默认端口19530）
- Excel文件必须包含指定的列名
- 向量维度固定为1024，与选用的模型对应
"""

import pandas as pd
from pymilvus import connections, Collection, DataType, utility, FieldSchema, CollectionSchema
from sentence_transformers import SentenceTransformer

# 初始化嵌入模型（单例模式）
_embedding_model = None

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer("BAAI/bge-m3", trust_remote_code=True)
    return _embedding_model

# 初始化Milvus集合
def init_milvus_collection(collection_name="test_cases"):
    """初始化Milvus集合"""
    try:
        # 连接到Milvus服务器
        connections.connect(host="localhost", port="19530")
        
        # 检查集合是否存在
        if utility.has_collection(collection_name):
            return Collection(name=collection_name)
        
        # 如果集合不存在，创建新集合
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=10000),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1024)
        ]
        schema = CollectionSchema(fields=fields, description="测试用例知识库")
        collection = Collection(name=collection_name, schema=schema)
        
        # 创建索引
        index_params = {
            "index_type": "IVF_FLAT",
            "metric_type": "IP",
            "params": {"nlist": 128}
        }
        collection.create_index(field_name="embedding", index_params=index_params)
        collection.load()
        
        return collection
        
    except Exception as e:
        raise Exception(f"初始化Milvus集合失败: {str(e)}")

# 处理Excel文件
def process_excel(file_path):
    try:
        df = pd.read_excel(file_path, engine='openpyxl')
        test_cases = []
        for _, row in df.iterrows():
            # 构建更完整的测试用例文本，包含更多相关信息
            text = (
                f"用例名称：{row['用例名称']}\n"
                f"ID：{row['ID']}\n"
                f"前置条件：{row['前置条件']}\n"
                f"所属模块：{row['所属模块']}\n"
                f"步骤描述：{row['步骤描述']}\n"
                f"预期结果：{row['预期结果']}\n"
                f"标签：{row['标签']}\n"
                f"用例等级：{row['用例等级']}\n"
                f"创建人：{row['创建人']}"
            )
            # 只有当关键字段不为空时才添加到测试用例列表中
            if row['用例名称'] and row['步骤描述'] and row['预期结果']:
                test_cases.append(text.strip())
        
        # 返回非空的测试用例列表
        return [case for case in test_cases if case]
    except KeyError as e:
        # 处理列名不存在的错误
        raise ValueError(f"Excel文件缺少必要的列: {str(e)}")
    except Exception as e:
        # 处理其他错误
        raise ValueError(f"Excel解析失败: {str(e)}")