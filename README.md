# memos-langchain

🧠 **将 MemOS 长期记忆能力集成到 LangChain 智能体的示例项目**

本项目演示如何使用 [MemOS](https://github.com/MemTensor/MemOS) 为 LangChain 智能体添加长期记忆能力，让 AI 助手能够记住用户的偏好、历史对话等信息。

## ✨ 功能特点

- 🤖 **Chatbot with Memory**: 基于 LangChain 的聊天机器人，使用 MemOS 实现长期记忆
- 🔧 **LangChain Agent + Memory**: 将 MemOS 作为中间件集成到 LangChain 1.x 智能体
- 🌲 **树形记忆存储**: 使用 Neo4j 作为图数据库后端，支持层次化记忆组织
- 🔍 **智能检索**: 基于语义相似度检索相关记忆

## 📁 项目结构

```
memos-langchain/
├── chatbot_with_memos.py      # 演示：带长期记忆的聊天机器人
├── simple_langchain_agent.py  # 演示：LangChain 1.x 智能体 + Tavily 搜索 + 记忆
├── memos_memory_helper.py     # MemOS 记忆助手封装
├── memos_middleware.py        # LangChain 中间件实现
├── requirements.txt           # Python 依赖
├── .env.example               # 环境变量示例
└── README.md
```

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

复制 `.env.example` 为 `.env` 并填入你的 API 密钥：

```bash
cp .env.example .env
```

编辑 `.env` 文件：

```env
# OpenAI API (必需)
OPENAI_API_KEY=your-openai-api-key
OPENAI_API_BASE=https://api.openai.com/v1

# Neo4j 数据库 (必需)
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your-neo4j-password

# Tavily 搜索 (仅 simple_langchain_agent.py 需要)
TAVILY_API_KEY=your-tavily-api-key
```

### 3. 启动 Neo4j 数据库

你可以使用 Docker 快速启动 Neo4j：

```bash
docker run -d \
  --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/yourpassword \
  neo4j:latest
```

### 4. 运行示例

**运行聊天机器人：**

```bash
python chatbot_with_memos.py
```

**运行 LangChain 智能体（带搜索功能）：**

```bash
python simple_langchain_agent.py
```

## 📖 使用说明

### 交互命令

在运行示例时，你可以使用以下命令：

| 命令 | 说明 |
|------|------|
| `/memory` | 显示当前所有记忆 |
| `/clear` | 清空所有记忆 |
| `/save` | 强制保存记忆 |
| `quit` / `exit` | 退出程序 |

### 示例对话

```
👤 [You] 我叫张三，我喜欢踢足球
🤖 [Assistant] 你好张三！踢足球是很好的运动...

👤 [You] 我最喜欢的球星是梅西
🤖 [Assistant] 梅西确实是一位传奇球员...

# 下次对话时，AI 会记住你的信息：
👤 [You] 你还记得我的爱好吗？
🤖 [Assistant] 当然记得！你喜欢踢足球，而且你最喜欢的球星是梅西。
```

## 🔧 自定义配置

### 在你的项目中使用记忆助手

```python
from memos_memory_helper import MemosMemoryHelper

# 创建记忆助手
helper = MemosMemoryHelper(
    user_id="my_user",
    top_k=5,  # 检索时返回的记忆数量
)

# 添加对话
helper.add_conversation("用户消息", "助手回复")

# 检索相关记忆
memories = helper.search_memories("查询内容")

# 强制保存记忆
helper.force_memorize()

# 关闭资源
helper.close()
```

### 在 LangChain 智能体中使用中间件

```python
from langchain.agents import create_agent
from memos_middleware import create_memos_middleware

# 创建记忆中间件
memos_middleware = create_memos_middleware(
    user_id="agent_user",
    top_k=5,
    auto_memorize=True,
)

# 创建智能体
agent = create_agent(
    model="gpt-4o-mini",
    tools=[...],
    middleware=[memos_middleware],
)
```

## 📦 依赖说明

| 依赖 | 版本 | 说明 |
|------|------|------|
| `memos` | latest | MemOS SDK - 记忆管理系统 |
| `langchain` | >=0.3.0 | LangChain 框架 |
| `langchain-openai` | latest | LangChain OpenAI 集成 |
| `langchain-tavily` | latest | Tavily 搜索工具 |
| `langgraph` | latest | LangGraph 运行时 |
| `python-dotenv` | latest | 环境变量管理 |
| `neo4j` | latest | Neo4j Python 驱动 |

## 📄 License

MIT License

## 🔗 相关链接

- [MemOS 官方仓库](https://github.com/MemTensor/MemOS)
- [LangChain 文档](https://python.langchain.com/)
- [Neo4j 官网](https://neo4j.com/)
