"""
LangChain 1.0 智能体示例 - 带 MemOS 长期记忆
============================================

功能：
1. 使用 create_agent + Tavily 搜索工具进行网络搜索
2. 使用 MemOS 中间件实现长期记忆能力
3. 智能体可以记住用户的偏好、历史对话等信息

依赖：
- langchain, langchain-tavily
- memos (MemOS)
- Neo4j 数据库（用于树形记忆存储）
"""

import os
import warnings
import logging

# 抑制警告 - 必须在导入 memos 模块之前设置
warnings.filterwarnings("ignore", message="Pydantic serializer warnings")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message="Field name")  # 抑制 langchain_tavily 警告

# 抑制 MemOS 相关的警告日志 - 必须在导入 memos 之前设置
logging.basicConfig(level=logging.INFO)
logging.getLogger("memos").setLevel(logging.ERROR)
logging.getLogger("memos.mem_cube.general").setLevel(logging.ERROR)
logging.getLogger("memos.api.config").setLevel(logging.ERROR)
logging.getLogger("neo4j").setLevel(logging.ERROR)

from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 导入 LangChain 1.0 的 create_agent
from langchain.agents import create_agent
from langchain_tavily import TavilySearch

# 导入 MemOS 中间件
from memos_middleware import create_memos_middleware

# 确保设置了必要的环境变量
# OPENAI_API_KEY - OpenAI API 密钥
# TAVILY_API_KEY - Tavily 搜索 API 密钥
# NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD - Neo4j 配置

# 创建 Tavily 搜索工具
tavily_search = TavilySearch(
    max_results=3,      # 最多返回 3 条搜索结果
    search_depth="advanced"       # 使用高级搜索模式
)

# 创建 MemOS 记忆中间件
# 这将使智能体具备长期记忆能力
memos_middleware = create_memos_middleware(
    user_id="langchain_agent_user",  # 用户ID，用于隔离不同用户的记忆
    top_k=5,                          # 每次检索返回的记忆数量
    auto_memorize=True,               # 自动将对话添加到记忆
    verbose=True,                     # 打印详细日志
)

# 创建带记忆能力的智能体
agent = create_agent(
    model="gpt-4o-mini",  # 使用 GPT-4o-mini 模型
    tools=[tavily_search],
    system_prompt="""你是一个拥有长期记忆能力的智能助手。

## 回答策略

**必须使用搜索工具的情况**：
- 用户使用"搜索"、"查一下"、"查询"等词时
- 需要最新的时事新闻、体育赛事、股票、天气等实时信息
- 用户询问近期发生的事件（如"昨天"、"今天"、"最近"）

**优先使用记忆的情况**：
- 用户询问他们自己的偏好、历史对话
- 系统注入的"长期记忆上下文"中有相关信息

**使用自身知识**：
- 通用知识问答
- 记忆中没有、且不需要实时信息的问题

## 重要提示
- 当用户明确要求搜索时，**必须调用搜索工具**，不要拒绝
- 请用中文回答问题""",
    middleware=[memos_middleware],  # 注入 MemOS 中间件
)

# 运行智能体
if __name__ == "__main__":
    print("=" * 60)
    print("LangChain 1.0 智能体 - Tavily 搜索 + MemOS 长期记忆")
    print("=" * 60)
    
    # 显示当前记忆数量
    print(f"\n📊 当前记忆数量: {memos_middleware.get_memory_count()}")
    
    # 交互式对话
    print("\n进入交互模式 (输入命令):")
    print("  - 'quit'/'exit': 退出")
    print("  - '/memory': 显示当前记忆")
    print("  - '/clear': 清空记忆")
    print("  - '/save': 强制保存记忆")
    print()
    
    while True:
        user_input = input("👤 [You] ").strip()
        
        # 处理特殊命令
        if user_input.lower() in {"quit", "exit", "q"}:
            # 退出前保存记忆并关闭资源
            memos_middleware.force_memorize()
            print("💾 记忆已保存")
            memos_middleware.close()
            print("👋 Goodbye!")
            break
        
        if user_input == "/memory":
            memos_middleware.show_memories()
            continue
        
        if user_input == "/clear":
            memos_middleware.clear_memories()
            continue
        
        if user_input == "/save":
            memos_middleware.force_memorize()
            print("💾 记忆已保存")
            continue
        
        if not user_input:
            continue
        
        # 调用智能体
        response = agent.invoke(
            {"messages": [{"role": "user", "content": user_input}]}
        )
        
        # 输出回答
        if "messages" in response:
            last_message = response["messages"][-1]
            if hasattr(last_message, "content"):
                print(f"🤖 [Assistant] {last_message.content}\n")
            else:
                print(f"🤖 [Assistant] {last_message}\n")
        else:
            print(f"🤖 [Assistant] {response}\n")
