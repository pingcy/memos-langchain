"""
🤖 Chatbot with MemOS V3 - 基于 LangChain 的长期记忆聊天机器人
============================================================

功能演示:
1. 使用 LangChain ChatOpenAI 作为对话模型
2. 使用 MemOS SingleCubeView 实现长期记忆 (V2 API 风格)
3. 增量式记忆管理（只添加新的对话，不重复添加历史）
4. 基于记忆的个性化对话

依赖:
- pip install langchain langchain-openai
- Qdrant (本地文件存储，无需额外服务)
- OpenAI API

V3 变更说明:
- 使用 init_server + SingleCubeView 替代原有的 MOS + GeneralMemCube
- 使用 APIADDRequest / APISearchRequest 进行记忆操作
"""

import warnings
import logging
import os
from datetime import datetime
from typing import List, Dict, Optional

# 抑制警告
warnings.filterwarnings("ignore", message="Pydantic serializer warnings")
warnings.filterwarnings("ignore", message="`torch_dtype` is deprecated")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*Neo4j.*")
warnings.filterwarnings("ignore", message=".*relationship type.*")
warnings.filterwarnings("ignore", message=".*PARENT.*")

# 设置日志级别 - 抑制 memos 内部的调试输出
logging.basicConfig(level=logging.ERROR, format='%(message)s')  # 只显示 ERROR 以上
logging.getLogger("memos").setLevel(logging.ERROR)
logging.getLogger("memos.api.config").setLevel(logging.CRITICAL)
logging.getLogger("memos.mem_cube").setLevel(logging.CRITICAL)  # 抑制 mem_cube 警告
logging.getLogger("memos.mem_cube.general").setLevel(logging.CRITICAL)  # 抑制 pref_mem 警告
logging.getLogger("neo4j").setLevel(logging.CRITICAL)  # 完全抑制 Neo4j 日志
logging.getLogger("neo4j.notifications").setLevel(logging.CRITICAL)
logging.getLogger("neo4j.io").setLevel(logging.CRITICAL)
logging.getLogger("httpx").setLevel(logging.ERROR)

# 抑制 memos 内部的 trace-id 格式警告（需要在 import memos 之前设置）
import memos.settings
memos.settings.DEBUG = False
# 设置 memos 的根日志级别为 ERROR
logging.getLogger("memos").setLevel(logging.ERROR)

from dotenv import load_dotenv
load_dotenv()

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# MemOS V2 API imports
from memos.api.handlers import init_server
from memos.log import get_logger
from memos.multi_mem_cube.single_cube import SingleCubeView
from memos.api.product_models import APIADDRequest, APISearchRequest

# 再次设置日志级别（确保在 memos 模块导入后生效）
logging.getLogger("memos").setLevel(logging.ERROR)
logging.getLogger("memos.mem_cube.general").setLevel(logging.ERROR)
logging.getLogger("memos.api.config").setLevel(logging.ERROR)

# 获取 logger
logger = get_logger(__name__)


# ==================== 记忆系统 Prompt 模板 ====================
MEMORY_SYSTEM_PROMPT = """# Role
你是一个拥有长期记忆能力的旅游规划助手 (Travel Assistant)。
你的目标是结合检索到的记忆片段，为用户提供高度个性化且逻辑严密的回答。
你应该在每次给出建议前尽量了解用户的信息、偏好并及时调整。
尽量简短的回答问题。

# System Context
- 当前时间: {current_time}

# Memory Data
以下是检索到的与用户相关的记忆信息：

<memories>
{memories}
</memories>

# Instructions
1. 结合记忆中的信息为用户提供个性化回答
2. 如果记忆与当前问题无关，可以忽略
3. 直接回答问题，不要提及"记忆"、"检索"等系统内部术语
4. 如果记忆中没有相关信息，正常回答即可"""


class MemOSChatbot:
    """基于 LangChain 和 MemOS 的长期记忆聊天机器人 (V3 - 使用 V2 API)
    
    使用 SingleCubeView API 进行记忆操作，更简洁的接口设计。
    """
    
    def __init__(
        self,
        user_id: str = "chatbot_user",
        cube_id: str = None,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.7,
        top_k: int = 5
    ):
        """
        初始化 Chatbot
        
        Args:
            user_id: 用户ID，用于区分不同用户的记忆
            cube_id: MemCube ID，用于区分不同的记忆空间
            model_name: OpenAI 模型名称
            temperature: 生成温度
            top_k: 记忆检索数量
        """
        self.user_id = user_id
        self.cube_id = cube_id or f"{user_id}_chatbot_cube"
        self.top_k = top_k
        
        # 获取 API 配置
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.openai_base = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
        
        if not self.openai_key:
            raise ValueError("❌ 未配置 OPENAI_API_KEY，请在 .env 文件中设置")
        
        # 初始化 LangChain ChatOpenAI
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            openai_api_key=self.openai_key,
            openai_api_base=self.openai_base
        )
        
        # 初始化 MemOS（使用 V2 API）
        self.is_new_user = self._init_memos()
        
        # 当前会话的对话历史（用于上下文）
        self.conversation_history: List[Dict[str, str]] = []
        
        # 已经添加到记忆库的消息数（用于增量添加）
        self.memorized_message_count = 0
        
        # 会话ID（用于追踪）
        self.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        print(f"✅ Chatbot 初始化完成")
        print(f"   - 用户ID: {self.user_id}")
        print(f"   - Cube ID: {self.cube_id}")
        print(f"   - 会话ID: {self.session_id}")
        print(f"   - 模型: {model_name}")
        print(f"   - 是否新用户: {'是' if self.is_new_user else '否（已有历史记忆）'}")
    
    def _init_memos(self) -> bool:
        """
        初始化 MemOS 记忆系统 (使用 V2 API)
        
        使用 init_server + SingleCubeView 方式初始化。
        
        Returns:
            bool: True 表示是新用户（无历史记忆），False 表示已有历史记忆
        """
        print("📦 初始化 MemOS 记忆系统 (V2 API)...")
        
        # 初始化服务器组件
        self.components = init_server()
        
        # 创建 SingleCubeView
        self.cube = SingleCubeView(
            cube_id=self.cube_id,
            naive_mem_cube=self.components["naive_mem_cube"],
            mem_reader=self.components["mem_reader"],
            mem_scheduler=self.components["mem_scheduler"],
            logger=logger,
            searcher=self.components["searcher"],
        )
        
        # 设置可写入和可读取的 cube_id 列表
        self.writable_cube_ids = [self.cube_id]
        self.readable_cube_ids = [self.cube_id]
        
        # 检查是否有历史记忆
        is_new_user = True
        try:
            # 尝试搜索一条记忆来判断是否有历史
            test_results = self.cube.search_memories(
                APISearchRequest(
                    user_id=self.user_id,
                    readable_cube_ids=self.readable_cube_ids,
                    query="用户信息",
                )
            )
            
            if test_results.get("text_mem") and test_results["text_mem"][0].get("memories"):
                memory_count = len(test_results["text_mem"][0]["memories"])
                if memory_count > 0:
                    print(f"   ✅ 发现历史记忆")
                    is_new_user = False
                else:
                    print(f"   🆕 未发现历史记忆")
            else:
                print(f"   🆕 未发现历史记忆")
        except Exception as e:
            print(f"   🆕 未发现历史记忆 (检查时出错: {e})")
        
        return is_new_user
    
    def _search_memories(self, query: str) -> List[str]:
        """
        根据查询检索相关记忆 (使用 V2 API)
        
        Args:
            query: 查询文本
            
        Returns:
            记忆列表（字符串格式）
        """
        memories = []
        
        try:
            results = self.cube.search_memories(
                APISearchRequest(
                    user_id=self.user_id,
                    readable_cube_ids=self.readable_cube_ids,
                    query=query,
                )
            )
            
            if results.get("text_mem") and results["text_mem"][0].get("memories"):
                for mem_item in results["text_mem"][0]["memories"][:self.top_k]:
                    # V2 API 返回的是字典格式
                    memory_text = mem_item.get('memory', '') if isinstance(mem_item, dict) else mem_item.memory
                    if memory_text:
                        memories.append(memory_text)
                    
        except Exception as e:
            print(f"   ⚠️ 检索记忆时出错: {e}")
        
        return memories
    
    def _format_memories(self, memories: List[str]) -> str:
        """将记忆列表格式化为字符串"""
        if not memories:
            return "暂无相关记忆"
        
        formatted = []
        for i, mem in enumerate(memories, 1):
            formatted.append(f"[{i}] {mem}")
        
        return "\n".join(formatted)
    
    def _add_memories_incrementally(self):
        """增量式添加记忆到 MemOS (使用 V2 API)"""
        current_count = len(self.conversation_history)
        
        if current_count <= self.memorized_message_count:
            return
        
        new_messages = self.conversation_history[self.memorized_message_count:]
        
        if len(new_messages) >= 2:
            try:
                add_req = APIADDRequest(
                    user_id=self.user_id,
                    messages=new_messages,
                    writable_cube_ids=self.writable_cube_ids,
                    async_mode='sync'
                )
                self.cube.add_memories(add_req)
                self.memorized_message_count = current_count
                print(f"   💾 已将 {len(new_messages)} 条新消息添加到记忆")
            except Exception as e:
                print(f"   ⚠️ 添加记忆时出错: {e}")
    
    def chat(self, user_input: str) -> str:
        """
        与 Chatbot 对话
        
        流程：
        1. 检索相关记忆
        2. 构建带记忆上下文的 prompt
        3. 调用 LLM 生成回答
        4. 增量添加对话到记忆
        
        Args:
            user_input: 用户输入
            
        Returns:
            助手回复
        """
        # 1. 检索相关记忆
        memories = self._search_memories(user_input)
        formatted_memories = self._format_memories(memories)
        
        if memories:
            print(f"   🔍 检索到 {len(memories)} 条相关记忆")
        
        # 2. 构建 prompt
        system_prompt = MEMORY_SYSTEM_PROMPT.format(
            current_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            memories=formatted_memories
        )
        
        # 构建消息列表（包含最近几轮对话作为上下文）
        messages = [SystemMessage(content=system_prompt)]
        
        # 添加最近的对话历史作为上下文
        recent_history = self.conversation_history[-10:]  # 最近 5 轮
        for msg in recent_history:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            else:
                messages.append(AIMessage(content=msg["content"]))
        
        # 添加当前用户输入
        messages.append(HumanMessage(content=user_input))
        
        # 3. 调用 LLM
        response = self.llm.invoke(messages)
        assistant_reply = response.content
        
        # 4. 添加到对话历史
        self.conversation_history.append({"role": "user", "content": user_input})
        self.conversation_history.append({"role": "assistant", "content": assistant_reply})
        
        # 5. 每 4 条消息增量添加记忆
        if len(self.conversation_history) % 4 == 0:
            self._add_memories_incrementally()
        
        return assistant_reply
    
    def force_memorize(self):
        """强制将所有未记忆的对话添加到记忆库"""
        print("🔄 强制同步记忆...")
        self._add_memories_incrementally()
    
    def get_memory_count(self) -> int:
        """获取记忆数量（通过搜索估算）"""
        try:
            results = self.cube.search_memories(
                APISearchRequest(
                    user_id=self.user_id,
                    readable_cube_ids=self.readable_cube_ids,
                    query="*",  # 通配符查询
                )
            )
            
            if results.get("text_mem") and results["text_mem"][0].get("memories"):
                return len(results["text_mem"][0]["memories"])
            return 0
        except:
            return 0
    
    def show_memories(self, limit: int = 20):
        """显示所有记忆"""
        try:
            results = self.cube.search_memories(
                APISearchRequest(
                    user_id=self.user_id,
                    readable_cube_ids=self.readable_cube_ids,
                    query="用户信息 偏好 历史",  # 宽泛查询获取更多记忆
                )
            )
            
            if results.get("text_mem") and results["text_mem"][0].get("memories"):
                memories = results["text_mem"][0]["memories"][:limit]
                print(f"\n📚 检索到 {len(memories)} 条记忆:")
                for i, mem_item in enumerate(memories, 1):
                    if isinstance(mem_item, dict):
                        mem_text = mem_item.get('memory', '')[:80]
                        mem_type = mem_item.get('metadata', {}).get('memory_type', 'Unknown') if isinstance(mem_item.get('metadata'), dict) else 'Unknown'
                    else:
                        mem_text = str(mem_item)[:80]
                        mem_type = 'Unknown'
                    print(f"  [{i}] [{mem_type}] {mem_text}...")
            else:
                print("\n📚 暂无记忆")
        except Exception as e:
            print(f"❌ 获取记忆失败: {e}")
    
    def clear_memories(self):
        """清空所有记忆（重置会话）"""
        try:
            
            self.conversation_history = []
            self.memorized_message_count = 0
            self.cube.naive_mem_cube.text_mem.delete_all()
            print("✅ 记忆已清空")
        except Exception as e:
            print(f"❌ 清空记忆失败: {e}")
    
    def close(self):
        """
        关闭 Chatbot，释放资源
        
        在程序退出前应调用此方法，否则可能导致程序无法正常退出
        """
        try:
            # 关闭相关组件
            if hasattr(self, 'components'):
                # 尝试关闭可能存在的连接
                if 'naive_mem_cube' in self.components:
                    naive_cube = self.components['naive_mem_cube']
                    if hasattr(naive_cube, 'close'):
                        naive_cube.close()
            
            print("✅ Chatbot 资源已释放")
        except Exception as e:
            print(f"⚠️ 关闭 Chatbot 资源时出错: {e}")


def main():
    """主函数 - 交互式对话"""
    print("=" * 60)
    print("🤖 MemOS Chatbot V3 - 拥有长期记忆的聊天机器人")
    print("   (使用 V2 API: SingleCubeView)")
    print("=" * 60)
    
    # 初始化 Chatbot
    chatbot = MemOSChatbot(user_id="demo_user")
    
    print(f"\n📊 当前记忆数量: {chatbot.get_memory_count()}")
    
    # 交互式对话
    print("\n进入交互模式 (输入命令):")
    print("  - 'quit'/'exit': 退出")
    print("  - '/memory': 显示当前记忆")
    print("  - '/clear': 清空记忆")
    print("  - '/save': 强制保存记忆")
    print()
    
    while True:
        try:
            user_input = input("👤 [You] ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n👋 Goodbye!")
            break
        
        # 处理特殊命令
        if user_input.lower() in {"quit", "exit", "q"}:
            # 退出前保存记忆并关闭资源
            chatbot.force_memorize()
            print("💾 记忆已保存")
            chatbot.close()
            print("👋 Goodbye!")
            break
        
        if user_input == "/memory":
            chatbot.show_memories()
            continue
        
        if user_input == "/clear":
            chatbot.clear_memories()
            continue
        
        if user_input == "/save":
            chatbot.force_memorize()
            print("💾 记忆已保存")
            continue
        
        if not user_input:
            continue
        
        # 对话
        response = chatbot.chat(user_input)
        print(f"🤖 [Assistant] {response}\n")


if __name__ == "__main__":
    main()
