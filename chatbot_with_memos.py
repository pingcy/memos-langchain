"""
🤖 Chatbot with MemOS - 基于 LangChain 的长期记忆聊天机器人
============================================================

功能演示:
1. 使用 LangChain ChatOpenAI 作为对话模型
2. 使用 MemOS GeneralTextMemory 实现长期记忆
3. 增量式记忆管理（只添加新的对话，不重复添加历史）
4. 基于记忆的个性化对话

依赖:
- pip install langchain langchain-openai
- Qdrant (本地文件存储，无需额外服务)
- OpenAI API
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

# MemOS imports
from memos.configs.mem_os import MemOSConfigFactory
from memos.mem_os.main import MOS
from memos.mem_cube.general import GeneralMemCube
from memos.configs.mem_cube import GeneralMemCubeConfig

# 再次设置日志级别（确保在 memos 模块导入后生效）
logging.getLogger("memos").setLevel(logging.ERROR)
logging.getLogger("memos.mem_cube.general").setLevel(logging.ERROR)
logging.getLogger("memos.api.config").setLevel(logging.ERROR)


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
    """基于 LangChain 和 MemOS 的长期记忆聊天机器人
    
    使用 tree_text (Neo4j) 后端，记忆数据直接存储在 Neo4j 数据库中，
    无需手动 dump/load。
    """
    
    def __init__(
        self,
        user_id: str = "chatbot_user",
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.7,
        top_k: int = 5
    ):
        """
        初始化 Chatbot
        
        Args:
            user_id: 用户ID，用于区分不同用户的记忆
            model_name: OpenAI 模型名称
            temperature: 生成温度
            top_k: 记忆检索数量
        """
        self.user_id = user_id
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
        
        # 初始化 MemOS（会尝试加载已有记忆）
        self.is_new_user = self._init_memos()
        
        # 当前会话的对话历史（用于上下文）
        self.conversation_history: List[Dict[str, str]] = []
        
        # 已经添加到记忆库的消息数（用于增量添加）
        self.memorized_message_count = 0
        
        # 会话ID（用于追踪）
        self.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        print(f"✅ Chatbot 初始化完成")
        print(f"   - 用户ID: {self.user_id}")
        print(f"   - 会话ID: {self.session_id}")
        print(f"   - 模型: {model_name}")
        print(f"   - 是否新用户: {'是' if self.is_new_user else '否（已有历史记忆）'}")
    
    def _init_memos(self) -> bool:
        """
        初始化 MemOS 记忆系统
        
        对于 tree_text (Neo4j) 后端，数据直接存在数据库中，不需要 dump/load。
        只需要查询 Neo4j 中是否有该用户的记忆来判断是否新用户。
        
        Returns:
            bool: True 表示是新用户（无历史记忆），False 表示已有历史记忆
        """
        print("📦 初始化 MemOS 记忆系统...")
        
        # 配置 MOS
        mos_config = MemOSConfigFactory(
            config={
                "user_id": "root",
                "chat_model": {
                    "backend": "openai",
                    "config": {
                        "model_name_or_path": "gpt-4o-mini",
                        "temperature": 0.0,
                        "max_tokens": 8192,
                        "api_key": self.openai_key,
                        "api_base": self.openai_base
                    }
                },
                "mem_reader": {
                    "backend": "simple_struct",
                    "config": {
                        "llm": {
                            "backend": "openai",
                            "config": {
                                "model_name_or_path": "gpt-4o-mini",
                                "temperature": 0.0,
                                "max_tokens": 8192,
                                "api_key": self.openai_key,
                                "api_base": self.openai_base
                            }
                        },
                        "embedder": {
                            "backend": "universal_api",
                            "config": {
                                "provider": "openai",
                                "model_name_or_path": "text-embedding-3-small",
                                "api_key": self.openai_key,
                                "base_url": self.openai_base
                            }
                        },
                        "chunker": {
                            "backend": "sentence",
                            "config": {
                                "tokenizer_or_token_counter": "character",
                                "chunk_size": 512,
                                "chunk_overlap": 128,
                                "min_sentences_per_chunk": 1
                            }
                        }
                    }
                },
                "max_turns_window": 20,
                "top_k": self.top_k,
                "enable_textual_memory": True,
                "enable_activation_memory": False,
                "enable_parametric_memory": False,
                "enable_mem_scheduler": False
            }
        )
        
        self.mos = MOS(mos_config.config)
        self.mos.create_user(user_id=self.user_id)
        
        # cube_id 固定
        self.cube_id = f"{self.user_id}_chatbot_cube"
        
        # Neo4j 配置
        neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        neo4j_user = os.getenv("NEO4J_USER", "neo4j")
        neo4j_password = os.getenv("NEO4J_PASSWORD", "yourpassword")
        
        print(f"   📡 连接 Neo4j: {neo4j_uri}")
        
        # 创建 MemCube 连接到 Neo4j (tree_text 后端)
        mem_cube_config = GeneralMemCubeConfig(
            user_id=self.user_id,
            cube_id=self.cube_id,
            text_mem={
                "backend": "tree_text",
                "config": {
                    "extractor_llm": {
                        "backend": "openai",
                        "config": {
                            "model_name_or_path": "gpt-4o-mini",
                            "temperature": 0.0,
                            "max_tokens": 8192,
                            "api_key": self.openai_key,
                            "api_base": self.openai_base
                        }
                    },
                    "dispatcher_llm": {
                        "backend": "openai",
                        "config": {
                            "model_name_or_path": "gpt-4o-mini",
                            "temperature": 0.0,
                            "max_tokens": 8192,
                            "api_key": self.openai_key,
                            "api_base": self.openai_base
                        }
                    },
                    "embedder": {
                        "backend": "universal_api",
                        "config": {
                            "provider": "openai",
                            "model_name_or_path": "text-embedding-3-small",
                            "api_key": self.openai_key,
                            "base_url": self.openai_base
                        }
                    },
                    "graph_db": {
                        "backend": "neo4j",
                        "config": {
                            "uri": neo4j_uri,
                            "user": neo4j_user,
                            "password": neo4j_password,
                            "db_name": "memos",
                            "embedding_dimension": 1536
                        }
                    },
                    "reorganize": True
                }
            },
            act_mem={"backend": "uninitialized"},
            para_mem={"backend": "uninitialized"}
        )
        
        self.mem_cube = GeneralMemCube(mem_cube_config)
        self.mos.register_mem_cube(
            mem_cube_name_or_path=self.mem_cube,
            mem_cube_id=self.cube_id,
            user_id=self.user_id
        )
        
        # 检查是否有历史记忆
        is_new_user = True
        try:
            # 尝试获取记忆数量
            all_memories = self.mem_cube.text_mem.get_all(user_name=self.user_id)
            if isinstance(all_memories, dict) and 'nodes' in all_memories:
                memory_count = len(all_memories.get('nodes', []))
            else:
                memory_count = len(all_memories)
            
            if memory_count > 0:
                print(f"   ✅ 发现 {memory_count} 条历史记忆")
                is_new_user = False
            else:
                print(f"   🆕 未发现历史记忆")
        except:
            print(f"   🆕 未发现历史记忆")
        
        return is_new_user
    
    def _search_memories(self, query: str) -> List[str]:
        """
        根据查询检索相关记忆
        
        Args:
            query: 查询文本
            
        Returns:
            记忆列表（字符串格式）
        """
        memories = []
        
        try:
            results = self.mos.search(query=query, user_id=self.user_id)
            
            if results.get("text_mem") and results["text_mem"][0]["memories"]:
                for mem_item in results["text_mem"][0]["memories"][:self.top_k]:
                    memories.append(mem_item.memory)
                    
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
        """增量式添加记忆到 MemOS"""
        current_count = len(self.conversation_history)
        
        if current_count <= self.memorized_message_count:
            return
        
        new_messages = self.conversation_history[self.memorized_message_count:]
        
        if len(new_messages) >= 2:
            try:
                self.mos.add(
                    messages=new_messages,
                    user_id=self.user_id,
                    mem_cube_id=self.cube_id
                )
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
        """获取记忆数量"""
        try:
            all_memories = self.mem_cube.text_mem.get_all(user_name=self.user_id)
            if isinstance(all_memories, dict) and 'nodes' in all_memories:
                return len(all_memories.get('nodes', []))
            return len(all_memories)
        except:
            return 0
    
    def show_memories(self, limit: int = 20):
        """显示所有记忆"""
        try:
            all_memories = self.mem_cube.text_mem.get_all(user_name=self.user_id)
            
            if isinstance(all_memories, dict) and 'nodes' in all_memories:
                nodes = all_memories.get('nodes', [])[:limit]
                print(f"\n📚 当前共有 {len(nodes)} 条记忆:")
                for i, node in enumerate(nodes, 1):
                    mem_text = node.get('memory', '')[:80]
                    mem_type = node.get('metadata', {}).get('memory_type', 'Unknown')
                    print(f"  [{i}] [{mem_type}] {mem_text}...")
            else:
                print(f"\n📚 当前共有 {len(all_memories)} 条记忆:")
                for i, mem_item in enumerate(all_memories[:limit], 1):
                    print(f"  [{i}] {mem_item.memory[:80]}...")
        except Exception as e:
            print(f"❌ 获取记忆失败: {e}")
    
    def clear_memories(self):
        """清空所有记忆"""
        try:
            self.mem_cube.text_mem.delete_all()
            self.conversation_history = []
            self.memorized_message_count = 0
            print("✅ 记忆已清空")
        except Exception as e:
            print(f"❌ 清空记忆失败: {e}")
    
    def close(self):
        """
        关闭 Chatbot，释放资源
        
        在程序退出前应调用此方法，否则可能导致程序无法正常退出
        """
        try:
            # 关闭 tree_text 的 memory_manager (包含 reorganizer 线程)
            if hasattr(self.mem_cube.text_mem, 'memory_manager'):
                self.mem_cube.text_mem.memory_manager.close()
            
            # 关闭 Neo4j 连接
            if hasattr(self.mem_cube.text_mem, 'graph_store'):
                self.mem_cube.text_mem.graph_store.close()
            
            print("✅ Chatbot 资源已释放")
        except Exception as e:
            print(f"⚠️ 关闭 Chatbot 资源时出错: {e}")


def main():
    """主函数 - 交互式对话"""
    print("=" * 60)
    print("🤖 MemOS Chatbot - 拥有长期记忆的聊天机器人")
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
