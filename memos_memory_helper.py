"""
MemOS 记忆助手模块
==================

封装 MemOS 树形记忆 (tree_text) 的初始化和操作，供 LangChain 中间件使用。

功能：
- 初始化 MemOS 和 MemCube
- 记忆的增加、检索、获取
- 支持增量式记忆添加
"""

import os
import warnings
import logging
from datetime import datetime
from typing import List, Dict, Optional, Any

# 抑制警告
warnings.filterwarnings("ignore", message="Pydantic serializer warnings")
warnings.filterwarnings("ignore", message="`torch_dtype` is deprecated")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*Neo4j.*")

# 设置日志 - 需要在导入 memos 之前和之后都设置
logging.getLogger("memos").setLevel(logging.ERROR)
logging.getLogger("neo4j").setLevel(logging.CRITICAL)

from dotenv import load_dotenv
load_dotenv()

# MemOS imports
from memos.configs.mem_os import MemOSConfigFactory
from memos.mem_os.main import MOS
from memos.mem_cube.general import GeneralMemCube
from memos.configs.mem_cube import GeneralMemCubeConfig

# 再次设置日志级别 - memos 模块导入后才能生效
logging.getLogger("memos").setLevel(logging.ERROR)
logging.getLogger("memos.mem_cube.general").setLevel(logging.ERROR)
logging.getLogger("memos.api.config").setLevel(logging.ERROR)


class MemosMemoryHelper:
    """
    MemOS 记忆助手
    
    封装树形记忆 (tree_text + Neo4j) 的所有操作，
    为 LangChain 中间件提供简洁的接口。
    """
    
    def __init__(
        self,
        user_id: str = "langchain_agent_user",
        top_k: int = 5,
        auto_memorize_interval: int = 4,  # 每 N 条消息自动添加记忆
    ):
        """
        初始化 MemOS 记忆助手
        
        Args:
            user_id: 用户ID，用于区分不同用户的记忆
            top_k: 记忆检索时返回的最大数量
            auto_memorize_interval: 自动添加记忆的间隔（消息数）
        """
        self.user_id = user_id
        self.top_k = top_k
        self.auto_memorize_interval = auto_memorize_interval
        
        # 获取 API 配置
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.openai_base = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
        
        if not self.openai_key:
            raise ValueError("❌ 未配置 OPENAI_API_KEY，请在 .env 文件中设置")
        
        # 对话历史追踪
        self.conversation_history: List[Dict[str, str]] = []
        self.memorized_message_count = 0
        self.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 初始化 MemOS
        self._init_memos()
        
        print(f"✅ MemOS 记忆助手初始化完成")
        print(f"   - 用户ID: {self.user_id}")
        print(f"   - 会话ID: {self.session_id}")
        print(f"   - 记忆检索 top_k: {self.top_k}")
    
    def _init_memos(self):
        """初始化 MemOS 记忆系统"""
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
        self.cube_id = f"{self.user_id}_agent_cube"
        
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
        memory_count = self.get_memory_count()
        if memory_count > 0:
            print(f"   ✅ 发现 {memory_count} 条历史记忆")
        else:
            print(f"   🆕 未发现历史记忆")
    
    def search_memories(self, query: str) -> List[str]:
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
    
    def add_conversation(self, user_message: str, assistant_message: str):
        """
        添加一轮简单对话到历史记录（不包含工具调用）
        
        Args:
            user_message: 用户消息
            assistant_message: 助手回复
        """
        self.conversation_history.append({"role": "user", "content": user_message})
        self.conversation_history.append({"role": "assistant", "content": assistant_message})
        
        # 检查是否需要自动添加记忆
        if len(self.conversation_history) % self.auto_memorize_interval == 0:
            self._add_memories_incrementally()
    
    def add_full_conversation(self, messages: List[Dict[str, Any]]):
        """
        添加完整对话到历史记录，支持工具调用
        
        支持的消息类型：
        - role: "user" - 用户消息
        - role: "assistant" - 助手消息（可包含 tool_calls）
        - role: "tool" - 工具返回结果
        
        Args:
            messages: OpenAI 格式的消息列表，例如：
                [
                    {"role": "user", "content": "搜索今天的新闻"},
                    {"role": "assistant", "content": "", "tool_calls": [...]},
                    {"role": "tool", "tool_call_id": "xxx", "name": "search", "content": "..."},
                    {"role": "assistant", "content": "根据搜索结果..."}
                ]
        """
        for msg in messages:
            self.conversation_history.append(msg)
        
        # 检查是否需要自动添加记忆
        if len(self.conversation_history) % self.auto_memorize_interval == 0:
            self._add_memories_incrementally()
    
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
    
    def clear_memories(self):
        """清空所有记忆"""
        try:
            self.mem_cube.text_mem.delete_all()
            self.conversation_history = []
            self.memorized_message_count = 0
            print("✅ 记忆已清空")
        except Exception as e:
            print(f"❌ 清空记忆失败: {e}")
    
    def get_all_memories(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        获取所有记忆
        
        Args:
            limit: 返回的最大数量
            
        Returns:
            记忆列表
        """
        try:
            all_memories = self.mem_cube.text_mem.get_all(user_name=self.user_id)
            
            if isinstance(all_memories, dict) and 'nodes' in all_memories:
                nodes = all_memories.get('nodes', [])[:limit]
                return [
                    {
                        "memory": node.get('memory', ''),
                        "type": node.get('metadata', {}).get('memory_type', 'Unknown')
                    }
                    for node in nodes
                ]
            else:
                return [
                    {
                        "memory": mem_item.memory,
                        "type": "general"
                    }
                    for mem_item in all_memories[:limit]
                ]
        except Exception as e:
            print(f"⚠️ 获取记忆失败: {e}")
            return []
    
    def format_memories_for_prompt(self, memories: List[str]) -> str:
        """
        将记忆列表格式化为可用于 prompt 的字符串
        
        Args:
            memories: 记忆列表
            
        Returns:
            格式化的字符串
        """
        if not memories:
            return "暂无相关记忆"
        
        formatted = []
        for i, mem in enumerate(memories, 1):
            formatted.append(f"[{i}] {mem}")
        
        return "\n".join(formatted)
    
    def close(self):
        """
        关闭 MemOS 资源，释放连接池和后台线程
        
        在程序退出前应调用此方法，否则可能导致程序无法正常退出
        """
        try:
            # 关闭 tree_text 的 memory_manager (包含 reorganizer 线程)
            if hasattr(self.mem_cube.text_mem, 'memory_manager'):
                self.mem_cube.text_mem.memory_manager.close()
            
            # 关闭 Neo4j 连接
            if hasattr(self.mem_cube.text_mem, 'graph_store'):
                self.mem_cube.text_mem.graph_store.close()
            
            print("✅ MemOS 资源已释放")
        except Exception as e:
            print(f"⚠️ 关闭 MemOS 资源时出错: {e}")


# 全局单例实例（可选使用）
_global_memory_helper: Optional[MemosMemoryHelper] = None


def get_memory_helper(
    user_id: str = "langchain_agent_user",
    top_k: int = 5
) -> MemosMemoryHelper:
    """
    获取或创建全局 MemosMemoryHelper 实例
    
    Args:
        user_id: 用户ID
        top_k: 记忆检索数量
        
    Returns:
        MemosMemoryHelper 实例
    """
    global _global_memory_helper
    
    if _global_memory_helper is None:
        _global_memory_helper = MemosMemoryHelper(user_id=user_id, top_k=top_k)
    
    return _global_memory_helper


if __name__ == "__main__":
    # 测试
    helper = MemosMemoryHelper(user_id="test_user")
    
    # 添加对话
    helper.add_conversation("我喜欢踢足球", "太棒了！运动对身体很有好处。")
    helper.add_conversation("我最喜欢的球星是梅西", "梅西确实是一位出色的球员！")
    
    # 强制添加记忆
    helper.force_memorize()
    
    # 搜索记忆
    memories = helper.search_memories("我的爱好是什么？")
    print("检索到的记忆:", memories)
    
    # 显示所有记忆
    all_mems = helper.get_all_memories()
    print("所有记忆:", all_mems)
