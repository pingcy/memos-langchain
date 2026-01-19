"""
MemOS 记忆助手模块 V2
======================

封装 MemOS SingleCubeView API 的初始化和操作，供 LangChain 中间件使用。

功能：
- 使用 init_server + SingleCubeView 初始化记忆系统
- 使用 APIADDRequest / APISearchRequest 进行记忆操作
- 记忆的增加、检索、获取
- 支持增量式记忆添加

V2 变更说明:
- 使用 init_server + SingleCubeView 替代原有的 MOS + GeneralMemCube
- 使用 APIADDRequest / APISearchRequest 进行记忆操作
- 配置更简洁，由 init_server 统一管理
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

# MemOS V2 API imports
from memos.api.handlers import init_server
from memos.log import get_logger
from memos.multi_mem_cube.single_cube import SingleCubeView
from memos.api.product_models import APIADDRequest, APISearchRequest

# 再次设置日志级别 - memos 模块导入后才能生效
logging.getLogger("memos").setLevel(logging.ERROR)
logging.getLogger("memos.mem_cube.general").setLevel(logging.ERROR)
logging.getLogger("memos.api.config").setLevel(logging.ERROR)

# 获取 logger
logger = get_logger(__name__)


class MemosMemoryHelperV2:
    """
    MemOS 记忆助手 V2
    
    使用 SingleCubeView API 封装记忆操作，
    为 LangChain 中间件提供简洁的接口。
    """
    
    def __init__(
        self,
        user_id: str = "langchain_agent_user",
        cube_id: str = None,
        top_k: int = 5,
        auto_memorize_interval: int = 4,  # 每 N 条消息自动添加记忆
    ):
        """
        初始化 MemOS 记忆助手 V2
        
        Args:
            user_id: 用户ID，用于区分不同用户的记忆
            cube_id: MemCube ID，用于区分不同的记忆空间（默认根据 user_id 生成）
            top_k: 记忆检索时返回的最大数量
            auto_memorize_interval: 自动添加记忆的间隔（消息数）
        """
        self.user_id = user_id
        self.cube_id = cube_id or f"{user_id}_agent_cube"
        self.top_k = top_k
        self.auto_memorize_interval = auto_memorize_interval
        
        # 对话历史追踪
        self.conversation_history: List[Dict[str, str]] = []
        self.memorized_message_count = 0
        self.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 初始化 MemOS (使用 V2 API)
        self._init_memos()
        
        print(f"✅ MemOS 记忆助手 V2 初始化完成")
        print(f"   - 用户ID: {self.user_id}")
        print(f"   - Cube ID: {self.cube_id}")
        print(f"   - 会话ID: {self.session_id}")
        print(f"   - 记忆检索 top_k: {self.top_k}")
    
    def _init_memos(self):
        """初始化 MemOS 记忆系统 (使用 V2 API)"""
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
        memory_count = self.get_memory_count()
        if memory_count > 0:
            print(f"   ✅ 发现历史记忆")
        else:
            print(f"   🆕 未发现历史记忆")
    
    def search_memories(self, query: str) -> List[str]:
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
                    query="用户信息",  # 通用查询
                )
            )
            
            if results.get("text_mem") and results["text_mem"][0].get("memories"):
                return len(results["text_mem"][0]["memories"])
            return 0
        except:
            return 0
    
    def clear_memories(self):
        """清空所有记忆（重置会话状态）"""
        try:
            # V2 API 可能没有直接的删除方法，这里清空本地状态
            # 如果需要真正删除，可能需要额外的 API 支持
            self.cube.naive_mem_cube.text_mem.delete_all()
            self.conversation_history = []
            self.memorized_message_count = 0
            print("✅ 本地会话已清空（注：持久化记忆可能仍在后端存储中）")
        except Exception as e:
            print(f"❌ 清空记忆失败: {e}")
    
    def get_all_memories(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        获取所有记忆 (通过宽泛搜索)
        
        Args:
            limit: 返回的最大数量
            
        Returns:
            记忆列表
        """
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
                result = []
                for mem_item in memories:
                    if isinstance(mem_item, dict):
                        result.append({
                            "memory": mem_item.get('memory', ''),
                            "type": mem_item.get('metadata', {}).get('memory_type', 'Unknown') 
                                    if isinstance(mem_item.get('metadata'), dict) else 'Unknown'
                        })
                    else:
                        result.append({
                            "memory": str(mem_item),
                            "type": "general"
                        })
                return result
            return []
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
            # 关闭相关组件
            if hasattr(self, 'components'):
                # 尝试关闭可能存在的连接
                if 'naive_mem_cube' in self.components:
                    naive_cube = self.components['naive_mem_cube']
                    if hasattr(naive_cube, 'close'):
                        naive_cube.close()
            
            print("✅ MemOS 资源已释放")
        except Exception as e:
            print(f"⚠️ 关闭 MemOS 资源时出错: {e}")


# 全局单例实例（可选使用）
_global_memory_helper_v2: Optional[MemosMemoryHelperV2] = None


def get_memory_helper_v2(
    user_id: str = "langchain_agent_user",
    cube_id: str = None,
    top_k: int = 5
) -> MemosMemoryHelperV2:
    """
    获取或创建全局 MemosMemoryHelperV2 实例
    
    Args:
        user_id: 用户ID
        cube_id: MemCube ID
        top_k: 记忆检索数量
        
    Returns:
        MemosMemoryHelperV2 实例
    """
    global _global_memory_helper_v2
    
    if _global_memory_helper_v2 is None:
        _global_memory_helper_v2 = MemosMemoryHelperV2(
            user_id=user_id,
            cube_id=cube_id,
            top_k=top_k
        )
    
    return _global_memory_helper_v2


if __name__ == "__main__":
    # 测试
    helper = MemosMemoryHelperV2(user_id="test_user_v2")
    
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
    
    # 关闭资源
    helper.close()
