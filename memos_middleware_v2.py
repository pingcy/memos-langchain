"""
MemOS LangChain 中间件 V2
==========================

使用 LangChain 1.0 的 wrap 风格 hooks + class 类型中间件，
将 MemOS 树形记忆能力注入到 LangChain 智能体中。

V2 变更说明:
- 使用 MemosMemoryHelperV2（基于 init_server + SingleCubeView）
- 配置更简洁，由 init_server 统一管理
- API 接口保持与 V1 兼容

功能：
1. before_agent: 在 agent 开始时检索相关记忆（只执行一次）
2. wrap_model_call: 将检索到的记忆注入到 system prompt
3. after_agent: 在智能体完成后，将对话添加到记忆库

参考：
- https://docs.langchain.com/oss/python/langchain/middleware/custom
- chatbot_with_memos_v3.py
"""

from datetime import datetime
from typing import Any, Callable, Optional

from langchain.agents.middleware import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
)
from langchain.messages import SystemMessage
from langgraph.runtime import Runtime

from memos_memory_helper_v2 import MemosMemoryHelperV2


# ==================== 记忆系统 Prompt 模板 ====================
MEMORY_INJECTION_TEMPLATE = """
# 长期记忆上下文

以下是与当前对话相关的历史记忆信息：

<memories>
{memories}
</memories>

请结合这些记忆信息来回答用户的问题。如果记忆与当前问题无关，可以忽略。
不要直接提及"记忆"或"检索"等系统内部术语。
"""


class MemosMiddlewareV2(AgentMiddleware):
    """
    MemOS 长期记忆中间件 V2
    
    使用 MemosMemoryHelperV2（基于 SingleCubeView API）实现记忆功能。
    
    记忆检索策略：
    - before_agent: 在 agent 开始时检索记忆（只执行一次，避免 ReAct 循环中重复检索）
    - wrap_model_call: 将已检索的记忆注入到每次模型调用的 system prompt
    - after_agent: 在智能体完成后，将对话添加到记忆库
    """
    
    def __init__(
        self,
        user_id: str = "langchain_agent_user",
        cube_id: str = None,
        top_k: int = 5,
        auto_memorize: bool = True,
        verbose: bool = True,
    ):
        """
        初始化 MemOS 中间件 V2
        
        Args:
            user_id: 用户ID，用于区分不同用户的记忆
            cube_id: MemCube ID，用于区分不同的记忆空间
            top_k: 记忆检索时返回的最大数量
            auto_memorize: 是否自动将对话添加到记忆
            verbose: 是否打印详细日志
        """
        self.user_id = user_id
        self.cube_id = cube_id
        self.top_k = top_k
        self.auto_memorize = auto_memorize
        self.verbose = verbose
        
        # 初始化 MemOS 记忆助手 V2
        self.memory_helper = MemosMemoryHelperV2(
            user_id=user_id,
            cube_id=cube_id,
            top_k=top_k
        )
        
        # 当前任务的记忆缓存（每次 agent 调用时重置）
        self._current_memories: list[str] = []
        self._current_query: Optional[str] = None
    
    def _log(self, message: str):
        """打印日志"""
        if self.verbose:
            print(f"🧠 [MemosMiddlewareV2] {message}")
    
    def before_agent(
        self,
        state: AgentState,
        runtime: Runtime,
    ) -> dict[str, Any] | None:
        """
        Node 风格 hook: 在 agent 开始时执行（只执行一次）
        
        在这里检索记忆，避免在 ReAct 循环中重复检索
        
        Args:
            state: 智能体状态
            runtime: 运行时上下文
            
        Returns:
            可选的状态更新
        """
        
        # 重置缓存
        self._current_memories = []
        self._current_query = None
        
        # 从 state 中提取用户的最新消息
        messages = state.get("messages", [])
        user_query = None
        
        # 从后往前查找最新的用户消息
        for msg in reversed(messages):
            msg_type = getattr(msg, 'type', None) or msg.__class__.__name__.lower()
            if msg_type in ('human', 'humanmessage'):
                user_query = msg.content
                break
        
        if user_query:
            self._current_query = user_query
            self._log(f"检测到用户查询: {user_query[:50]}...")
            
            # 检索相关记忆（只执行一次）
            self._current_memories = self.memory_helper.search_memories(user_query)
            
            if self._current_memories:
                self._log(f"检索到 {len(self._current_memories)} 条相关记忆")
            else:
                self._log("未检索到相关记忆")
        
        return None
    
    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """
        Wrap 风格 hook: 在每次模型调用时执行
        
        将 before_agent 中检索到的记忆注入到 system prompt
        （不再重复检索，只注入已缓存的记忆）
        
        Args:
            request: 模型请求对象
            handler: 原始模型调用处理器
            
        Returns:
            模型响应
        """
        # 如果有缓存的记忆，注入到 system prompt
        if self._current_memories:
            request = self._inject_memories_to_prompt(request, self._current_memories)
        
        # 调用原始模型并返回响应
        return handler(request)
    
    def after_agent(
        self,
        state: AgentState,
        runtime: Runtime,
    ) -> dict[str, Any] | None:
        """
        Node 风格 hook: 在智能体完成后执行
        
        将本轮对话添加到记忆库，包括：
        - user: 用户消息
        - assistant: 助手回复（可能包含 tool_calls）
        - tool: 工具返回结果
        
        Args:
            state: 智能体状态
            runtime: 运行时上下文
            
        Returns:
            可选的状态更新
        """
        if not self.auto_memorize:
            return None
        
        # 从 state 的 messages 中提取本轮完整对话
        messages = state.get("messages", [])
        conversation = self._extract_full_conversation(messages)
        
        if conversation:
            self._log(f"将对话添加到记忆库 ({len(conversation)} 条消息)")
            self.memory_helper.add_full_conversation(conversation)
        
        return None
    
    def _extract_full_conversation(self, messages: list) -> list[dict[str, Any]]:
        """
        从消息列表中提取本轮完整对话
        
        包括 user、assistant（带 tool_calls）、tool 消息
        
        Args:
            messages: LangChain 消息列表
            
        Returns:
            OpenAI 格式的消息列表
        """
        # 找到最后一个 user 消息的索引
        user_index = None
        for i in range(len(messages) - 1, -1, -1):
            msg = messages[i]
            msg_type = getattr(msg, 'type', None) or msg.__class__.__name__.lower()
            if msg_type in ('human', 'humanmessage'):
                user_index = i
                break
        
        if user_index is None:
            return []
        
        # 从 user 消息开始，提取后续所有消息
        conversation = []
        for msg in messages[user_index:]:
            msg_type = getattr(msg, 'type', None) or msg.__class__.__name__.lower()
            
            if msg_type in ('human', 'humanmessage'):
                # 用户消息
                conversation.append({
                    "role": "user",
                    "content": msg.content
                })
                
            elif msg_type in ('ai', 'aimessage'):
                # 助手消息
                assistant_msg = {
                    "role": "assistant",
                    "content": msg.content or ""
                }
                
                # 检查是否有 tool_calls
                tool_calls = getattr(msg, 'tool_calls', None)
                if tool_calls:
                    # 转换为 OpenAI 格式的 tool_calls
                    formatted_tool_calls = []
                    for tc in tool_calls:
                        if isinstance(tc, dict):
                            formatted_tool_calls.append({
                                "id": tc.get("id", ""),
                                "type": "function",
                                "function": {
                                    "name": tc.get("name", ""),
                                    "arguments": tc.get("args", "{}")
                                    if isinstance(tc.get("args"), str)
                                    else str(tc.get("args", "{}"))
                                }
                            })
                        else:
                            # 可能是对象形式
                            formatted_tool_calls.append({
                                "id": getattr(tc, 'id', ''),
                                "type": "function",
                                "function": {
                                    "name": getattr(tc, 'name', ''),
                                    "arguments": str(getattr(tc, 'args', '{}'))
                                }
                            })
                    
                    if formatted_tool_calls:
                        assistant_msg["tool_calls"] = formatted_tool_calls
                
                conversation.append(assistant_msg)
                
            elif msg_type in ('tool', 'toolmessage'):
                # 工具返回消息
                tool_call_id = getattr(msg, 'tool_call_id', None) or getattr(msg, 'id', '')
                tool_name = getattr(msg, 'name', 'unknown_tool')
                
                # 处理 content - 可能是字符串或复杂对象
                content = msg.content
                if not isinstance(content, str):
                    import json
                    try:
                        content = json.dumps(content, ensure_ascii=False, default=str)
                    except:
                        content = str(content)
                
                # 截断过长的工具结果（避免存储过多数据）
                if len(content) > 2000:
                    content = content[:2000] + "... [truncated]"
                
                conversation.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "name": tool_name,
                    "content": content
                })
        
        return conversation
    
    def _inject_memories_to_prompt(
        self,
        request: ModelRequest,
        memories: list[str],
    ) -> ModelRequest:
        """
        将检索到的记忆注入到 system prompt
        
        Args:
            request: 原始请求
            memories: 记忆列表
            
        Returns:
            修改后的请求
        """
        # 格式化记忆
        formatted_memories = self.memory_helper.format_memories_for_prompt(memories)
        
        # 构建记忆注入文本
        memory_context = MEMORY_INJECTION_TEMPLATE.format(memories=formatted_memories)
        
        # 获取当前的 system message content blocks
        current_blocks = list(request.system_message.content_blocks)
        
        # 添加记忆上下文到 system message
        new_content = current_blocks + [
            {"type": "text", "text": memory_context}
        ]
        
        new_system_message = SystemMessage(content=new_content)
        
        return request.override(system_message=new_system_message)
    
    # ==================== 便捷方法 ====================
    
    def force_memorize(self):
        """强制将所有未记忆的对话添加到记忆库"""
        self.memory_helper.force_memorize()
    
    def clear_memories(self):
        """清空所有记忆"""
        self.memory_helper.clear_memories()
    
    def get_memory_count(self) -> int:
        """获取记忆数量"""
        return self.memory_helper.get_memory_count()
    
    def show_memories(self, limit: int = 20):
        """显示记忆"""
        memories = self.memory_helper.get_all_memories(limit=limit)
        print(f"\n📚 当前共有 {len(memories)} 条记忆:")
        for i, mem in enumerate(memories, 1):
            print(f"  [{i}] [{mem['type']}] {mem['memory'][:80]}...")
    
    def close(self):
        """关闭中间件，释放 MemOS 资源"""
        self.memory_helper.close()


# ==================== 便捷函数：创建带记忆的中间件 ====================

def create_memos_middleware_v2(
    user_id: str = "langchain_agent_user",
    cube_id: str = None,
    top_k: int = 5,
    auto_memorize: bool = True,
    verbose: bool = True,
) -> MemosMiddlewareV2:
    """
    创建 MemOS 记忆中间件 V2 的便捷函数
    
    Args:
        user_id: 用户ID
        cube_id: MemCube ID
        top_k: 检索数量
        auto_memorize: 是否自动记忆
        verbose: 是否打印日志
        
    Returns:
        MemosMiddlewareV2 实例
    """
    return MemosMiddlewareV2(
        user_id=user_id,
        cube_id=cube_id,
        top_k=top_k,
        auto_memorize=auto_memorize,
        verbose=verbose,
    )


if __name__ == "__main__":
    # 测试中间件创建
    middleware = create_memos_middleware_v2(user_id="test_middleware_user_v2")
    print(f"中间件 V2 创建成功，当前记忆数量: {middleware.get_memory_count()}")
    
    # 关闭资源
    middleware.close()
