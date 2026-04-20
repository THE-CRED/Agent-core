"""
Tool execution loop.

Handles the cycle of LLM response -> tool execution -> continue until done.
"""

import asyncio
from collections.abc import Awaitable, Callable

from agent.errors import ToolExecutionError
from agent.messages import AgentRequest, Message
from agent.response import AgentResponse
from agent.tools import Tool, ToolRegistry
from agent.types.config import ToolLoopConfig
from agent.types.tools import ToolCall, ToolResult


class ToolLoop:
    """
    Manages the tool calling loop.

    Executes tools requested by the LLM and continues the conversation
    until the LLM produces a final response without tool calls.
    """

    def __init__(
        self,
        tools: list[Tool],
        config: ToolLoopConfig | None = None,
    ):
        self.config = config or ToolLoopConfig()
        self.registry = ToolRegistry()
        for tool in tools:
            self.registry.register(tool)

    def execute_tool_calls(
        self,
        tool_calls: list[ToolCall],
    ) -> list[ToolResult]:
        """
        Execute a list of tool calls synchronously.

        Args:
            tool_calls: Tool calls to execute

        Returns:
            List of tool results
        """
        results: list[ToolResult] = []

        for call in tool_calls[: self.config.max_tool_calls_per_iteration]:
            tool = self.registry.get(call.name)

            if tool is None:
                results.append(
                    ToolResult(
                        tool_call_id=call.id,
                        name=call.name,
                        content=f"Error: Unknown tool '{call.name}'",
                        is_error=True,
                    )
                )
                continue

            try:
                content = tool.execute_sync(call.arguments)
                results.append(
                    ToolResult(
                        tool_call_id=call.id,
                        name=call.name,
                        content=content,
                        is_error=False,
                    )
                )
            except Exception as e:
                if self.config.stop_on_error:
                    raise ToolExecutionError(
                        f"Tool '{call.name}' failed: {e}",
                        tool_name=call.name,
                    ) from e
                results.append(
                    ToolResult(
                        tool_call_id=call.id,
                        name=call.name,
                        content=f"Error: {e}",
                        is_error=True,
                    )
                )

        return results

    async def execute_tool_calls_async(
        self,
        tool_calls: list[ToolCall],
    ) -> list[ToolResult]:
        """
        Execute a list of tool calls asynchronously.

        Args:
            tool_calls: Tool calls to execute

        Returns:
            List of tool results
        """
        calls_to_process = tool_calls[: self.config.max_tool_calls_per_iteration]

        if self.config.parallel_tool_execution:
            # Execute tools in parallel
            tasks = [self._execute_single_tool_async(call) for call in calls_to_process]
            return await asyncio.gather(*tasks)
        else:
            # Execute tools sequentially
            results = []
            for call in calls_to_process:
                result = await self._execute_single_tool_async(call)
                results.append(result)
            return results

    async def _execute_single_tool_async(self, call: ToolCall) -> ToolResult:
        """Execute a single tool call."""
        tool = self.registry.get(call.name)

        if tool is None:
            return ToolResult(
                tool_call_id=call.id,
                name=call.name,
                content=f"Error: Unknown tool '{call.name}'",
                is_error=True,
            )

        try:
            if tool.timeout:
                content = await asyncio.wait_for(
                    tool.execute(call.arguments),
                    timeout=tool.timeout,
                )
            else:
                content = await asyncio.wait_for(
                    tool.execute(call.arguments),
                    timeout=self.config.timeout_per_tool,
                )

            return ToolResult(
                tool_call_id=call.id,
                name=call.name,
                content=content,
                is_error=False,
            )
        except asyncio.TimeoutError:
            return ToolResult(
                tool_call_id=call.id,
                name=call.name,
                content=f"Error: Tool '{call.name}' timed out",
                is_error=True,
            )
        except Exception as e:
            if self.config.stop_on_error:
                raise ToolExecutionError(
                    f"Tool '{call.name}' failed: {e}",
                    tool_name=call.name,
                ) from e
            return ToolResult(
                tool_call_id=call.id,
                name=call.name,
                content=f"Error: {e}",
                is_error=True,
            )

    def build_tool_messages(
        self,
        response: AgentResponse,
        results: list[ToolResult],
    ) -> list[Message]:
        """
        Build messages to append to conversation after tool execution.

        Args:
            response: The LLM response containing tool calls
            results: Results from tool execution

        Returns:
            List of messages to append
        """
        messages: list[Message] = []

        # Add assistant message with tool calls
        messages.append(
            Message.assistant(
                content=response.text or "",
                tool_calls=[tc.to_dict() for tc in response.tool_calls],
            )
        )

        # Add tool result messages
        for result in results:
            messages.append(
                Message.tool(
                    content=result.content,
                    tool_call_id=result.tool_call_id,
                    name=result.name,
                )
            )

        return messages

    def run_loop(
        self,
        initial_request: AgentRequest,
        run_fn: Callable[[AgentRequest], AgentResponse],
    ) -> AgentResponse:
        """
        Run the tool loop until completion.

        Args:
            initial_request: The initial request
            run_fn: Function to call the LLM

        Returns:
            Final response after tool loop completion
        """
        request = initial_request
        messages = list(request.messages)

        for iteration in range(self.config.max_iterations):
            # Build request with current messages
            current_request = AgentRequest(
                input=request.input if iteration == 0 else None,
                messages=messages,
                system=request.system,
                tools=request.tools,
                schema=request.schema,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                top_p=request.top_p,
                stop=request.stop,
                metadata=request.metadata,
                session_id=request.session_id,
            )

            # Get response
            response = run_fn(current_request)

            # If no tool calls, we're done
            if not response.has_tool_calls:
                return response

            # Execute tools
            results = self.execute_tool_calls(response.tool_calls)

            # Build messages for next iteration
            tool_messages = self.build_tool_messages(response, results)
            messages.extend(tool_messages)

        # Max iterations reached
        return response

    async def run_loop_async(
        self,
        initial_request: AgentRequest,
        run_fn: Callable[[AgentRequest], Awaitable[AgentResponse]],
    ) -> AgentResponse:
        """
        Run the tool loop asynchronously until completion.

        Args:
            initial_request: The initial request
            run_fn: Async function to call the LLM

        Returns:
            Final response after tool loop completion
        """
        request = initial_request
        messages = list(request.messages)

        for iteration in range(self.config.max_iterations):
            current_request = AgentRequest(
                input=request.input if iteration == 0 else None,
                messages=messages,
                system=request.system,
                tools=request.tools,
                schema=request.schema,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                top_p=request.top_p,
                stop=request.stop,
                metadata=request.metadata,
                session_id=request.session_id,
            )

            response = await run_fn(current_request)

            if not response.has_tool_calls:
                return response

            results = await self.execute_tool_calls_async(response.tool_calls)
            tool_messages = self.build_tool_messages(response, results)
            messages.extend(tool_messages)

        return response


# Re-export for backwards compatibility
__all__ = ["ToolLoopConfig", "ToolLoop"]
