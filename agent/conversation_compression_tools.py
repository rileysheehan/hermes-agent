"""Tool refresh at the existing compaction commit boundary."""
import logging

logger = logging.getLogger("agent.conversation_compression")


def refresh_agent_tool_definitions(agent) -> bool:
    """Rebuild agent.tools at the compaction commit boundary (the only moment config reaches a forever-session's
    frozen tool schemas; the prompt cache is already invalid). Returns True when tools were added."""
    from agent.evaluation_policy import active
    if active(agent):
        return False
    from tools.mcp_tool_agent import refresh_agent_mcp_tools
    added = refresh_agent_mcp_tools(agent, content_aware=True)
    if added:
        logger.info("Compaction tool refresh added tools: %s", sorted(added))
    return bool(added)
