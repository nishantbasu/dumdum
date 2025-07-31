// src/memory_agent/graph.js
const { createReactAgent } = require("@langchain/langgraph/prebuilt");
const { ChatOpenAI } = require("@langchain/openai");
const { MemorySaver } = require("@langchain/langgraph");
const { DynamicTool } = require("@langchain/core/tools");

// Custom LLM configuration
function createCustomLLM(modelName) {
  const modelConfigs = {
    bottlerocket: {
      baseURL: process.env.BOTTLEROCKET_BASE_URL,
      model: process.env.BOTTLEROCKET_MODEL || 'bottlerocket-model'
    },
    tesla: {
      baseURL: process.env.TESLA_BASE_URL,
      model: process.env.TESLA_MODEL || 'tesla-model'
    },
    grok: {
      baseURL: process.env.GROK_BASE_URL,
      model: process.env.GROK_MODEL || 'grok-model'
    }
  };
  
  const config = modelConfigs[modelName] || modelConfigs.grok;
  
  return new ChatOpenAI({
    apiKey: process.env.CUSTOM_API_KEY,
    baseURL: config.baseURL,
    model: config.model,
    temperature: 0.7,
    streaming: true,
  });
}

// Create simple tools for memory agent
function createMemoryTools() {
  const tools = [];

  // Memory recall tool
  tools.push(new DynamicTool({
    name: "recall_info",
    description: "Recall specific information from our conversation history",
    func: async (query) => {
      return `Searching memory for: "${query}". This tool helps demonstrate memory capabilities.`;
    }
  }));

  // Note taking tool
  tools.push(new DynamicTool({
    name: "take_note",
    description: "Take a note about something important the user mentioned",
    func: async (note) => {
      return `Note recorded: "${note}". I'll remember this for future conversations.`;
    }
  }));

  return tools;
}

// Create the memory agent using createReactAgent (simpler approach)
const memory = new MemorySaver();

const graph = createReactAgent({
  llm: createCustomLLM(process.env.DEFAULT_MODEL || 'grok'),
  tools: createMemoryTools(),
  checkpointSaver: memory,
  messageModifier: `You are a memory-enhanced AI assistant with perfect recall of our conversation history. 

Key capabilities:
- Remember all previous conversations and context
- Reference past topics naturally in your responses
- Build on previous discussions seamlessly
- Maintain conversation continuity across sessions
- Help users by recalling what they've mentioned before

Available tools:
- recall_info: Search through conversation memory
- take_note: Record important information for future reference

When responding:
- Always reference relevant past conversations when appropriate
- Build naturally on previous context and discussions
- Show that you remember important details the user has shared
- Ask follow-up questions based on our conversation history
- Use your tools to demonstrate memory capabilities when helpful

Your memory persists across all conversations in this thread. Always be helpful while demonstrating your enhanced memory capabilities.`
});

module.exports = { graph };
