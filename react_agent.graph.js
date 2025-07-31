// src/react_agent/graph.js
const { createReactAgent } = require("@langchain/langgraph/prebuilt");
const { ChatOpenAI } = require("@langchain/openai");
const { TavilySearchResults } = require("@langchain/community/tools/tavily_search");
const { Calculator } = require("@langchain/community/tools/calculator");
const { DynamicTool } = require("@langchain/core/tools");

// Custom LLM configuration for your models
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

// Create tools for the ReAct agent
function createReactAgentTools() {
  const tools = [];

  // Calculator tool
  tools.push(new Calculator());

  // Web search tool (optional - requires Tavily API key)
  if (process.env.TAVILY_API_KEY) {
    tools.push(new TavilySearchResults({
      maxResults: 5,
      apiKey: process.env.TAVILY_API_KEY
    }));
  }

  // Current time tool
  tools.push(new DynamicTool({
    name: "current_time",
    description: "Get the current date and time",
    func: async () => {
      return new Date().toISOString();
    }
  }));

  // System info tool
  tools.push(new DynamicTool({
    name: "system_info",
    description: "Get system information including Node.js version and platform",
    func: async () => {
      return JSON.stringify({
        nodeVersion: process.version,
        platform: process.platform,
        arch: process.arch,
        uptime: process.uptime(),
        memoryUsage: process.memoryUsage()
      }, null, 2);
    }
  }));

  // Random number generator
  tools.push(new DynamicTool({
    name: "random_number",
    description: "Generate a random number between min and max. Input format: 'min,max'",
    func: async (input) => {
      const [min, max] = input.split(',').map(n => parseInt(n.trim()));
      if (isNaN(min) || isNaN(max)) {
        return 'Please provide valid numbers in format: min,max';
      }
      const random = Math.floor(Math.random() * (max - min + 1)) + min;
      return `Random number between ${min} and ${max}: ${random}`;
    }
  }));

  return tools;
}

// Create the ReAct agent graph
const graph = createReactAgent({
  llm: createCustomLLM(process.env.DEFAULT_MODEL || 'grok'),
  tools: createReactAgentTools(),
  messageModifier: `You are a helpful ReAct (Reasoning + Acting) agent. You can think step by step, use tools when needed, and provide helpful responses.

Available tools:
- Calculator: For mathematical calculations
- Web search: For finding current information online (if configured)
- Current time: Get the current date and time
- System info: Get information about the system
- Random number: Generate random numbers

When solving problems:
1. Think through the problem step by step
2. Use tools when necessary to gather information or perform calculations
3. Provide clear, helpful responses

Always be accurate and helpful in your responses.`
});

module.exports = { graph };
