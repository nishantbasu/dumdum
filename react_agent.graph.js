const { createReactAgent } = require("@langchain/langgraph/prebuilt");
const { ChatOpenAI } = require("@langchain/openai");
const { DynamicTool } = require("@langchain/core/tools");

// Your custom LLM
const llm = new ChatOpenAI({
  apiKey: process.env.CUSTOM_API_KEY,
  baseURL: "https://inference.bottlerocket.tesla.com/models/grok/v1", 
  model: "grok-2-latest",
  temperature: 0.7
});

// Simple tools
const tools = [
  new DynamicTool({
    name: "current_time",
    description: "Get current date and time",
    func: async () => new Date().toISOString()
  }),
  
  new DynamicTool({
    name: "echo",
    description: "Echo back the input",
    func: async (input) => `You said: ${input}`
  })
];

// Create agent - this works in JavaScript!
const graph = createReactAgent({
  llm,
  tools,
  messageModifier: "You are a helpful assistant with access to tools."
});

module.exports = { graph };
