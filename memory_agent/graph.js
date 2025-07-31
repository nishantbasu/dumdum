// src/memory_agent/graph.js
const { StateGraph, START, END, Annotation } = require("@langchain/langgraph");
const { ChatOpenAI } = require("@langchain/openai");
const { MemorySaver } = require("@langchain/langgraph");
const { HumanMessage, SystemMessage, AIMessage } = require("@langchain/core/messages");

// Define memory agent state using proper Annotation
const MemoryState = Annotation.Root({
  messages: Annotation({
    reducer: (x, y) => x.concat(y), 
    default: () => []
  })
});

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

// Memory management node
async function memoryNode(state) {
  const { messages } = state;
  const llm = createCustomLLM(process.env.DEFAULT_MODEL || 'grok');
  
  // System message for memory agent
  const systemMessage = new SystemMessage(`You are a memory-enhanced AI assistant. You have perfect recall of our entire conversation history and can reference previous topics, questions, and context seamlessly.

Key capabilities:
- Remember all previous conversations and context
- Reference past topics naturally 
- Build on previous discussions
- Maintain conversation continuity
- Help users by recalling what they've mentioned before

When responding:
- Reference relevant past conversations when appropriate
- Build on previous context
- Show that you remember important details the user has shared
- Ask follow-up questions based on conversation history

Always be helpful while demonstrating your memory capabilities.`);

  // Create messages array with system message first, then conversation history
  let messagesToSend = [systemMessage];
  
  // Add all messages from state
  if (messages && messages.length > 0) {
    messagesToSend = messagesToSend.concat(messages);
  }
  
  try {
    const response = await llm.invoke(messagesToSend);
    
    return {
      messages: [response]
    };
  } catch (error) {
    console.error('Memory agent error:', error);
    
    const errorMessage = new AIMessage("I apologize, but I'm experiencing a technical issue. Please try again.");
    return {
      messages: [errorMessage]
    };
  }
}

// Create the memory agent workflow
const workflow = new StateGraph(MemoryState)
  .addNode("memory_node", memoryNode)
  .addEdge(START, "memory_node")
  .addEdge("memory_node", END);

// Compile with memory checkpointer for persistence
const memory = new MemorySaver();
const graph = workflow.compile({
  checkpointer: memory
});

module.exports = { graph };
