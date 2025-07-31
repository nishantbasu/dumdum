// src/research_agent/graph.js
const { StateGraph, START, END, Annotation } = require("@langchain/langgraph");
const { ChatOpenAI } = require("@langchain/openai");
const { DynamicTool } = require("@langchain/core/tools");
const { HumanMessage, SystemMessage, AIMessage } = require("@langchain/core/messages");

// Conditionally import Tavily only if available
let TavilySearchResults;
try {
  TavilySearchResults = require("@langchain/community/tools/tavily_search").TavilySearchResults;
} catch (error) {
  console.log("Tavily not available - web search disabled for research agent");
}

// Define research state using proper Annotation
const ResearchState = Annotation.Root({
  messages: Annotation({
    reducer: (x, y) => x.concat(y),
    default: () => []
  }),
  research_query: Annotation({
    reducer: (x, y) => y ?? x ?? "",
    default: () => ""
  }),
  search_results: Annotation({
    reducer: (x, y) => y ?? x ?? [],
    default: () => []
  }),
  research_summary: Annotation({
    reducer: (x, y) => y ?? x ?? "",
    default: () => ""
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
    temperature: 0.3, // Lower temperature for research accuracy
    streaming: true,
  });
}

// Create research tools
function createResearchTools() {
  const tools = [];

  // Web search tool (optional)
  if (process.env.TAVILY_API_KEY && TavilySearchResults) {
    tools.push(new TavilySearchResults({
      maxResults: 10,
      apiKey: process.env.TAVILY_API_KEY,
      includeAnswer: true,
      includeRawContent: true
    }));
  }

  // Mock research database tool (replace with your actual data sources)
  tools.push(new DynamicTool({
    name: "research_database",
    description: "Search internal research database for academic papers and reports",
    func: async (query) => {
      // Mock implementation - replace with actual database search
      return `Searched internal database for: "${query}". Found 3 relevant research papers: 
      1. "AI Applications in Modern Technology" (2024) - Discusses current AI trends and applications
      2. "Machine Learning Trends and Analysis" (2024) - Comprehensive analysis of ML developments
      3. "Future of Artificial Intelligence" (2023) - Predictions and forecasts for AI advancement
      
      Note: This is a mock result. Replace with actual database integration.`;
    }
  }));

  // Citation formatter tool
  tools.push(new DynamicTool({
    name: "format_citations",
    description: "Format research citations in APA or MLA style. Input format: 'style|citation'",
    func: async (input) => {
      try {
        const [style, ...citationParts] = input.split('|');
        const citation = citationParts.join('|');
        
        if (style.toLowerCase() === 'apa') {
          return `APA Format: ${citation} (Retrieved ${new Date().toLocaleDateString()})`;
        } else if (style.toLowerCase() === 'mla') {
          return `MLA Format: ${citation}. Web. ${new Date().toLocaleDateString()}.`;
        }
        return `Citation: ${citation}`;
      } catch (error) {
        return 'Error formatting citation. Use format: style|citation';
      }
    }
  }));

  return tools;
}

// Research planning node
async function planResearch(state) {
  const { messages } = state;
  const llm = createCustomLLM(process.env.DEFAULT_MODEL || 'grok');
  
  const lastMessage = messages[messages.length - 1];
  const userQuery = lastMessage?.content || "No query provided";
  
  const planningPrompt = `You are a research planning assistant. Analyze this research request and create a structured research plan:

User Request: "${userQuery}"

Create a research plan that includes:
1. Key research questions to investigate
2. Search terms and keywords to use
3. Types of sources to look for
4. Expected output format

Respond with a clear research plan.`;

  try {
    const response = await llm.invoke([new HumanMessage(planningPrompt)]);
    
    return {
      research_query: userQuery,
      messages: [new AIMessage(`Research Plan Created:\n\n${response.content}`)]
    };
  } catch (error) {
    console.error('Research planning error:', error);
    return {
      research_query: userQuery,
      messages: [new AIMessage("Error creating research plan. Proceeding with basic research.")]
    };
  }
}

// Research execution node
async function executeResearch(state) {
  const { research_query, messages } = state;
  const tools = createResearchTools();
  
  const searchResults = [];
  
  // Execute web search if available
  if (process.env.TAVILY_API_KEY && TavilySearchResults) {
    const searchTool = tools.find(t => t.name === 'search' || t.name.includes('search'));
    if (searchTool && research_query) {
      try {
        const searchResult = await searchTool.func(research_query);
        searchResults.push({
          source: 'web_search',
          query: research_query,
          results: searchResult
        });
      } catch (error) {
        console.error('Web search error:', error);
      }
    }
  }

  // Execute database search
  const dbTool = tools.find(t => t.name === 'research_database');
  if (dbTool && research_query) {
    try {
      const dbResult = await dbTool.func(research_query);
      searchResults.push({
        source: 'research_database',
        query: research_query,
        results: dbResult
      });
    } catch (error) {
      console.error('Database search error:', error);
    }
  }

  return {
    search_results: searchResults,
    messages: [...messages, new AIMessage(`Research executed. Found ${searchResults.length} sources of information.`)]
  };
}

// Research synthesis node
async function synthesizeResearch(state) {
  const { research_query, search_results, messages } = state;
  const llm = createCustomLLM(process.env.DEFAULT_MODEL || 'grok');
  
  const synthesisPrompt = `You are a research synthesis expert. Based on the research query and gathered information, create a comprehensive research summary.

Original Query: "${research_query}"

Research Results:
${search_results.map((result, index) => 
  `Source ${index + 1} (${result.source}): ${result.results}`
).join('\n\n')}

Please provide:
1. A comprehensive summary of findings
2. Key insights and patterns
3. Relevant data and statistics (if available)
4. Conclusions and recommendations
5. Areas for further research
6. Properly formatted citations

Format your response as a professional research report.`;

  try {
    const response = await llm.invoke([new HumanMessage(synthesisPrompt)]);
    
    return {
      research_summary: response.content,
      messages: [...messages, response]
    };
  } catch (error) {
    console.error('Research synthesis error:', error);
    return {
      messages: [...messages, new AIMessage("Error synthesizing research. Please try again.")]
    };
  }
}

// Create the research agent workflow
const workflow = new StateGraph(ResearchState)
  .addNode("plan", planResearch)
  .addNode("execute", executeResearch)
  .addNode("synthesize", synthesizeResearch)
  .addEdge(START, "plan")
  .addEdge("plan", "execute")
  .addEdge("execute", "synthesize")
  .addEdge("synthesize", END);

// Compile the graph
const graph = workflow.compile();

module.exports = { graph };
