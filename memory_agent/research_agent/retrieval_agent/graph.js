// src/retrieval_agent/graph.js
const { StateGraph, MessagesAnnotation, START, END } = require("@langchain/langgraph");
const { ChatOpenAI } = require("@langchain/openai");
const { DynamicTool } = require("@langchain/core/tools");
const { HumanMessage, SystemMessage, AIMessage } = require("@langchain/core/messages");

// Define retrieval state
const RetrievalAnnotation = MessagesAnnotation.spec({
  query: {
    value: (x, y) => y ?? x ?? "",
    default: () => ""
  },
  retrieved_docs: {
    value: (x, y) => y ?? x ?? [],
    default: () => []
  },
  context: {
    value: (x, y) => y ?? x ?? "",
    default: () => ""
  }
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
    temperature: 0.1, // Very low temperature for accurate retrieval
    streaming: true,
  });
}

// Mock document store (replace with your actual vector database/knowledge base)
const mockDocuments = [
  {
    id: "doc1",
    title: "Introduction to Machine Learning",
    content: "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed. It involves algorithms that can identify patterns in data and make predictions or decisions.",
    metadata: { category: "AI/ML", date: "2024-01-15", author: "Tech Team" }
  },
  {
    id: "doc2", 
    title: "Web Development Best Practices",
    content: "Modern web development follows several key principles: responsive design, performance optimization, accessibility, security, and maintainable code structure. React, Node.js, and TypeScript are popular technologies in the current ecosystem.",
    metadata: { category: "Web Dev", date: "2024-02-10", author: "Dev Team" }
  },
  {
    id: "doc3",
    title: "Database Design Principles",
    content: "Good database design involves normalization, proper indexing, query optimization, and understanding relationships between entities. SQL and NoSQL databases each have their use cases depending on the application requirements.",
    metadata: { category: "Database", date: "2024-01-20", author: "Data Team" }
  },
  {
    id: "doc4",
    title: "API Security Guidelines",
    content: "API security involves authentication, authorization, rate limiting, input validation, and proper error handling. JWT tokens, OAuth 2.0, and API keys are common authentication methods. Always use HTTPS and validate all inputs.",
    metadata: { category: "Security", date: "2024-02-05", author: "Security Team" }
  },
  {
    id: "doc5",
    title: "Cloud Computing Overview",
    content: "Cloud computing provides on-demand access to computing resources including servers, storage, databases, and applications. Major providers include AWS, Azure, and Google Cloud. Key benefits include scalability, cost-effectiveness, and reliability.",
    metadata: { category: "Cloud", date: "2024-01-30", author: "Infrastructure Team" }
  }
];

// Create retrieval tools
function createRetrievalTools() {
  const tools = [];

  // Document search tool
  tools.push(new DynamicTool({
    name: "search_documents",
    description: "Search through the knowledge base documents using keywords",
    func: async (query) => {
      const searchTerms = query.toLowerCase().split(' ');
      
      const relevantDocs = mockDocuments.filter(doc => {
        const searchableText = `${doc.title} ${doc.content} ${doc.metadata.category}`.toLowerCase();
        return searchTerms.some(term => searchableText.includes(term));
      });

      return JSON.stringify(relevantDocs.map(doc => ({
        id: doc.id,
        title: doc.title,
        content: doc.content.substring(0, 200) + '...',
        category: doc.metadata.category,
        relevanceScore: Math.random() * 0.3 + 0.7 // Mock relevance score
      })));
    }
  }));

  // Document retrieval by ID
  tools.push(new DynamicTool({
    name: "get_document",
    description: "Retrieve a specific document by its ID",
    func: async (docId) => {
      const doc = mockDocuments.find(d => d.id === docId);
      if (!doc) {
        return "Document not found";
      }
      return JSON.stringify(doc);
    }
  }));

  // Semantic similarity search (mock implementation)
  tools.push(new DynamicTool({
    name: "semantic_search",
    description: "Find documents semantically similar to the query",
    func: async (query) => {
      // Mock semantic search - in production, use vector embeddings
      const keywords = {
        'machine learning': ['AI/ML'],
        'web development': ['Web Dev'],
        'database': ['Database'],
        'security': ['Security'],
        'cloud': ['Cloud']
      };

      const queryLower = query.toLowerCase();
      let relevantCategories = [];
      
      for (const [key, categories] of Object.entries(keywords)) {
        if (queryLower.includes(key)) {
          relevantCategories.push(...categories);
        }
      }

      const semanticDocs = mockDocuments.filter(doc => 
        relevantCategories.includes(doc.metadata.category)
      );

      return JSON.stringify(semanticDocs.map(doc => ({
        ...doc,
        semanticScore: Math.random() * 0.2 + 0.8 // Mock semantic score
      })));
    }
  }));

  return tools;
}

// Query analysis node
async function analyzeQuery(state) {
  const { messages } = state;
  const llm = createCustomLLM(process.env.DEFAULT_MODEL || 'grok');
  
  const lastMessage = messages[messages.length - 1];
  const userQuery = lastMessage.content;
  
  const analysisPrompt = `Analyze this user query and determine the best retrieval strategy:

User Query: "${userQuery}"

Consider:
1. What type of information is the user looking for?
2. What are the key search terms and concepts?
3. Should we use keyword search, semantic search, or both?
4. What document categories might be relevant?

Respond with your analysis and recommended search strategy.`;

  try {
    const response = await llm.invoke([new HumanMessage(analysisPrompt)]);
    
    return {
      query: userQuery,
      messages: [new AIMessage(`Query Analysis:\n\n${response.content}`)]
    };
  } catch (error) {
    console.error('Query analysis error:', error);
    return {
      messages: [new AIMessage("Error analyzing query. Proceeding with default search.")]
    };
  }
}

// Document retrieval node
async function retrieveDocuments(state) {
  const { query, messages } = state;
  const tools = createRetrievalTools();
  
  const retrievedDocs = [];
  
  try {
    // Perform keyword search
    const searchTool = tools.find(t => t.name === 'search_documents');
    if (searchTool) {
      const searchResults = await searchTool.func(query);
      const docs = JSON.parse(searchResults);
      retrievedDocs.push(...docs.map(doc => ({ ...doc, searchType: 'keyword' })));
    }

    // Perform semantic search
    const semanticTool = tools.find(t => t.name === 'semantic_search');
    if (semanticTool) {
      const semanticResults = await semanticTool.func(query);
      const semanticDocs = JSON.parse(semanticResults);
      
      // Merge with existing results, avoiding duplicates
      semanticDocs.forEach(doc => {
        if (!retrievedDocs.find(existing => existing.id === doc.id)) {
          retrievedDocs.push({ ...doc, searchType: 'semantic' });
        }
      });
    }

    // Sort by relevance/semantic score
    retrievedDocs.sort((a, b) => {
      const scoreA = a.relevanceScore || a.semanticScore || 0;
      const scoreB = b.relevanceScore || b.semanticScore || 0;
      return scoreB - scoreA;
    });

    return {
      retrieved_docs: retrievedDocs,
      messages: [...messages, new AIMessage(`Retrieved ${retrievedDocs.length} relevant documents.`)]
    };
    
  } catch (error) {
    console.error('Document retrieval error:', error);
    return {
      retrieved_docs: [],
      messages: [...messages, new AIMessage("Error retrieving documents. Please try again.")]
    };
  }
}

// Answer generation node
async function generateAnswer(state) {
  const { query, retrieved_docs, messages } = state;
  const llm = createCustomLLM(process.env.DEFAULT_MODEL || 'grok');
  
  // Create context from retrieved documents
  const context = retrieved_docs.map((doc, index) => 
    `Document ${index + 1} (${doc.title}):\n${doc.content}\nCategory: ${doc.category || doc.metadata?.category}\n`
  ).join('\n---\n');

  const answerPrompt = `You are a knowledgeable assistant with access to a document knowledge base. Answer the user's question based on the retrieved documents.

User Question: "${query}"

Retrieved Context:
${context}

Instructions:
1. Answer the question using information from the retrieved documents
2. Cite which documents you're referencing
3. If the documents don't contain enough information, say so clearly
4. Provide a comprehensive but concise answer
5. Include relevant details and examples from the context

Answer:`;

  try {
    const response = await llm.invoke([new HumanMessage(answerPrompt)]);
    
    return {
      context,
      messages: [...messages, response]
    };
  } catch (error) {
    console.error('Answer generation error:', error);
    return {
      messages: [...messages, new AIMessage("Error generating answer. Please try again.")]
    };
  }
}

// Create the retrieval agent workflow
const workflow = new StateGraph(RetrievalAnnotation)
  .addNode("analyze", analyzeQuery)
  .addNode("retrieve", retrieveDocuments)
  .addNode("generate", generateAnswer)
  .addEdge(START, "analyze")
  .addEdge("analyze", "retrieve")
  .addEdge("retrieve", "generate")
  .addEdge("generate", END);

// Compile the graph
const graph = workflow.compile();

module.exports = { graph };
