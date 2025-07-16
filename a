import json
import re
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from fuzzywuzzy import fuzz
from collections import defaultdict
import logging

# Tool 1: Lexicon Fetch Tool
class LexiconFetchTool(LangTool):
    def __init__(self, master_svc_client):
        super().__init__()
        self.client = master_svc_client
        self.current_lexicon = None
        self.lexicon_metadata = {}
        self.fetch_timestamp = None
        
    def fetch_lexicon(self, model: str, country: str, date: Optional[str] = None):
        """
        Fetch lexicon data from master service. NEVER returns the full JSON to LLM.
        Only returns metadata and confirmation.
        """
        try:
            # Use latest date if not specified
            if date is None or date.lower() == "latest":
                date = datetime.now().strftime("%Y-%m-%d")
            
            # Fetch from master service
            lexicon_data = self.client.fetch_options_lexicon(
                model=model, 
                country=country, 
                date=date
            )
            
            # Store internally - NEVER expose to LLM
            self.current_lexicon = lexicon_data
            self.fetch_timestamp = datetime.now()
            
            # Extract metadata only
            self.lexicon_metadata = {
                "model": model,
                "country": country,
                "date": date,
                "total_options": self._count_options(lexicon_data),
                "categories": self._extract_categories(lexicon_data),
                "data_size_mb": self._calculate_size_mb(lexicon_data),
                "fetch_time": self.fetch_timestamp.isoformat()
            }
            
            return {
                "status": "success",
                "message": f"Lexicon fetched successfully for {model} in {country}",
                "metadata": self.lexicon_metadata,
                "warning": "Full lexicon data stored internally - not exposed to maintain token efficiency"
            }
            
        except Exception as e:
            logging.error(f"Failed to fetch lexicon: {str(e)}")
            return {
                "status": "error",
                "message": f"Failed to fetch lexicon: {str(e)}",
                "suggestion": "Please verify model, country, and date parameters"
            }
    
    def _count_options(self, data: Dict) -> int:
        """Count total options in lexicon"""
        count = 0
        def count_recursive(obj):
            nonlocal count
            if isinstance(obj, dict):
                count += len(obj)
                for value in obj.values():
                    count_recursive(value)
            elif isinstance(obj, list):
                for item in obj:
                    count_recursive(item)
        count_recursive(data)
        return count
    
    def _extract_categories(self, data: Dict) -> List[str]:
        """Extract top-level categories"""
        return list(data.keys()) if isinstance(data, dict) else []
    
    def _calculate_size_mb(self, data: Dict) -> float:
        """Calculate approximate size in MB"""
        json_str = json.dumps(data)
        return len(json_str.encode('utf-8')) / (1024 * 1024)
    
    def get_definition(self):
        return {
            "type": "function",
            "name": "fetch_lexicon",
            "description": "Fetch options lexicon data from master service. CRITICAL: This tool stores large JSON internally and only returns metadata to prevent token overflow. Never asks LLM to process the full lexicon JSON.",
            "parameters": {
                "type": "object",
                "properties": {
                    "model": {
                        "type": "string",
                        "description": "Vehicle model (e.g., 'Model Y')"
                    },
                    "country": {
                        "type": "string", 
                        "description": "Country code (e.g., 'USA')"
                    },
                    "date": {
                        "type": "string",
                        "description": "Date in YYYY-MM-DD format, or 'latest' for current date"
                    }
                },
                "required": ["model", "country"]
            }
        }


# Tool 2: Lexicon Query Tool
class LexiconQueryTool(LangTool):
    def __init__(self, fetch_tool: LexiconFetchTool):
        super().__init__()
        self.fetch_tool = fetch_tool
        self.search_index = {}
        self.path_index = {}
        self.keyword_index = defaultdict(list)
        
    def query_lexicon(self, query: str, context_hints: Optional[str] = None):
        """
        Query lexicon data with intelligent search. NEVER returns full JSON paths.
        Uses internal processing to find relevant information.
        """
        if not self.fetch_tool.current_lexicon:
            return {
                "status": "error",
                "message": "No lexicon data available. Please fetch lexicon first.",
                "required_action": "Use fetch_lexicon tool first"
            }
        
        try:
            # Build search index if not exists
            if not self.search_index:
                self._build_search_index()
            
            # Process query with multiple strategies
            results = self._multi_strategy_search(query, context_hints)
            
            # Format results for LLM (NO raw JSON paths)
            formatted_results = self._format_search_results(results, query)
            
            return {
                "status": "success",
                "query": query,
                "results": formatted_results,
                "result_count": len(results),
                "search_metadata": {
                    "strategies_used": ["exact_match", "fuzzy_match", "semantic_match"],
                    "confidence_scores": [r.get("confidence", 0) for r in results]
                }
            }
            
        except Exception as e:
            logging.error(f"Query failed: {str(e)}")
            return {
                "status": "error",
                "message": f"Query processing failed: {str(e)}",
                "suggestion": "Try rephrasing your query or provide more context"
            }
    
    def _build_search_index(self):
        """Build comprehensive search index from lexicon data"""
        lexicon = self.fetch_tool.current_lexicon
        
        def index_recursive(obj, path=""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    current_path = f"{path}.{key}" if path else key
                    
                    # Index the key itself
                    self.path_index[current_path] = {
                        "value": value,
                        "type": type(value).__name__,
                        "path": current_path,
                        "parent": path
                    }
                    
                    # Index keywords from key
                    keywords = self._extract_keywords(key)
                    for keyword in keywords:
                        self.keyword_index[keyword.lower()].append(current_path)
                    
                    # Index description if exists
                    if isinstance(value, dict) and "description" in value:
                        desc_keywords = self._extract_keywords(value["description"])
                        for keyword in desc_keywords:
                            self.keyword_index[keyword.lower()].append(current_path)
                    
                    # Continue recursively
                    index_recursive(value, current_path)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    index_recursive(item, f"{path}[{i}]")
        
        index_recursive(lexicon)
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract searchable keywords from text"""
        # Split camelCase and snake_case
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        text = text.replace('_', ' ').replace('-', ' ')
        
        # Extract words
        words = re.findall(r'\b\w+\b', text.lower())
        return [word for word in words if len(word) > 2]
    
    def _multi_strategy_search(self, query: str, context_hints: Optional[str]) -> List[Dict]:
        """Execute multiple search strategies and combine results"""
        results = []
        query_lower = query.lower()
        query_keywords = self._extract_keywords(query)
        
        # Strategy 1: Exact keyword matching
        exact_matches = self._exact_keyword_search(query_keywords)
        results.extend(exact_matches)
        
        # Strategy 2: Fuzzy string matching
        fuzzy_matches = self._fuzzy_search(query_lower)
        results.extend(fuzzy_matches)
        
        # Strategy 3: Context-based search
        if context_hints:
            context_matches = self._context_search(query_keywords, context_hints)
            results.extend(context_matches)
        
        # Strategy 4: Semantic relationship search
        semantic_matches = self._semantic_search(query_keywords)
        results.extend(semantic_matches)
        
        # Deduplicate and rank results
        return self._rank_and_deduplicate(results)
    
    def _exact_keyword_search(self, keywords: List[str]) -> List[Dict]:
        """Find exact keyword matches"""
        matches = []
        for keyword in keywords:
            if keyword in self.keyword_index:
                for path in self.keyword_index[keyword]:
                    matches.append({
                        "path": path,
                        "data": self.path_index[path],
                        "match_type": "exact_keyword",
                        "confidence": 1.0,
                        "matched_keyword": keyword
                    })
        return matches
    
    def _fuzzy_search(self, query: str) -> List[Dict]:
        """Fuzzy string matching for typos and partial matches"""
        matches = []
        for keyword, paths in self.keyword_index.items():
            ratio = fuzz.ratio(query, keyword)
            if ratio >= 70:  # 70% similarity threshold
                for path in paths:
                    matches.append({
                        "path": path,
                        "data": self.path_index[path],
                        "match_type": "fuzzy",
                        "confidence": ratio / 100,
                        "matched_keyword": keyword
                    })
        return matches
    
    def _context_search(self, keywords: List[str], context: str) -> List[Dict]:
        """Search within specific context/category"""
        context_keywords = self._extract_keywords(context)
        matches = []
        
        for path, data in self.path_index.items():
            path_keywords = self._extract_keywords(path)
            
            # Check if context keywords appear in path
            context_match = any(ck in path_keywords for ck in context_keywords)
            keyword_match = any(k in path_keywords for k in keywords)
            
            if context_match and keyword_match:
                matches.append({
                    "path": path,
                    "data": data,
                    "match_type": "context",
                    "confidence": 0.8,
                    "context_matched": context
                })
        
        return matches
    
    def _semantic_search(self, keywords: List[str]) -> List[Dict]:
        """Semantic relationship search using predefined mappings"""
        # Predefined semantic mappings for automotive domain
        semantic_groups = {
            "safety": ["airbag", "brake", "collision", "warning", "alert", "sensor"],
            "comfort": ["seat", "climate", "leather", "heated", "ventilated", "massage"],
            "performance": ["engine", "turbo", "sport", "suspension", "transmission"],
            "exterior": ["paint", "wheels", "lights", "bumper", "mirror", "roof"],
            "interior": ["dashboard", "console", "trim", "upholstery", "ambient"],
            "technology": ["display", "navigation", "bluetooth", "usb", "wireless", "app"]
        }
        
        matches = []
        for keyword in keywords:
            for group, related_terms in semantic_groups.items():
                if keyword in related_terms:
                    # Find paths containing other terms from the same group
                    for term in related_terms:
                        if term in self.keyword_index:
                            for path in self.keyword_index[term]:
                                matches.append({
                                    "path": path,
                                    "data": self.path_index[path],
                                    "match_type": "semantic",
                                    "confidence": 0.6,
                                    "semantic_group": group
                                })
        
        return matches
    
    def _rank_and_deduplicate(self, results: List[Dict]) -> List[Dict]:
        """Remove duplicates and rank by confidence"""
        # Deduplicate by path
        unique_results = {}
        for result in results:
            path = result["path"]
            if path not in unique_results or result["confidence"] > unique_results[path]["confidence"]:
                unique_results[path] = result
        
        # Sort by confidence
        ranked_results = sorted(unique_results.values(), key=lambda x: x["confidence"], reverse=True)
        
        # Limit to top 20 results
        return ranked_results[:20]
    
    def _format_search_results(self, results: List[Dict], original_query: str) -> List[Dict]:
        """Format results for LLM consumption - NO raw JSON exposure"""
        formatted = []
        
        for result in results:
            data = result["data"]
            
            # Create safe, summarized representation
            formatted_result = {
                "option_name": self._extract_option_name(result["path"]),
                "category": self._extract_category(result["path"]),
                "match_confidence": result["confidence"],
                "match_type": result["match_type"],
                "summary": self._create_summary(data),
                "availability": self._check_availability(data),
                "key_attributes": self._extract_key_attributes(data)
            }
            
            # Add disambiguation info if needed
            if result["confidence"] < 0.8:
                formatted_result["disambiguation_note"] = f"Partial match - please confirm if this is what you're looking for"
            
            formatted.append(formatted_result)
        
        # Add suggestions if results are ambiguous
        if len(formatted) > 5:
            formatted.append({
                "type": "suggestion",
                "message": f"Found {len(formatted)} matches for '{original_query}'. Consider being more specific or ask me to filter by category."
            })
        
        return formatted
    
    def _extract_option_name(self, path: str) -> str:
        """Extract human-readable option name from path"""
        parts = path.split('.')
        return parts[-1].replace('_', ' ').title()
    
    def _extract_category(self, path: str) -> str:
        """Extract category from path"""
        parts = path.split('.')
        return parts[0] if parts else "Unknown"
    
    def _create_summary(self, data: Any) -> str:
        """Create concise summary of option data"""
        if isinstance(data, dict):
            if "description" in data:
                return data["description"][:200] + "..." if len(data["description"]) > 200 else data["description"]
            elif "name" in data:
                return f"Option: {data['name']}"
            else:
                return f"Configuration with {len(data)} properties"
        else:
            return str(data)[:100] + "..." if len(str(data)) > 100 else str(data)
    
    def _check_availability(self, data: Any) -> str:
        """Check if option is available"""
        if isinstance(data, dict):
            if "available" in data:
                return "Available" if data["available"] else "Not Available"
            elif "status" in data:
                return data["status"]
        return "Status Unknown"
    
    def _extract_key_attributes(self, data: Any) -> List[str]:
        """Extract key attributes safely"""
        if isinstance(data, dict):
            important_keys = ["price", "code", "type", "category", "standard", "optional"]
            return [f"{k}: {v}" for k, v in data.items() if k in important_keys]
        return []
    
    def get_definition(self):
        return {
            "type": "function",
            "name": "query_lexicon",
            "description": "Query lexicon data using intelligent search. CRITICAL: This tool processes large JSON internally and returns only summarized, relevant information. Never exposes raw JSON paths or full data structures to prevent token overflow.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language query about lexicon options (e.g., 'what are the extra factories', 'show me safety features')"
                    },
                    "context_hints": {
                        "type": "string",
                        "description": "Optional context to narrow search (e.g., 'interior', 'safety', 'performance')"
                    }
                },
                "required": ["query"]
            }
        }


# Tool 3: Lexicon Modify Tool
class LexiconModifyTool(LangTool):
    def __init__(self, fetch_tool: LexiconFetchTool, query_tool: LexiconQueryTool, change_order_client):
        super().__init__()
        self.fetch_tool = fetch_tool
        self.query_tool = query_tool
        self.change_order_client = change_order_client
        self.pending_changes = []
        self.modification_history = []
        
    def modify_lexicon(self, modification_request: str, target_path: Optional[str] = None, confirmation_id: Optional[str] = None):
        """
        Handle lexicon modifications with intelligent path resolution and user confirmation.
        NEVER processes full JSON directly - uses internal indexing.
        """
        if not self.fetch_tool.current_lexicon:
            return {
                "status": "error",
                "message": "No lexicon data available. Please fetch lexicon first.",
                "required_action": "Use fetch_lexicon tool first"
            }
        
        try:
            # If confirmation_id provided, execute pending change
            if confirmation_id:
                return self._execute_confirmed_change(confirmation_id)
            
            # Parse modification request
            modification_intent = self._parse_modification_request(modification_request)
            
            # Find target locations using query tool's index
            target_locations = self._find_modification_targets(
                modification_intent, 
                target_path
            )
            
            # Handle ambiguous targets
            if len(target_locations) > 1:
                return self._handle_ambiguous_targets(target_locations, modification_intent)
            
            if not target_locations:
                return self._handle_no_targets_found(modification_request)
            
            # Create modification plan
            modification_plan = self._create_modification_plan(
                target_locations[0], 
                modification_intent
            )
            
            # Request user confirmation
            return self._request_modification_confirmation(modification_plan)
            
        except Exception as e:
            logging.error(f"Modification failed: {str(e)}")
            return {
                "status": "error",
                "message": f"Modification processing failed: {str(e)}",
                "suggestion": "Please clarify your modification request"
            }
    
    def _parse_modification_request(self, request: str) -> Dict:
        """Parse natural language modification request"""
        request_lower = request.lower()
        
        # Identify modification type
        modification_type = "update"  # default
        if any(word in request_lower for word in ["add", "create", "insert", "new"]):
            modification_type = "add"
        elif any(word in request_lower for word in ["remove", "delete", "drop"]):
            modification_type = "remove"
        elif any(word in request_lower for word in ["change", "update", "modify", "edit"]):
            modification_type = "update"
        
        # Extract target and value
        target_keywords = self._extract_target_keywords(request)
        new_value = self._extract_new_value(request, modification_type)
        
        return {
            "type": modification_type,
            "target_keywords": target_keywords,
            "new_value": new_value,
            "original_request": request
        }
    
    def _extract_target_keywords(self, request: str) -> List[str]:
        """Extract keywords that identify the target"""
        # Remove common modification verbs
        cleaned_request = re.sub(r'\b(add|remove|delete|change|update|modify|edit|to|from|with)\b', '', request.lower())
        
        # Extract meaningful keywords
        keywords = re.findall(r'\b\w+\b', cleaned_request)
        return [k for k in keywords if len(k) > 2]
    
    def _extract_new_value(self, request: str, mod_type: str) -> Optional[str]:
        """Extract new value from modification request"""
        if mod_type == "remove":
            return None
        
        # Look for patterns like "change X to Y" or "set X to Y"
        to_patterns = [
            r'(?:change|set|update).*?to\s+(.+)',
            r'(?:add|insert).*?["\'](.+?)["\']',
            r'(?:add|insert)\s+(.+)',
        ]
        
        for pattern in to_patterns:
            match = re.search(pattern, request.lower())
            if match:
                return match.group(1).strip()
        
        return None
    
    def _find_modification_targets(self, modification_intent: Dict, target_path: Optional[str]) -> List[Dict]:
        """Find target locations for modification using query tool's index"""
        if target_path:
            # Direct path specified
            if target_path in self.query_tool.path_index:
                return [self.query_tool.path_index[target_path]]
            else:
                return []
        
        # Use keywords to find targets
        targets = []
        for keyword in modification_intent["target_keywords"]:
            if keyword in self.query_tool.keyword_index:
                for path in self.query_tool.keyword_index[keyword]:
                    target_data = self.query_tool.path_index[path]
                    targets.append({
                        "path": path,
                        "data": target_data,
                        "matched_keyword": keyword,
                        "confidence": 0.9
                    })
        
        # Deduplicate and sort by confidence
        unique_targets = {}
        for target in targets:
            path = target["path"]
            if path not in unique_targets or target["confidence"] > unique_targets[path]["confidence"]:
                unique_targets[path] = target
        
        return list(unique_targets.values())
    
    def _handle_ambiguous_targets(self, targets: List[Dict], modification_intent: Dict) -> Dict:
        """Handle case where multiple targets match"""
        options = []
        for i, target in enumerate(targets[:5]):  # Limit to 5 options
            options.append({
                "option_id": f"target_{i}",
                "option_name": self._extract_option_name(target["path"]),
                "category": self._extract_category(target["path"]),
                "current_value": self._safe_value_preview(target["data"]),
                "confidence": target["confidence"]
            })
        
        return {
            "status": "disambiguation_required",
            "message": f"Multiple targets found for modification: '{modification_intent['original_request']}'",
            "options": options,
            "instruction": "Please specify which option you want to modify by referring to the option_id or being more specific in your request."
        }
    
    def _handle_no_targets_found(self, request: str) -> Dict:
        """Handle case where no targets are found"""
        # Suggest similar options
        suggestions = self._suggest_similar_options(request)
        
        return {
            "status": "no_targets_found",
            "message": f"Could not find any options matching: '{request}'",
            "suggestions": suggestions,
            "instruction": "Please try a different search term or be more specific about the option you want to modify."
        }
    
    def _suggest_similar_options(self, request: str) -> List[str]:
        """Suggest similar options when exact match not found"""
        request_keywords = self.query_tool._extract_keywords(request)
        suggestions = []
        
        for keyword in request_keywords:
            # Find fuzzy matches
            for indexed_keyword in self.query_tool.keyword_index.keys():
                if fuzz.ratio(keyword, indexed_keyword) >= 60:
                    suggestions.append(indexed_keyword)
        
        return list(set(suggestions))[:10]  # Return top 10 unique suggestions
    
    def _create_modification_plan(self, target: Dict, modification_intent: Dict) -> Dict:
        """Create detailed modification plan"""
        plan_id = f"mod_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        current_value = self._safe_value_preview(target["data"])
        
        plan = {
            "plan_id": plan_id,
            "modification_type": modification_intent["type"],
            "target_path": target["path"],
            "target_name": self._extract_option_name(target["path"]),
            "current_value": current_value,
            "new_value": modification_intent["new_value"],
            "original_request": modification_intent["original_request"],
            "impact_assessment": self._assess_modification_impact(target, modification_intent),
            "requires_confirmation": True
        }
        
        # Store pending change
        self.pending_changes.append(plan)
        
        return plan
    
    def _assess_modification_impact(self, target: Dict, modification_intent: Dict) -> Dict:
        """Assess the impact of the proposed modification"""
        return {
            "affected_option": self._extract_option_name(target["path"]),
            "modification_type": modification_intent["type"],
            "reversible": True,
            "complexity": "medium",
            "estimated_change_scope": "single_option"
        }
    
    def _request_modification_confirmation(self, plan: Dict) -> Dict:
        """Request user confirmation for modification"""
        return {
            "status": "confirmation_required",
            "message": "Please confirm the following modification:",
            "modification_plan": {
                "plan_id": plan["plan_id"],
                "action": f"{plan['modification_type']} '{plan['target_name']}'",
                "current_value": plan["current_value"],
                "new_value": plan["new_value"],
                "impact": plan["impact_assessment"]
            },
            "instruction": f"To proceed, call this tool again with confirmation_id: '{plan['plan_id']}'"
        }
    
    def _execute_confirmed_change(self, confirmation_id: str) -> Dict:
        """Execute a confirmed modification"""
        # Find pending change
        pending_change = None
        for change in self.pending_changes:
            if change["plan_id"] == confirmation_id:
                pending_change = change
                break
        
        if not pending_change:
            return {
                "status": "error",
                "message": f"No pending change found with ID: {confirmation_id}",
                "available_confirmations": [c["plan_id"] for c in self.pending_changes]
            }
        
        try:
            # Apply modification to internal lexicon
            modified_lexicon = self._apply_modification(pending_change)
            
            # Generate change order using external service
            change_order = self._generate_change_order(
                self.fetch_tool.current_lexicon,
                modified_lexicon
            )
            
            # Update internal state
            self.fetch_tool.current_lexicon = modified_lexicon
            self.modification_history.append(pending_change)
            self.pending_changes.remove(pending_change)
            
            return {
                "status": "success",
                "message": f"Successfully applied modification: {pending_change['modification_type']} '{pending_change['target_name']}'",
                "change_order": change_order,
                "modification_summary": {
                    "action": pending_change["modification_type"],
                    "target": pending_change["target_name"],
                    "previous_value": pending_change["current_value"],
                    "new_value": pending_change["new_value"]
                }
            }
            
        except Exception as e:
            logging.error(f"Failed to execute modification: {str(e)}")
            return {
                "status": "error",
                "message": f"Failed to execute modification: {str(e)}",
                "suggestion": "Please try again or contact support"
            }
    
    def _apply_modification(self, change_plan: Dict) -> Dict:
        """Apply modification to lexicon data"""
        lexicon_copy = json.loads(json.dumps(self.fetch_tool.current_lexicon))
        
        # Navigate to target path
        path_parts = change_plan["target_path"].split('.')
        current_obj = lexicon_copy
        
        # Navigate to parent of target
        for part in path_parts[:-1]:
            if '[' in part and ']' in part:
                # Handle array indices
                key, index = part.split('[')
                index = int(index.rstrip(']'))
                current_obj = current_obj[key][index]
            else:
                current_obj = current_obj[part]
        
        # Apply modification
        target_key = path_parts[-1]
        
        if change_plan["modification_type"] == "add":
            current_obj[target_key] = change_plan["new_value"]
        elif change_plan["modification_type"] == "remove":
            if target_key in current_obj:
                del current_obj[target_key]
        elif change_plan["modification_type"] == "update":
            if target_key in current_obj:
                current_obj[target_key] = change_plan["new_value"]
        
        return lexicon_copy
    
    def _generate_change_order(self, original_lexicon: Dict, modified_lexicon: Dict) -> Dict:
        """Generate change order using external service"""
        try:
            change_order = self.change_order_client.generate_change_order(
                original_lexicon=original_lexicon,
                modified_lexicon=modified_lexicon
            )
            return change_order
        except Exception as e:
            logging.error(f"Failed to generate change order: {str(e)}")
            return {
                "error": "Failed to generate change order",
                "details": str(e)
            }
    
    def _safe_value_preview(self, data: Any) -> str:
        """Create safe preview of value without exposing full structure"""
        if isinstance(data, dict):
            if len(data) == 0:
                return "Empty object"
            elif len(data) == 1:
                key, value = next(iter(data.items()))
                return f"Object with '{key}': {self._safe_value_preview(value)}"
            else:
                return f"Object with {len(data)} properties"
        elif isinstance(data, list):
            return f"Array with {len(data)} items"
        else:
            str_val = str(data)
            return str_val[:100] + "..." if len(str_val) > 100 else str_val
    
    def _extract_option_name(self, path: str) -> str:
        """Extract human-readable option name from path"""
        parts = path.split('.')
        return parts[-1].replace('_', ' ').title()
    
    def _extract_category(self, path: str) -> str:
        """Extract category from path"""
        parts = path.split('.')
        return parts[0] if parts else "Unknown"
    
    def get_definition(self):
        return {
            "type": "function",
            "name": "modify_lexicon",
            "description": "Handle lexicon modifications with intelligent path resolution and user confirmation. CRITICAL: This tool processes large JSON internally and never exposes full data structures. All modifications require user confirmation and generate change orders through external service.",
            "parameters": {
                "type": "object",
                "properties": {
                    "modification_request": {
                        "type": "string",
                        "description": "Natural language modification request (e.g., 'add new safety feature X', 'remove leather option', 'change price to $500')"
                    },
                    "target_path": {
                        "type": "string",
                        "description": "Optional specific path to target (use only if exact path is known)"
                    },
                    "confirmation_id": {
                        "type": "string",
                        "description": "Confirmation ID for executing a previously planned modification"
                    }
                },
                "required": ["modification_request"]
            }
        }


# Master Prompt Addition for LLM System
LEXICON_TOOLS_SYSTEM_PROMPT = """
CRITICAL LEXICON HANDLING INSTRUCTIONS:

You have access to three specialized tools for handling vehicle options lexicon data:

1. **LexiconFetchTool**: Fetches lexicon data from master service
2. **LexiconQueryTool**: Searches and queries lexicon data
3. **LexiconModifyTool**: Handles modifications with confirmation workflow

ABSOLUTE RULES - NEVER VIOLATE:

🚫 **NEVER** request, process, or attempt to analyze the full lexicon JSON directly
🚫 **NEVER** ask the user to provide the full lexicon data in the conversation
🚫 **NEVER** attempt to work with raw lexicon paths or full data structures
🚫 **NEVER** try to access lexicon data without using the provided tools

✅ **ALWAYS** use the tools for any lexicon-related operations
✅ **ALWAYS** work with the summarized, processed results from the tools
✅ **ALWAYS** respect the tool's internal processing and token management

## Tool Usage Patterns:

### For Fetching Lexicon:
- User: "fetch options lexicon for model Y USA"
- You: Use LexiconFetchTool → get metadata only
- Present: Confirmation and basic metadata to user

### For Querying:
- User: "what are the extra factories?"
- You: Use LexiconQueryTool → get processed results
- Present: Formatted, human-readable answers

### For Modifications:
- User: "add new safety feature X"
- You: Use LexiconModifyTool → get confirmation request
- Present: Modification plan requiring user confirmation
- User confirms → Execute with confirmation_id

## Key Behaviors:

1. **Multi-step Workflow**: Always fetch lexicon first, then query/modify
2. **Confirmation Required**: All modifications need explicit user confirmation
3. **Smart Disambiguation**: When multiple matches found, help user choose
4. **Error Handling**: Provide helpful suggestions when operations fail
5. **Token Efficiency**: Tools handle large JSON internally - you work with summaries

## Response Patterns:

- **Success**: Present results clearly with context
- **Ambiguous**: Help user disambiguate with specific options
- **Errors**: Provide actionable suggestions for resolution
- **Modifications**: Always confirm before applying changes

Remember: These tools are designed to prevent token overflow by processing large JSON internally. Trust their processing and work with their formatted outputs.
"""

# Example Usage and Registration
def register_lexicon_tools(langchain_agent, master_svc_client, change_order_client):
    """
    Register all lexicon tools with the LangChain agent
    """
    
    # Initialize tools
    fetch_tool = LexiconFetchTool(master_svc_client)
    query_tool = LexiconQueryTool(fetch_tool)
    modify_tool = LexiconModifyTool(fetch_tool, query_tool, change_order_client)
    
    # Register tools with agent
    langchain_agent.register_tool(fetch_tool)
    langchain_agent.register_tool(query_tool)
    langchain_agent.register_tool(modify_tool)
    
    # Add system prompt
    langchain_agent.add_system_prompt(LEXICON_TOOLS_SYSTEM_PROMPT)
    
    return {
        "fetch_tool": fetch_tool,
        "query_tool": query_tool,
        "modify_tool": modify_tool
    }

# Example conversation flow
"""
User: "fetch options lexicon for model Y USA latest"
→ LexiconFetchTool.fetch_lexicon("Model Y", "USA", "latest")
→ Returns metadata only, stores JSON internally

User: "what are the extra factories?"
→ LexiconQueryTool.query_lexicon("what are the extra factories?")
→ Returns processed, formatted results

User: "add premium sound system option"
→ LexiconModifyTool.modify_lexicon("add premium sound system option")
→ Returns confirmation request with plan_id

User: "yes, proceed with that change"
→ LexiconModifyTool.modify_lexicon("confirm", confirmation_id="mod_20240115_143022")
→ Executes change and returns change order
"""

# Token Safety Validation
class TokenSafetyValidator:
    """
    Validates that tools never expose large JSON structures
    """
    
    @staticmethod
    def validate_tool_response(response: Dict) -> bool:
        """
        Validate that response doesn't contain large JSON structures
        """
        response_str = json.dumps(response)
        
        # Check response size
        if len(response_str) > 50000:  # 50KB limit
            logging.error("Tool response too large - potential token overflow")
            return False
        
        # Check for suspicious patterns
        suspicious_patterns = [
            r'"[^"]{1000,}"',  # Very long string values
            r'\{[^}]{5000,}\}',  # Very large objects
            r'\[[^\]]{5000,}\]',  # Very large arrays
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, response_str):
                logging.error(f"Suspicious pattern detected: {pattern}")
                return False
        
        return True
    
    @staticmethod
    def sanitize_response(response: Dict) -> Dict:
        """
        Sanitize response to ensure token safety
        """
        def sanitize_value(value):
            if isinstance(value, str) and len(value) > 500:
                return value[:500] + "... [truncated for token safety]"
            elif isinstance(value, dict):
                return {k: sanitize_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [sanitize_value(item) for item in value[:10]]  # Limit to 10 items
            else:
                return value
        
        return sanitize_value(response)

# Error Handling Wrapper
def safe_tool_execution(tool_func):
    """
    Decorator to ensure safe tool execution with token validation
    """
    def wrapper(*args, **kwargs):
        try:
            result = tool_func(*args, **kwargs)
            
            # Validate token safety
            if not TokenSafetyValidator.validate_tool_response(result):
                result = TokenSafetyValidator.sanitize_response(result)
            
            return result
            
        except Exception as e:
            logging.error(f"Tool execution failed: {str(e)}")
            return {
                "status": "error",
                "message": f"Tool execution failed: {str(e)}",
                "suggestion": "Please try again or contact support"
            }
    
    return wrapper

# Apply safety wrapper to all tool methods
for tool_class in [LexiconFetchTool, LexiconQueryTool, LexiconModifyTool]:
    for method_name in dir(tool_class):
        if method_name.startswith('_') or method_name in ['get_definition']:
            continue
        method = getattr(tool_class, method_name)
        if callable(method):
            setattr(tool_class, method_name, safe_tool_execution(method))
