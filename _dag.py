"""
DAG Related includes with reasoning/actionable classification

Author: g023 (https://github.com/g023)
License: MIT
"""

import json
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum

class NodeCategory(Enum):
    """Classification of node types for agent guidance"""
    REASONING = "reasoning"      # Analysis, planning, research steps
    ACTIONABLE = "actionable"    # Direct bash commands, file operations
    VERIFICATION = "verification" # Check results, validate outcomes
    DECISION = "decision"         # Branching points requiring user input

@dataclass
class DAGNode:
    id: str
    label: str
    type: str  # action, decision, start, end (legacy compatibility)
    category: NodeCategory  # New: reasoning vs actionable classification
    expected_output: Optional[str] = None  # What this step should produce
    dependencies: List[str] = field(default_factory=list)  # Files/data this step needs
    produces: List[str] = field(default_factory=list)  # Files/data this step creates

@dataclass
class DAGEdge:
    from_node: str
    to_node: str
    condition: Optional[str]
    reason: str

@dataclass
class ProcessedDAG:
    reasoning: str
    nodes: Dict[str, DAGNode]
    edges: List[DAGEdge]
    metadata: Dict
    is_valid: bool = True
    validation_errors: List[str] = field(default_factory=list)
    topological_order: List[str] = field(default_factory=list)
    
    def get_nodes_by_category(self, category: NodeCategory) -> List[DAGNode]:
        """Get all nodes of a specific category"""
        return [node for node in self.nodes.values() if node.category == category]
    
    def get_reasoning_nodes(self) -> List[DAGNode]:
        """Get all reasoning-type nodes"""
        return self.get_nodes_by_category(NodeCategory.REASONING)
    
    def get_actionable_nodes(self) -> List[DAGNode]:
        """Get all actionable nodes"""
        return self.get_nodes_by_category(NodeCategory.ACTIONABLE)
    
    def get_verification_nodes(self) -> List[DAGNode]:
        """Get all verification nodes"""
        return self.get_nodes_by_category(NodeCategory.VERIFICATION)

def process_dag_from_json(json_response: str) -> ProcessedDAG:
    """
    Process and validate a DAG JSON response from the LLM.
    
    Args:
        json_response: JSON string from LLM
        
    Returns:
        ProcessedDAG object with validation results
        
    Raises:
        ValueError: If JSON is invalid or validation fails
    """
    
    def clean_json_response(response: str) -> str:
        """Remove markdown code blocks from response"""
        cleaned = response.strip()
        if cleaned.startswith('```json'):
            cleaned = cleaned[7:]
        elif cleaned.startswith('```'):
            cleaned = cleaned[3:]
        if cleaned.endswith('```'):
            cleaned = cleaned[:-3]
        return cleaned.strip()
    
    def classify_node(node_id: str, label: str, node_type: str) -> NodeCategory:
        """Intelligently classify node based on label and type"""
        label_lower = label.lower()
        
        # Reasoning indicators
        reasoning_keywords = ['analyze', 'research', 'plan', 'think', 'reason', 
                              'consider', 'evaluate', 'assess', 'determine', 
                              'check if', 'verify if', 'understand', 'review']
        
        # Actionable indicators  
        actionable_keywords = ['create', 'write', 'build', 'implement', 'execute',
                               'run', 'install', 'configure', 'edit', 'modify',
                               'delete', 'move', 'copy', 'download', 'fetch']
        
        # Verification indicators
        verification_keywords = ['validate', 'confirm', 'test', 'verify', 'ensure',
                                 'check', 'assert', 'compare', 'diff']
        
        # Decision indicators
        decision_keywords = ['decide', 'choose', 'select', 'branch', 'if', 'else']
        
        # Classification priority: decision > verification > reasoning > actionable
        if any(kw in label_lower for kw in decision_keywords) or node_type == 'decision':
            return NodeCategory.DECISION
        elif any(kw in label_lower for kw in verification_keywords):
            return NodeCategory.VERIFICATION
        elif any(kw in label_lower for kw in reasoning_keywords):
            return NodeCategory.REASONING
        elif any(kw in label_lower for kw in actionable_keywords) or node_type == 'action':
            return NodeCategory.ACTIONABLE
        else:
            # Default classification based on node type
            if node_type == 'start':
                return NodeCategory.REASONING
            elif node_type == 'end':
                return NodeCategory.VERIFICATION
            else:
                return NodeCategory.ACTIONABLE
    
    def infer_expected_output(label: str, category: NodeCategory) -> Optional[str]:
        """Infer what output this step should produce"""
        if category == NodeCategory.REASONING:
            return "Analysis conclusion or decision"
        elif category == NodeCategory.ACTIONABLE:
            label_lower = label.lower()
            if 'create' in label_lower or 'write' in label_lower:
                return "New file or content created"
            elif 'install' in label_lower:
                return "Package installed"
            elif 'configure' in label_lower:
                return "Configuration updated"
            else:
                return "Command execution result"
        elif category == NodeCategory.VERIFICATION:
            return "Validation result (success/failure details)"
        else:  # DECISION
            return "Decision outcome with reasoning"
    
    def validate_no_self_loops(edges: List[Dict], errors: List[str]) -> bool:
        """No edge where from == to"""
        valid = True
        for edge in edges:
            if edge['from'] == edge['to']:
                errors.append(f"Self-loop detected: '{edge['from']}' → '{edge['to']}'")
                valid = False
        return valid
    
    def validate_end_nodes_have_no_outgoing(edges: List[Dict], nodes: Dict[str, DAGNode], errors: List[str]) -> bool:
        """End nodes must have NO outgoing edges"""
        end_node_ids = [node_id for node_id, node in nodes.items() if node.type == 'end']
        valid = True
        
        for edge in edges:
            if edge['from'] in end_node_ids:
                errors.append(
                    f"End node '{edge['from']}' (type='end') has outgoing edge to '{edge['to']}'. "
                    f"End nodes must have no outgoing edges."
                )
                valid = False
        return valid
    
    def validate_topological_order(edges: List[Dict], topo_order: List[str], errors: List[str]) -> bool:
        """Validate that all edges respect the topological order"""
        if not topo_order:
            errors.append("Topological order is missing or empty")
            return False
        
        position = {node_id: idx for idx, node_id in enumerate(topo_order)}
        all_nodes_in_edges = set()
        for edge in edges:
            all_nodes_in_edges.add(edge['from'])
            all_nodes_in_edges.add(edge['to'])
        
        missing_nodes = all_nodes_in_edges - set(position.keys())
        if missing_nodes:
            errors.append(f"Topological order missing nodes: {missing_nodes}")
            return False
        
        valid = True
        for edge in edges:
            if position[edge['from']] >= position[edge['to']]:
                errors.append(
                    f"Edge '{edge['from']}' → '{edge['to']}' violates topological order. "
                    f"Positions: {edge['from']} at {position[edge['from']]}, "
                    f"{edge['to']} at {position[edge['to']]}"
                )
                valid = False
        
        return valid
    
    def validate_no_cycles(nodes: Dict[str, DAGNode], edges: List[DAGEdge], errors: List[str]) -> bool:
        """Detect cycles using DFS"""
        graph = defaultdict(list)
        for edge in edges:
            graph[edge.from_node].append(edge.to_node)
        
        visited = set()
        recursion_stack = set()
        
        def dfs(node: str) -> bool:
            visited.add(node)
            recursion_stack.add(node)
            
            for neighbor in graph[node]:
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in recursion_stack:
                    return True
            
            recursion_stack.remove(node)
            return False
        
        for node in nodes.keys():
            if node not in visited:
                if dfs(node):
                    errors.append(f"Cycle detected in graph involving node: {node}")
                    return False
        
        return True
    
    def validate_node_types(nodes: Dict[str, DAGNode], errors: List[str]) -> bool:
        """Validate node types are allowed"""
        allowed_types = {'action', 'decision', 'start', 'end'}
        valid = True
        
        for node_id, node in nodes.items():
            if node.type not in allowed_types:
                errors.append(f"Node '{node_id}' has invalid type '{node.type}'. "
                            f"Allowed types: {allowed_types}")
                valid = False
        
        if not any(node.type == 'end' for node in nodes.values()):
            errors.append("No 'end' node found. At least one end node is required.")
            valid = False
        
        start_nodes = [node for node in nodes.values() if node.type == 'start']
        if len(start_nodes) > 1:
            errors.append(f"Multiple start nodes found: {len(start_nodes)}. Only one start node allowed.")
            valid = False
        
        return valid
    
    # Main processing
    try:
        cleaned = clean_json_response(json_response)
        data = json.loads(cleaned)
        
        if 'reasoning' not in data:
            raise ValueError("Missing 'reasoning' field in JSON")
        if 'dag' not in data:
            raise ValueError("Missing 'dag' field in JSON")
        
        dag_data = data['dag']
        required_keys = {'nodes', 'edges', 'metadata'}
        if not all(key in dag_data for key in required_keys):
            raise ValueError(f"DAG missing required keys. Expected {required_keys}")
        
        # Parse nodes with enhanced classification
        nodes = {}
        for node_data in dag_data['nodes']:
            if not all(k in node_data for k in ['id', 'label', 'type']):
                raise ValueError(f"Node missing required fields: {node_data}")
            
            # Classify the node
            category = classify_node(
                node_data['id'], 
                node_data['label'], 
                node_data['type']
            )
            
            # Infer expected output
            expected_output = infer_expected_output(node_data['label'], category)
            
            # Get dependencies/produces if provided
            dependencies = node_data.get('dependencies', [])
            produces = node_data.get('produces', [])
            
            nodes[node_data['id']] = DAGNode(
                id=node_data['id'],
                label=node_data['label'],
                type=node_data['type'],
                category=category,
                expected_output=expected_output,
                dependencies=dependencies,
                produces=produces
            )
        
        # Parse edges
        edges = []
        for edge_data in dag_data['edges']:
            if not all(k in edge_data for k in ['from', 'to']):
                raise ValueError(f"Edge missing required fields: {edge_data}")
            edges.append(DAGEdge(
                from_node=edge_data['from'],
                to_node=edge_data['to'],
                condition=edge_data.get('condition'),
                reason=edge_data.get('reason', '')
            ))
        
        # Validation
        validation_errors = []
        is_valid = True
        
        if not validate_no_self_loops(dag_data['edges'], validation_errors):
            is_valid = False
        
        if not validate_end_nodes_have_no_outgoing(dag_data['edges'], nodes, validation_errors):
            is_valid = False
        
        if not validate_node_types(nodes, validation_errors):
            is_valid = False
        
        if not validate_no_cycles(nodes, edges, validation_errors):
            is_valid = False
        
        topo_order = dag_data['metadata'].get('topological_order', [])
        if topo_order:
            if not validate_topological_order(dag_data['edges'], topo_order, validation_errors):
                is_valid = False
        
        return ProcessedDAG(
            reasoning=data['reasoning'],
            nodes=nodes,
            edges=edges,
            metadata=dag_data['metadata'],
            is_valid=is_valid,
            validation_errors=validation_errors,
            topological_order=topo_order if topo_order else []
        )
        
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {e}\nResponse was: {json_response[:500]}...")
    except Exception as e:
        raise ValueError(f"Error processing DAG: {e}")

def compute_topological_order(nodes: Dict[str, DAGNode], edges: List[DAGEdge]) -> List[str]:
    """Compute topological order using Kahn's algorithm"""
    graph = defaultdict(list)
    in_degree = defaultdict(int)
    
    for node_id in nodes:
        in_degree[node_id] = 0
    
    for edge in edges:
        graph[edge.from_node].append(edge.to_node)
        in_degree[edge.to_node] += 1
    
    queue = deque([node_id for node_id, degree in in_degree.items() if degree == 0])
    result = []
    
    while queue:
        node = queue.popleft()
        result.append(node)
        
        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    
    if len(result) != len(nodes):
        raise ValueError(f"Cycle detected! Only {len(result)} of {len(nodes)} nodes processed")
    
    return result

def print_dag_summary(dag: ProcessedDAG):
    """Print a human-readable summary of the DAG with categories"""
    print("=" * 60)
    print("DAG PROCESSING SUMMARY")
    print("=" * 60)
    print(f"✅ Valid: {dag.is_valid}")
    print(f"📝 Reasoning: {dag.reasoning}\n")
    
    if not dag.is_valid:
        print("❌ Validation Errors:")
        for i, error in enumerate(dag.validation_errors, 1):
            print(f"  {i}. {error}")
        print()
    
    # Print categorized nodes
    print(f"📊 Nodes ({len(dag.nodes)}):")
    reasoning_nodes = dag.get_reasoning_nodes()
    actionable_nodes = dag.get_actionable_nodes()
    verification_nodes = dag.get_verification_nodes()
    
    if reasoning_nodes:
        print(f"\n  🧠 REASONING STEPS ({len(reasoning_nodes)}):")
        for node in reasoning_nodes:
            print(f"    • {node.id}: {node.label}")
            if node.expected_output:
                print(f"      Expected: {node.expected_output}")
    
    if actionable_nodes:
        print(f"\n  ⚡ ACTIONABLE STEPS ({len(actionable_nodes)}):")
        for node in actionable_nodes:
            print(f"    • {node.id}: {node.label}")
            if node.produces:
                print(f"      Produces: {', '.join(node.produces)}")
    
    if verification_nodes:
        print(f"\n  ✅ VERIFICATION STEPS ({len(verification_nodes)}):")
        for node in verification_nodes:
            print(f"    • {node.id}: {node.label}")
    
    print(f"\n🔗 Edges ({len(dag.edges)}):")
    for i, edge in enumerate(dag.edges, 1):
        condition_str = f" [{edge.condition}]" if edge.condition else ""
        print(f"  {i}. {edge.from_node} → {edge.to_node}{condition_str}")
    
    print(f"\n📈 Topological Order:")
    if dag.topological_order:
        order_with_categories = []
        for node_id in dag.topological_order:
            node = dag.nodes[node_id]
            emoji = "🧠" if node.category == NodeCategory.REASONING else "⚡" if node.category == NodeCategory.ACTIONABLE else "✅"
            order_with_categories.append(f"{emoji}{node_id}")
        print(f"  {' → '.join(order_with_categories)}")
    
    print("=" * 60)
