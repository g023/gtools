"""
DAG Related includes

Author: g023 (https://github.com/g023)
License: MIT
"""

import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque

@dataclass
class DAGNode:
    id: str
    label: str
    type: str  # action, decision, start, end

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
    
    def validate_no_self_loops(edges: List[Dict], errors: List[str]) -> bool:
        """Rule 7: No edge where from == to"""
        valid = True
        for edge in edges:
            if edge['from'] == edge['to']:
                errors.append(f"Self-loop detected: '{edge['from']}' → '{edge['to']}'")
                valid = False
        return valid
    
    def validate_end_nodes_have_no_outgoing(edges: List[Dict], nodes: Dict[str, DAGNode], errors: List[str]) -> bool:
        """Rule 6: End nodes must have NO outgoing edges"""
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
        
        # Create position map
        position = {node_id: idx for idx, node_id in enumerate(topo_order)}
        
        # Check all node IDs in topo_order exist
        all_nodes_in_edges = set()
        for edge in edges:
            all_nodes_in_edges.add(edge['from'])
            all_nodes_in_edges.add(edge['to'])
        
        missing_nodes = all_nodes_in_edges - set(position.keys())
        if missing_nodes:
            errors.append(f"Topological order missing nodes: {missing_nodes}")
            return False
        
        # Check each edge respects order
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
        # Build adjacency list
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
        
        # Check at least one end node
        if not any(node.type == 'end' for node in nodes.values()):
            errors.append("No 'end' node found. At least one end node is required.")
            valid = False
        
        # Check at most one start node (optional constraint)
        start_nodes = [node for node in nodes.values() if node.type == 'start']
        if len(start_nodes) > 1:
            errors.append(f"Multiple start nodes found: {len(start_nodes)}. Only one start node allowed.")
            valid = False
        
        return valid
    
    def validate_edge_conditions(edges: List[DAGEdge], errors: List[str]) -> bool:
        """Validate condition field format"""
        valid = True
        
        for i, edge in enumerate(edges):
            # Condition should be either null/None or a non-empty string
            if edge.condition is not None and not isinstance(edge.condition, str):
                errors.append(f"Edge {edge.from_node} → {edge.to_node} has invalid condition type")
                valid = False
            elif edge.condition == "":
                errors.append(f"Edge {edge.from_node} → {edge.to_node} has empty string condition. Use null instead.")
                valid = False
        
        return valid
    
    # Main processing
    try:
        # Parse JSON
        cleaned = clean_json_response(json_response)
        data = json.loads(cleaned)
        
        # Validate top-level structure
        if 'reasoning' not in data:
            raise ValueError("Missing 'reasoning' field in JSON")
        if 'dag' not in data:
            raise ValueError("Missing 'dag' field in JSON")
        
        dag_data = data['dag']
        required_keys = {'nodes', 'edges', 'metadata'}
        if not all(key in dag_data for key in required_keys):
            raise ValueError(f"DAG missing required keys. Expected {required_keys}")
        
        # Parse nodes
        nodes = {}
        for node_data in dag_data['nodes']:
            if not all(k in node_data for k in ['id', 'label', 'type']):
                raise ValueError(f"Node missing required fields: {node_data}")
            nodes[node_data['id']] = DAGNode(
                id=node_data['id'],
                label=node_data['label'],
                type=node_data['type']
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
        
        # Collect validation errors
        validation_errors = []
        
        # Run all validations
        is_valid = True
        
        # Rule 7: No self-loops
        if not validate_no_self_loops(dag_data['edges'], validation_errors):
            is_valid = False
        
        # Rule 6: End nodes have no outgoing
        if not validate_end_nodes_have_no_outgoing(dag_data['edges'], nodes, validation_errors):
            is_valid = False
        
        # Node type validation
        if not validate_node_types(nodes, validation_errors):
            is_valid = False
        
        # Edge condition validation
        if not validate_edge_conditions(edges, validation_errors):
            is_valid = False
        
        # Cycle detection
        if not validate_no_cycles(nodes, edges, validation_errors):
            is_valid = False
        
        # Topological order validation (if provided)
        topo_order = dag_data['metadata'].get('topological_order', [])
        if topo_order:
            if not validate_topological_order(dag_data['edges'], topo_order, validation_errors):
                is_valid = False
        
        # Additional check: topological order matches metadata claim
        if dag_data['metadata'].get('is_acyclic', False) and not is_valid:
            validation_errors.append("Metadata claims graph is acyclic but validation found issues")
        
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
    """
    Compute topological order using Kahn's algorithm if not provided.
    
    Returns:
        List of node IDs in topological order
        
    Raises:
        ValueError: If graph has cycles
    """
    # Build graph
    graph = defaultdict(list)
    in_degree = defaultdict(int)
    
    # Initialize all nodes
    for node_id in nodes:
        in_degree[node_id] = 0
    
    # Add edges
    for edge in edges:
        graph[edge.from_node].append(edge.to_node)
        in_degree[edge.to_node] += 1
    
    # Kahn's algorithm
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
    """Print a human-readable summary of the DAG"""
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
    
    print(f"📊 Nodes ({len(dag.nodes)}):")
    for node_id, node in dag.nodes.items():
        outgoing = [e.to_node for e in dag.edges if e.from_node == node_id]
        print(f"  • {node_id} [{node.type}]: {node.label}")
        if outgoing:
            print(f"    → {', '.join(outgoing)}")
    
    print(f"\n🔗 Edges ({len(dag.edges)}):")
    for i, edge in enumerate(dag.edges, 1):
        condition_str = f" [{edge.condition}]" if edge.condition else ""
        print(f"  {i}. {edge.from_node} → {edge.to_node}{condition_str}")
        if edge.reason:
            print(f"     Reason: {edge.reason}")
    
    print(f"\n📈 Topological Order:")
    if dag.topological_order:
        print(f"  {' → '.join(dag.topological_order)}")
    else:
        print("  Not provided (will compute if needed)")
    
    print("=" * 60)

# Example usage
if __name__ == "__main__":
    # The problematic JSON from before
    problematic_json = '''
    {
      "reasoning": "The input statement has reversed order. Planning must come before design, and design must come before implementation. Correcting to: Planning → Design → Build → Testing → Deployment.",
      "dag": {
        "nodes": [
          {"id": "planning", "label": "Planning", "type": "action"},
          {"id": "design", "label": "Design", "type": "action"},
          {"id": "build", "label": "Build", "type": "action"},
          {"id": "testing", "label": "Testing", "type": "action"},
          {"id": "deployment", "label": "Deployment", "type": "end"}
        ],
        "edges": [
          {"from": "planning", "to": "design", "condition": null, "reason": "Planning must precede design"},
          {"from": "design", "to": "build", "condition": null, "reason": "Design must precede build"},
          {"from": "build", "to": "testing", "condition": null, "reason": "Implementation must precede testing"},
          {"from": "testing", "to": "deployment", "condition": null, "reason": "Testing must precede deployment"},
          {"from": "deployment", "to": "deployment", "condition": null, "reason": "Deployment is end"}
        ],
        "metadata": {
          "is_acyclic": true,
          "cycle_explanation": "Linear progression: Planning → Design → Build → Testing → Deployment",
          "parallel_paths": [],
          "topological_order": ["planning", "design", "build", "testing", "deployment"]
        }
      }
    }
    '''
    
    # Process the DAG
    try:
        dag = process_dag_from_json(problematic_json)
        print_dag_summary(dag)
        
        # If valid, compute topological order if needed
        if dag.is_valid and not dag.topological_order:
            order = compute_topological_order(dag.nodes, dag.edges)
            print(f"\n🔄 Computed Topological Order: {' → '.join(order)}")
        
    except ValueError as e:
        print(f"❌ Error: {e}")
        
    print("\n" + "=" * 60)
    print("EXAMPLE WITH CORRECTED JSON")
    print("=" * 60)
    
    # Corrected JSON (without self-loop)
    corrected_json = '''
    {
      "reasoning": "The input statement has reversed order. Planning must come before design, and design must come before implementation. Correcting to: Planning → Design → Build → Testing → Deployment.",
      "dag": {
        "nodes": [
          {"id": "planning", "label": "Planning", "type": "action"},
          {"id": "design", "label": "Design", "type": "action"},
          {"id": "build", "label": "Build", "type": "action"},
          {"id": "testing", "label": "Testing", "type": "action"},
          {"id": "deployment", "label": "Deployment", "type": "end"}
        ],
        "edges": [
          {"from": "planning", "to": "design", "condition": null, "reason": "Planning must precede design"},
          {"from": "design", "to": "build", "condition": null, "reason": "Design must precede build"},
          {"from": "build", "to": "testing", "condition": null, "reason": "Implementation must precede testing"},
          {"from": "testing", "to": "deployment", "condition": null, "reason": "Testing must precede deployment"}
        ],
        "metadata": {
          "is_acyclic": true,
          "cycle_explanation": "Linear progression: Planning → Design → Build → Testing → Deployment",
          "parallel_paths": [],
          "topological_order": ["planning", "design", "build", "testing", "deployment"]
        }
      }
    }
    '''
    
    try:
        corrected_dag = process_dag_from_json(corrected_json)
        print_dag_summary(corrected_dag)
    except ValueError as e:
        print(f"❌ Error: {e}")