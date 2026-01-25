import pickle
from pathlib import Path

graph_path = Path('../knowledge_graph.pkl')
if graph_path.exists():
    with open(graph_path, 'rb') as f:
        cache_data = pickle.load(f)

    print('Cached graph metadata:')
    print(f'  Nodes: {cache_data["metadata"]["node_count"]}')
    print(f'  Edges: {cache_data["metadata"]["edge_count"]}')
    print(f'  Ingestion timestamp: {cache_data["ingestion_timestamp"]}')

    # Check the actual graph
    graph = cache_data['graph']
    print(f'Actual graph nodes: {len(graph.graph.nodes())}')
    print(f'Actual graph edges: {len(graph.graph.edges())}')
else:
    print('No cached graph found')