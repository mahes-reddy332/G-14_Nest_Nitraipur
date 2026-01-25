# Enhanced RAG System with Knowledge Graph Integration

## Overview

The Enhanced RAG (Retrieval-Augmented Generation) System provides intelligent query processing for clinical trial data through one-time CSV ingestion into a knowledge graph, with seamless agent integration for enhanced responses.

## Key Features

- **One-Time Data Ingestion**: Efficiently processes all CSV data once into a persistent knowledge graph
- **Agent Integration**: Leverages AI agents (Supervisor, Reconciliation, Coding, Liaison) for intelligent analysis
- **Intelligent Query Routing**: Automatically routes queries to optimal processing strategies
- **Knowledge Graph**: NetworkX-based graph database for complex relationship queries
- **Real-Time Responses**: Fast query processing with cached knowledge graph
- **Comprehensive Analytics**: Supports factual, analytical, diagnostic, predictive, and prescriptive queries

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Query Router   │ -> │  Enhanced RAG    │ -> │ Knowledge Graph │
│                 │    │   Pipeline       │    │                 │
│ • Route queries │    │ • Process queries│    │ • Patient nodes │
│ • Load balance  │    │ • Agent integration│   │ • Relationships │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │
                              v
                       ┌─────────────────┐
                       │   AI Agents     │
                       │ • Supervisor    │
                       │ • Reconciliation│
                       │ • Coding        │
                       │ • Liaison       │
                       └─────────────────┘
```

## Components

### 1. KnowledgeGraphBuilder
- **Purpose**: One-time ingestion of CSV data into knowledge graph
- **Features**:
  - Automatic data discovery across study directories
  - Incremental updates with change detection
  - Persistent caching for fast reloads
  - Comprehensive relationship mapping

### 2. EnhancedRAGPipeline
- **Purpose**: Intelligent query processing with agent integration
- **Features**:
  - Query type classification (factual, analytical, diagnostic, etc.)
  - Knowledge graph querying with relationship traversal
  - Agent insight integration
  - Conversational response generation

### 3. AgentRAGIntegration
- **Purpose**: Seamless integration between agents and RAG responses
- **Features**:
  - Context-aware agent selection
  - Response coordination and prioritization
  - LongCat AI enhancement for reasoning

### 4. QueryRouter
- **Purpose**: Intelligent routing based on query complexity and requirements
- **Strategies**:
  - `full_rag`: Complete pipeline with agents and graph
  - `agent_direct`: Direct agent consultation
  - `graph_only`: Knowledge graph queries only
  - `simple`: Basic factual responses

## Installation & Setup

### Prerequisites
- Python 3.8+
- Required packages: networkx, pandas, numpy, pathlib
- Clinical trial data in CSV/XLSX format

### Quick Start

```python
from pathlib import Path
from rag.enhanced_rag_system import EnhancedRAGSystem

# Initialize system
data_path = Path("QC Anonymized Study Files")
rag_system = EnhancedRAGSystem(data_path)

# One-time data ingestion
rag_system.initialize()

# Query the system
response = rag_system.query("How many patients are enrolled in Study 1?")
print(response['answer'])
```

## Usage Examples

### Basic Query
```python
response = rag_system.query("What are the common adverse events?")
print(response['answer'])
```

### Advanced Query with Context
```python
context = {
    'user_role': 'clinical_monitor',
    'study_focus': 'Study_1',
    'time_range': 'last_30_days'
}

response = rag_system.query("What critical issues need attention?", context)
```

### Prescriptive Query (Agent-Enhanced)
```python
response = rag_system.query("What should we do to improve data quality?")

# Access agent recommendations
for recommendation in response['agent_recommendations']:
    print(f"• {recommendation}")
```

## Query Types Supported

| Query Type | Description | Example | Agent Integration |
|------------|-------------|---------|-------------------|
| **Factual** | Direct data retrieval | "How many patients in Study 1?" | None |
| **Analytical** | Patterns and trends | "Common adverse events?" | Optional |
| **Diagnostic** | Root cause analysis | "Why are visits delayed?" | Supervisor Agent |
| **Predictive** | Risk assessment | "Which patients at risk?" | Supervisor Agent |
| **Prescriptive** | Recommendations | "What actions to take?" | All Agents |

## Agent Capabilities

### Supervisor Agent (Rex)
- Cross-study pattern analysis
- Risk assessment and forecasting
- Strategic recommendations

### Reconciliation Agent (Codex)
- Safety data analysis (SAEs)
- Data reconciliation issues
- Compliance monitoring

### Coding Agent (Coding)
- Medical coding patterns
- Query resolution analysis
- Dictionary management

### Site Liaison Agent (Liaison)
- Visit compliance monitoring
- Site performance analysis
- Patient retention insights

## Data Format Support

The system automatically processes:
- **CPID_EDC_Metrics.xlsx**: Patient enrollment and demographics
- **eSAE.xlsx**: Adverse event data
- **Visit_Projection.xlsx**: Visit scheduling and completion
- **GlobalCodingReport.xlsx**: Medical coding data
- **AuditTrail.xlsx**: Data change tracking

## Performance Optimization

### Caching Strategy
- Knowledge graph cached to disk after ingestion
- Change detection prevents unnecessary reingestion
- Query result caching for repeated questions

### Query Optimization
- Intelligent routing reduces processing overhead
- Graph traversal limited to relevant subgraphs
- Agent consultation only when needed

### Memory Management
- Lazy loading of large datasets
- Efficient graph storage with NetworkX
- Garbage collection for temporary objects

## Testing & Validation

### Run Test Suite
```bash
python rag/test_enhanced_rag.py
```

### Interactive Testing
```bash
python rag/rag_usage_example.py --interactive
```

### Performance Benchmarking
```python
from rag.test_enhanced_rag import RAGSystemTester

tester = RAGSystemTester(data_path)
results = tester.run_full_test_suite()
```

## API Reference

### EnhancedRAGSystem

#### `initialize(force_reingest=False)`
Initialize the system and perform data ingestion.

#### `query(user_query, context=None)`
Process a natural language query.

**Parameters:**
- `user_query` (str): The question to process
- `context` (dict, optional): Additional context information

**Returns:**
- `dict`: Response with answer, insights, recommendations, and metadata

#### `get_system_status()`
Get current system statistics and status.

#### `rebuild_knowledge_graph()`
Force rebuild of the knowledge graph from fresh data.

### Response Format

```json
{
  "success": true,
  "answer": "Comprehensive answer text...",
  "agent_insights": ["Insight 1", "Insight 2"],
  "agent_recommendations": ["Action 1", "Action 2"],
  "metadata": {
    "processing_time_ms": 150.5,
    "query_type": "prescriptive",
    "agents_consulted": ["supervisor", "liaison"]
  },
  "routing": {
    "strategy": "full_rag",
    "complexity_score": 8
  }
}
```

## Configuration

### Environment Variables
- `LONGCHAT_API_KEY`: API key for LongCat AI integration
- `RAG_CACHE_DIR`: Directory for knowledge graph caching
- `MAX_QUERY_TIMEOUT`: Maximum query processing time (seconds)

### Settings File
See `config/settings.py` for detailed configuration options.

## Troubleshooting

### Common Issues

1. **Data Not Found**
   - Ensure CSV files are in the correct directory structure
   - Check file permissions and formats

2. **Slow Initial Ingestion**
   - First run processes all data - subsequent runs use cache
   - Consider data preprocessing for very large datasets

3. **Agent Integration Errors**
   - Verify LongCat API configuration
   - Check agent framework imports

4. **Memory Issues**
   - Reduce graph traversal depth in complex queries
   - Use `graph_only` routing for large datasets

### Debug Mode
Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

### Adding New Agents
1. Create agent class inheriting from base agent
2. Add to `AgentRAGIntegration` routing logic
3. Update query type classification if needed

### Extending Data Sources
1. Add file pattern to `KnowledgeGraphBuilder._load_study_data()`
2. Implement data parsing logic
3. Add relationship mapping in `_build_study_graph()`

### Custom Query Types
1. Add to `QueryType` enum
2. Update `EnhancedRAGPipeline._classify_query_type()`
3. Add routing logic in `QueryRouter`

## License

This system is part of the Clinical Data Flow Optimizer project.

## Support

For issues and questions:
1. Check the test suite: `python rag/test_enhanced_rag.py`
2. Review logs for detailed error information
3. Ensure all dependencies are installed correctly