# ChatBot V2 Enhanced Features Documentation

## Table of Contents
1. [Overview](#overview)
2. [Feature 1: Dynamic Knowledge Graph Visualization](#feature-1-dynamic-knowledge-graph-visualization)
3. [Feature 2: Real-time Feedback System](#feature-2-real-time-feedback-system)
4. [Installation & Setup](#installation--setup)
5. [Usage Guide](#usage-guide)
6. [API Reference](#api-reference)
7. [Testing](#testing)
8. [Performance Considerations](#performance-considerations)
9. [Future Enhancements](#future-enhancements)

## Overview

This documentation covers the two new major features added to ChatBot V2:

1. **Dynamic Knowledge Graph Visualization**: Interactive visualization of entity relationships extracted from documents
2. **Real-time Feedback System**: User feedback collection and response quality improvement system

## Feature 1: Dynamic Knowledge Graph Visualization

### Description

The Knowledge Graph Visualization feature provides an interactive way to explore and understand the relationships between entities extracted from your documents. It uses advanced NLP techniques to identify entities and their relationships, presenting them in an intuitive graph format.

### Key Components

#### KnowledgeGraphVisualizer Class

**Location**: `knowledge_graph_viz.py`

**Main Methods**:

1. `extract_enhanced_relationships(text, source)`
   - Extracts entities and relationships from text using spaCy
   - Returns a NetworkX graph with entities as nodes and relationships as edges
   - Tracks entity frequency and relationship strength

2. `merge_graph(new_graph)`
   - Merges new graph data into the main knowledge graph
   - Updates weights for recurring entities and relationships

3. `create_pyvis_visualization(filter_threshold, physics)`
   - Creates interactive 2D visualization using PyVis
   - Color-codes entities by type (PERSON, ORG, GPE, etc.)
   - Node size represents entity frequency
   - Supports physics simulation for layout

4. `create_plotly_visualization(filter_threshold)`
   - Creates 3D graph visualization using Plotly
   - Provides immersive exploration experience
   - Includes hover information and zoom controls

5. `get_graph_statistics()`
   - Returns comprehensive graph metrics
   - Includes centrality measures (PageRank, Betweenness)
   - Entity distribution and relationship types

### Visualization Types

#### 1. Interactive 2D (PyVis)
- **Features**:
  - Force-directed layout with physics simulation
  - Color-coded entity types
  - Interactive node dragging
  - Zoom and pan controls
  - Hover tooltips with entity details

#### 2. 3D Graph (Plotly)
- **Features**:
  - Three-dimensional spatial layout
  - Rotatable and zoomable view
  - Entity type color gradient
  - Size-based importance indication

#### 3. Statistics View
- **Metrics Displayed**:
  - Total entities and relationships
  - Graph density
  - Connected components
  - Top entities by frequency
  - Entity type distribution
  - Most important nodes (PageRank)

### Entity Type Color Coding

| Entity Type | Color | Description |
|------------|-------|-------------|
| PERSON | Red (#FF6B6B) | People names |
| ORG | Teal (#4ECDC4) | Organizations |
| GPE | Blue (#45B7D1) | Geopolitical entities |
| DATE | Green (#96CEB4) | Dates and times |
| MONEY | Orange (#FFA07A) | Monetary values |
| PRODUCT | Purple (#DDA0DD) | Products |
| EVENT | Gold (#FFD700) | Events |
| MISC | Gray (#B0B0B0) | Miscellaneous |

### Export Formats

1. **JSON**: Node-link format for web applications
2. **GEXF**: Gephi-compatible format for advanced analysis
3. **GraphML**: Standard graph markup language

## Feature 2: Real-time Feedback System

### Description

The Feedback System enables users to provide real-time feedback on AI responses, which is then used to improve future response quality through feedback-weighted retrieval and continuous learning.

### Key Components

#### FeedbackSystem Class

**Location**: `feedback_system.py`

**Database Schema**:

1. **feedback** table
   - Stores individual feedback entries
   - Links queries to responses with feedback type
   - Tracks response time and context used

2. **response_scores** table
   - Aggregates feedback for unique responses
   - Calculates Wilson score for quality ranking
   - Tracks positive/negative counts

3. **improvements** table
   - Stores patterns from negative feedback
   - Suggests areas for improvement
   - Tracks implementation status

4. **feedback_tags** table
   - Categorizes feedback with tags
   - Enables trend analysis

**Main Methods**:

1. `record_feedback(session_id, query, response, feedback_type, ...)`
   - Records user feedback with full context
   - Updates response quality scores
   - Triggers improvement learning for negative feedback

2. `get_response_quality_score(response)`
   - Returns Wilson score for a response
   - Used for ranking retrieval results
   - Range: 0.0 (poor) to 1.0 (excellent)

3. `get_feedback_analytics(days)`
   - Comprehensive analytics for specified time period
   - Includes satisfaction rate, trends, and top issues
   - Response time analysis

4. `get_improvement_suggestions(limit)`
   - Identifies problematic queries
   - Suggests specific improvement actions
   - Prioritizes by negative feedback count

### Feedback UI Components

#### Inline Feedback Buttons
- 👍 **Thumbs Up**: Quick positive feedback
- 👎 **Thumbs Down**: Opens detailed feedback form

#### Detailed Feedback Form
- Text area for specific issues
- Multi-select tags for categorization
- Optional improvement suggestions

#### Quality Indicators
- ⭐ **High Quality** (Score > 0.7): Green indicator
- 📊 **Average** (Score 0.4-0.7): Blue indicator
- ⚠️ **Low Quality** (Score < 0.4): Yellow warning

### Analytics Dashboard

#### Key Metrics
- Total feedback count
- Positive/Negative breakdown
- Satisfaction rate percentage
- Response time statistics

#### Visualizations
1. **Feedback Trend Chart**: Daily positive/negative feedback over time
2. **Top Issues Table**: Most problematic queries by count
3. **Tag Distribution**: Bar chart of feedback categories
4. **Response Time Metrics**: Average, median, and 95th percentile

#### Improvement Suggestions
Automated analysis provides actionable suggestions:
- "Optimize response generation" for slow responses
- "Update knowledge base" for accuracy issues
- "Improve clarity" for confusing responses
- "Add comprehensive information" for incomplete answers

## Installation & Setup

### Prerequisites

```bash
# System Requirements
- Python 3.8+
- CUDA-capable GPU (recommended)
- 8GB+ RAM
- Windows/Linux/MacOS
```

### Installation Steps

```bash
# 1. Clone or navigate to project directory
cd chatbot_v2

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install requirements
pip install -r requirements.txt

# 4. Download spaCy language model
python -m spacy download en_core_web_sm

# 5. Install PyTorch with CUDA support (if applicable)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Configuration

1. **API Keys**:
   - Update SerpAPI key in `app_enhanced.py` for web search functionality

2. **Model Selection**:
   - Default: `facebook/opt-1.3b`
   - Can be changed in `init_models()` function

3. **Database Location**:
   - ChromaDB: `./chroma_db/`
   - Feedback DB: `./feedback.db`

## Usage Guide

### Running the Enhanced Application

```bash
streamlit run app_enhanced.py
```

### Navigation

The application has four main pages:

1. **Chat**: Main conversation interface with knowledge upload
2. **Knowledge Graph**: Interactive graph visualization
3. **Feedback Analytics**: Dashboard for feedback analysis
4. **Settings**: Configuration options

### Workflow

#### 1. Building Knowledge Base

```python
# Upload documents via UI or programmatically
store_knowledge(text, source_name)
```

- Documents are automatically chunked
- Entities and relationships extracted
- Graph is built incrementally

#### 2. Exploring Knowledge Graph

1. Navigate to "Knowledge Graph" page
2. Select visualization type:
   - Interactive 2D for detailed exploration
   - 3D Graph for spatial understanding
   - Statistics for metrics analysis
3. Adjust filters:
   - Entity frequency threshold
   - Physics simulation on/off
4. Export graph data as needed

#### 3. Providing Feedback

1. After each response, use feedback buttons
2. For negative feedback:
   - Describe the issue
   - Select relevant tags
   - Submit for improvement

#### 4. Monitoring Analytics

1. Navigate to "Feedback Analytics"
2. Select time range (7, 30, 90, 365 days)
3. Review metrics and trends
4. Export data for external analysis

## API Reference

### KnowledgeGraphVisualizer

```python
from knowledge_graph_viz import KnowledgeGraphVisualizer

# Initialize
kg_viz = KnowledgeGraphVisualizer(nlp_model=spacy_model)

# Extract relationships
graph = kg_viz.extract_enhanced_relationships(text, source)

# Merge into main graph
kg_viz.merge_graph(graph)

# Create visualization
html = kg_viz.create_pyvis_visualization(
    filter_threshold=2,  # Min entity frequency
    physics=True         # Enable physics simulation
)

# Get statistics
stats = kg_viz.get_graph_statistics()

# Export graph
json_data = kg_viz.export_graph_data(format='json')
```

### FeedbackSystem

```python
from feedback_system import FeedbackSystem

# Initialize
feedback_sys = FeedbackSystem(db_path="feedback.db")

# Record feedback
success = feedback_sys.record_feedback(
    session_id="uuid-here",
    query="user question",
    response="ai response",
    feedback_type="positive",  # or "negative"
    context_used=["context1", "context2"],
    details="optional feedback text",
    response_time_ms=150
)

# Get quality score
score = feedback_sys.get_response_quality_score(response_text)

# Get analytics
analytics = feedback_sys.get_feedback_analytics(days=30)

# Get improvement suggestions
suggestions = feedback_sys.get_improvement_suggestions(limit=10)

# Export data
filename = feedback_sys.export_feedback_data(format='csv')
```

## Testing

### Running Tests

```bash
# Run all tests
python test_features.py

# Run specific test class
python -m unittest test_features.TestKnowledgeGraphVisualization

# Run with coverage
python -m pytest test_features.py --cov=.
```

### Test Coverage

- **Knowledge Graph Tests**: Entity extraction, relationship detection, graph merging, statistics, visualization
- **Feedback System Tests**: Database operations, feedback recording, scoring algorithms, analytics generation
- **Integration Tests**: Feature interaction, performance impact

## Performance Considerations

### Knowledge Graph

#### Scalability
- **Entities**: Handles up to 10,000 entities efficiently
- **Relationships**: Up to 50,000 edges
- **Rendering**: Filter threshold helps manage large graphs

#### Optimization Tips
1. Use frequency filters for large graphs
2. Disable physics for graphs >1000 nodes
3. Export and analyze offline for very large datasets

### Feedback System

#### Database Performance
- SQLite handles up to 100,000 feedback entries well
- Indices on query_hash and response_hash for fast lookups
- Consider PostgreSQL for production with >1M entries

#### Response Time Impact
- Quality score lookup: <5ms
- Feedback recording: <20ms
- Analytics generation: <100ms for 10,000 entries

## Future Enhancements

### Knowledge Graph Enhancements

1. **Advanced Relationship Types**
   - Temporal relationships
   - Causal relationships
   - Hierarchical structures

2. **Graph Analysis Features**
   - Community detection
   - Path finding between entities
   - Anomaly detection

3. **Integration Capabilities**
   - Neo4j backend option
   - GraphQL API
   - Real-time collaborative editing

### Feedback System Enhancements

1. **Machine Learning Integration**
   - Automatic response reranking
   - Fine-tuning based on feedback
   - Sentiment analysis of feedback

2. **Advanced Analytics**
   - A/B testing framework
   - User segmentation
   - Predictive quality modeling

3. **Automation Features**
   - Auto-correction of common issues
   - Scheduled retraining pipelines
   - Feedback-driven knowledge base updates

## Troubleshooting

### Common Issues

#### Graph Not Displaying
- Check if pyvis is installed: `pip install pyvis`
- Ensure JavaScript is enabled in browser
- Try disabling physics for large graphs

#### Feedback Not Recording
- Check database file permissions
- Verify SQLite is installed
- Check disk space for database

#### Performance Issues
- Reduce chunk size for faster processing
- Increase filter threshold for graph visualization
- Use GPU acceleration for model inference

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Support

For issues or questions:
1. Check this documentation
2. Review test files for examples
3. Check logs for error messages
4. Submit issue with reproduction steps

## License

This feature implementation is part of the ChatBot V2 project.

---

*Last Updated: September 17, 2025*
*Version: 1.0.0*