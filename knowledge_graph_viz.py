"""Dynamic Knowledge Graph Visualization Module

This module provides interactive knowledge graph visualization capabilities
for the chatbot_v2 application.
"""

import networkx as nx
import plotly.graph_objects as go
from pyvis.network import Network
import streamlit as st
import streamlit.components.v1 as components
import json
import pandas as pd
from typing import Dict, List, Tuple, Optional
import spacy
from collections import defaultdict
import numpy as np

class KnowledgeGraphVisualizer:
    """Interactive Knowledge Graph Visualization System"""
    
    def __init__(self, nlp_model=None):
        """Initialize the visualizer with NLP model"""
        self.nlp = nlp_model or spacy.load("en_core_web_sm")
        self.graph = nx.Graph()
        self.entity_counts = defaultdict(int)
        self.relationship_counts = defaultdict(int)
        self.node_metadata = {}
        
    def extract_enhanced_relationships(self, text: str, source: str) -> nx.Graph:
        """Extract entities and relationships with enhanced metadata"""
        doc = self.nlp(text)
        local_graph = nx.Graph()
        
        # Extract entities with more detail
        entities = {}
        for ent in doc.ents:
            entities[ent.text] = {
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char,
                'source': source
            }
            self.entity_counts[ent.text] += 1
            
        # Add entities as nodes with metadata
        for entity, metadata in entities.items():
            local_graph.add_node(
                entity,
                label=metadata['label'],
                source=metadata['source'],
                weight=self.entity_counts[entity]
            )
            
        # Extract relationships using dependency parsing
        for token in doc:
            # Subject-Verb-Object relationships
            if token.dep_ == "nsubj" and token.head.pos_ == "VERB":
                subject = token.text
                verb = token.head.text
                
                # Find object
                for child in token.head.children:
                    if child.dep_ in ["dobj", "pobj", "attr"]:
                        obj = child.text
                        relationship = f"{subject}-{verb}-{obj}"
                        
                        if subject in entities or obj in entities:
                            local_graph.add_edge(
                                subject, obj,
                                relationship=verb,
                                weight=1.0
                            )
                            self.relationship_counts[relationship] += 1
                            
            # Compound relationships
            elif token.dep_ == "compound":
                if token.text in entities and token.head.text in entities:
                    local_graph.add_edge(
                        token.text, token.head.text,
                        relationship="compound",
                        weight=0.5
                    )
                    
        return local_graph
    
    def merge_graph(self, new_graph: nx.Graph) -> None:
        """Merge a new graph into the main knowledge graph"""
        # Add nodes with updated weights
        for node, attrs in new_graph.nodes(data=True):
            if self.graph.has_node(node):
                # Update weight
                current_weight = self.graph.nodes[node].get('weight', 1)
                self.graph.nodes[node]['weight'] = current_weight + attrs.get('weight', 1)
            else:
                self.graph.add_node(node, **attrs)
                
        # Add edges with updated weights
        for u, v, attrs in new_graph.edges(data=True):
            if self.graph.has_edge(u, v):
                # Update weight
                current_weight = self.graph.edges[u, v].get('weight', 1)
                self.graph.edges[u, v]['weight'] = current_weight + attrs.get('weight', 1)
            else:
                self.graph.add_edge(u, v, **attrs)
                
    def create_pyvis_visualization(self, 
                                  filter_threshold: int = 1,
                                  physics: bool = True) -> str:
        """Create interactive PyVis network visualization"""
        net = Network(
            height="600px",
            width="100%",
            bgcolor="#222222",
            font_color="white",
            notebook=False,
            directed=False
        )
        
        # Configure physics
        if physics:
            net.force_atlas_2based(
                gravity=-50,
                central_gravity=0.01,
                spring_length=100,
                spring_strength=0.08,
                damping=0.4
            )
        else:
            net.toggle_physics(False)
            
        # Add filtered nodes
        for node, attrs in self.graph.nodes(data=True):
            weight = attrs.get('weight', 1)
            if weight >= filter_threshold:
                label = attrs.get('label', 'MISC')
                
                # Color coding by entity type
                color_map = {
                    'PERSON': '#FF6B6B',
                    'ORG': '#4ECDC4',
                    'GPE': '#45B7D1',
                    'DATE': '#96CEB4',
                    'MONEY': '#FFA07A',
                    'PRODUCT': '#DDA0DD',
                    'EVENT': '#FFD700',
                    'MISC': '#B0B0B0'
                }
                
                color = color_map.get(label, '#B0B0B0')
                size = min(10 + weight * 2, 50)  # Node size based on frequency
                
                net.add_node(
                    node,
                    label=node,
                    title=f"Type: {label}\nFrequency: {weight}\nSource: {attrs.get('source', 'Unknown')}",
                    color=color,
                    size=size
                )
                
        # Add filtered edges
        for u, v, attrs in self.graph.edges(data=True):
            if self.graph.nodes[u].get('weight', 1) >= filter_threshold and \
               self.graph.nodes[v].get('weight', 1) >= filter_threshold:
                relationship = attrs.get('relationship', 'related')
                weight = attrs.get('weight', 1)
                
                net.add_edge(
                    u, v,
                    title=f"Relationship: {relationship}\nStrength: {weight}",
                    width=min(weight, 5)
                )
                
        # Generate HTML
        html = net.generate_html()
        return html
    
    def create_plotly_visualization(self, filter_threshold: int = 1) -> go.Figure:
        """Create Plotly 3D network visualization"""
        # Filter graph
        filtered_graph = nx.Graph()
        for node, attrs in self.graph.nodes(data=True):
            if attrs.get('weight', 1) >= filter_threshold:
                filtered_graph.add_node(node, **attrs)
                
        for u, v, attrs in self.graph.edges(data=True):
            if filtered_graph.has_node(u) and filtered_graph.has_node(v):
                filtered_graph.add_edge(u, v, **attrs)
                
        # Generate layout
        pos = nx.spring_layout(filtered_graph, dim=3, k=0.5, iterations=50)
        
        # Extract node positions
        node_x = []
        node_y = []
        node_z = []
        node_text = []
        node_color = []
        
        for node in filtered_graph.nodes():
            x, y, z = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_z.append(z)
            
            attrs = filtered_graph.nodes[node]
            text = f"{node}<br>Type: {attrs.get('label', 'MISC')}<br>Frequency: {attrs.get('weight', 1)}"
            node_text.append(text)
            
            # Color by entity type
            label = attrs.get('label', 'MISC')
            color_value = hash(label) % 10
            node_color.append(color_value)
            
        # Extract edge positions
        edge_x = []
        edge_y = []
        edge_z = []
        
        for edge in filtered_graph.edges():
            x0, y0, z0 = pos[edge[0]]
            x1, y1, z1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_z.extend([z0, z1, None])
            
        # Create traces
        edge_trace = go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z,
            mode='lines',
            line=dict(width=2, color='#888'),
            hoverinfo='none'
        )
        
        node_trace = go.Scatter3d(
            x=node_x, y=node_y, z=node_z,
            mode='markers+text',
            text=[node for node in filtered_graph.nodes()],
            textposition='top center',
            hoverinfo='text',
            hovertext=node_text,
            marker=dict(
                showscale=True,
                colorscale='Viridis',
                color=node_color,
                size=[filtered_graph.nodes[node].get('weight', 1) * 3 for node in filtered_graph.nodes()],
                colorbar=dict(
                    thickness=15,
                    title='Entity Type',
                    xanchor='left',
                    titleside='right'
                ),
                line=dict(width=2)
            )
        )
        
        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace])
        
        fig.update_layout(
            title='Knowledge Graph Visualization',
            showlegend=False,
            hovermode='closest',
            margin=dict(b=0, l=0, r=0, t=40),
            scene=dict(
                xaxis=dict(showgrid=False, zeroline=False, visible=False),
                yaxis=dict(showgrid=False, zeroline=False, visible=False),
                zaxis=dict(showgrid=False, zeroline=False, visible=False)
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        return fig
    
    def get_graph_statistics(self) -> Dict:
        """Get comprehensive graph statistics"""
        stats = {
            'total_nodes': self.graph.number_of_nodes(),
            'total_edges': self.graph.number_of_edges(),
            'density': nx.density(self.graph) if self.graph.number_of_nodes() > 0 else 0,
            'connected_components': nx.number_connected_components(self.graph),
            'avg_degree': np.mean([d for n, d in self.graph.degree()]) if self.graph.number_of_nodes() > 0 else 0,
            'top_entities': sorted(
                [(node, attrs.get('weight', 1)) for node, attrs in self.graph.nodes(data=True)],
                key=lambda x: x[1],
                reverse=True
            )[:10],
            'entity_distribution': self._get_entity_distribution(),
            'relationship_types': self._get_relationship_types()
        }
        
        # Add centrality measures for important nodes
        if self.graph.number_of_nodes() > 0:
            try:
                stats['betweenness_centrality'] = sorted(
                    nx.betweenness_centrality(self.graph).items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:5]
                
                stats['pagerank'] = sorted(
                    nx.pagerank(self.graph).items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:5]
            except:
                pass
                
        return stats
    
    def _get_entity_distribution(self) -> Dict:
        """Get distribution of entity types"""
        distribution = defaultdict(int)
        for node, attrs in self.graph.nodes(data=True):
            label = attrs.get('label', 'MISC')
            distribution[label] += 1
        return dict(distribution)
    
    def _get_relationship_types(self) -> Dict:
        """Get distribution of relationship types"""
        distribution = defaultdict(int)
        for u, v, attrs in self.graph.edges(data=True):
            rel = attrs.get('relationship', 'related')
            distribution[rel] += 1
        return dict(distribution)
    
    def export_graph_data(self, format: str = 'json') -> str:
        """Export graph data in various formats"""
        if format == 'json':
            data = nx.node_link_data(self.graph)
            return json.dumps(data, indent=2)
        elif format == 'gexf':
            # Export to GEXF format for Gephi
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.gexf', delete=False) as f:
                nx.write_gexf(self.graph, f.name)
                return f.name
        elif format == 'graphml':
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.graphml', delete=False) as f:
                nx.write_graphml(self.graph, f.name)
                return f.name
        else:
            raise ValueError(f"Unsupported format: {format}")

def render_knowledge_graph_ui(kg_viz: KnowledgeGraphVisualizer):
    """Render the knowledge graph UI in Streamlit"""
    st.subheader("🕸️ Dynamic Knowledge Graph Visualization")
    
    # Controls
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        viz_type = st.selectbox(
            "Visualization Type",
            ["Interactive 2D (PyVis)", "3D Graph (Plotly)", "Statistics"]
        )
        
    with col2:
        threshold = st.slider(
            "Entity Frequency Filter",
            min_value=1,
            max_value=10,
            value=1,
            help="Show only entities appearing at least N times"
        )
        
    with col3:
        physics = st.checkbox("Enable Physics", value=True, help="Enable force-directed layout")
        
    with col4:
        export_format = st.selectbox("Export Format", ["JSON", "GEXF", "GraphML"])
        
    # Render visualization based on selection
    if viz_type == "Interactive 2D (PyVis)":
        html = kg_viz.create_pyvis_visualization(threshold, physics)
        components.html(html, height=600)
        
    elif viz_type == "3D Graph (Plotly)":
        fig = kg_viz.create_plotly_visualization(threshold)
        st.plotly_chart(fig, use_container_width=True)
        
    elif viz_type == "Statistics":
        stats = kg_viz.get_graph_statistics()
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Entities", stats['total_nodes'])
        col2.metric("Total Relationships", stats['total_edges'])
        col3.metric("Graph Density", f"{stats['density']:.3f}")
        col4.metric("Components", stats['connected_components'])
        
        # Top entities
        st.write("### 🏆 Top Entities by Frequency")
        if stats['top_entities']:
            df = pd.DataFrame(stats['top_entities'], columns=['Entity', 'Frequency'])
            st.dataframe(df, use_container_width=True)
            
        # Entity distribution
        st.write("### 📊 Entity Type Distribution")
        if stats['entity_distribution']:
            df = pd.DataFrame(
                stats['entity_distribution'].items(),
                columns=['Type', 'Count']
            )
            st.bar_chart(df.set_index('Type'))
            
        # Important nodes
        if 'pagerank' in stats and stats['pagerank']:
            st.write("### 🎯 Most Important Entities (PageRank)")
            df = pd.DataFrame(stats['pagerank'], columns=['Entity', 'Score'])
            st.dataframe(df, use_container_width=True)
            
    # Export functionality
    if st.button("📥 Export Graph Data"):
        export_data = kg_viz.export_graph_data(export_format.lower())
        if export_format == "JSON":
            st.download_button(
                "Download JSON",
                export_data,
                file_name="knowledge_graph.json",
                mime="application/json"
            )
        else:
            st.success(f"Graph exported to: {export_data}")