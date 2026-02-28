"""Test script for Knowledge Graph and Feedback System features"""

import unittest
import tempfile
import os
import networkx as nx
import sqlite3
from knowledge_graph_viz import KnowledgeGraphVisualizer
from feedback_system import FeedbackSystem
import spacy

class TestKnowledgeGraphVisualization(unittest.TestCase):
    """Test cases for Knowledge Graph Visualization"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.nlp = spacy.load("en_core_web_sm")
        self.kg_viz = KnowledgeGraphVisualizer(self.nlp)
        
    def test_entity_extraction(self):
        """Test entity extraction from text"""
        text = "Apple Inc. was founded by Steve Jobs in Cupertino, California."
        source = "test_doc"
        
        graph = self.kg_viz.extract_enhanced_relationships(text, source)
        
        # Check entities were extracted
        self.assertIn("Apple Inc.", graph.nodes())
        self.assertIn("Steve Jobs", graph.nodes())
        self.assertIn("Cupertino", graph.nodes())
        
    def test_relationship_extraction(self):
        """Test relationship extraction"""
        text = "Microsoft acquired GitHub for $7.5 billion in 2018."
        source = "test_doc"
        
        graph = self.kg_viz.extract_enhanced_relationships(text, source)
        
        # Check for entities
        self.assertIn("Microsoft", graph.nodes())
        self.assertIn("GitHub", graph.nodes())
        self.assertIn("$7.5 billion", graph.nodes())
        
        # Check for edges (relationships)
        self.assertTrue(graph.number_of_edges() > 0)
        
    def test_graph_merging(self):
        """Test merging multiple graphs"""
        text1 = "Google develops Android operating system."
        text2 = "Android is used by Samsung phones."
        
        graph1 = self.kg_viz.extract_enhanced_relationships(text1, "doc1")
        graph2 = self.kg_viz.extract_enhanced_relationships(text2, "doc2")
        
        # Merge graphs
        self.kg_viz.merge_graph(graph1)
        self.kg_viz.merge_graph(graph2)
        
        # Check merged graph
        self.assertIn("Google", self.kg_viz.graph.nodes())
        self.assertIn("Android", self.kg_viz.graph.nodes())
        self.assertIn("Samsung", self.kg_viz.graph.nodes())
        
        # Android should have increased weight (appears in both)
        android_weight = self.kg_viz.graph.nodes["Android"].get('weight', 0)
        self.assertGreater(android_weight, 1)
        
    def test_graph_statistics(self):
        """Test graph statistics calculation"""
        text = "Tesla, led by Elon Musk, manufactures electric vehicles in Fremont."
        graph = self.kg_viz.extract_enhanced_relationships(text, "test")
        self.kg_viz.merge_graph(graph)
        
        stats = self.kg_viz.get_graph_statistics()
        
        self.assertIn('total_nodes', stats)
        self.assertIn('total_edges', stats)
        self.assertIn('density', stats)
        self.assertGreater(stats['total_nodes'], 0)
        
    def test_visualization_html_generation(self):
        """Test HTML visualization generation"""
        text = "Amazon Web Services provides cloud computing services."
        graph = self.kg_viz.extract_enhanced_relationships(text, "test")
        self.kg_viz.merge_graph(graph)
        
        html = self.kg_viz.create_pyvis_visualization(filter_threshold=1, physics=True)
        
        self.assertIsInstance(html, str)
        self.assertIn('<html>', html)
        self.assertIn('</html>', html)
        
    def test_export_functionality(self):
        """Test graph export in different formats"""
        text = "OpenAI created GPT-4 language model."
        graph = self.kg_viz.extract_enhanced_relationships(text, "test")
        self.kg_viz.merge_graph(graph)
        
        # Test JSON export
        json_data = self.kg_viz.export_graph_data(format='json')
        self.assertIsInstance(json_data, str)
        
        import json
        parsed = json.loads(json_data)
        self.assertIn('nodes', parsed)
        self.assertIn('links', parsed)

class TestFeedbackSystem(unittest.TestCase):
    """Test cases for Feedback System"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.feedback_system = FeedbackSystem(db_path=self.temp_db.name)
        
    def tearDown(self):
        """Clean up test fixtures"""
        self.temp_db.close()
        os.unlink(self.temp_db.name)
        
    def test_database_initialization(self):
        """Test database tables are created correctly"""
        conn = sqlite3.connect(self.temp_db.name)
        cursor = conn.cursor()
        
        # Check tables exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        self.assertIn('feedback', tables)
        self.assertIn('response_scores', tables)
        self.assertIn('improvements', tables)
        self.assertIn('feedback_tags', tables)
        
        conn.close()
        
    def test_positive_feedback_recording(self):
        """Test recording positive feedback"""
        success = self.feedback_system.record_feedback(
            session_id="test_session",
            query="What is machine learning?",
            response="Machine learning is a subset of AI...",
            feedback_type="positive",
            context_used=["ML context 1", "ML context 2"],
            response_time_ms=150
        )
        
        self.assertTrue(success)
        
        # Verify in database
        conn = sqlite3.connect(self.temp_db.name)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM feedback WHERE feedback_type='positive'")
        count = cursor.fetchone()[0]
        self.assertEqual(count, 1)
        conn.close()
        
    def test_negative_feedback_recording(self):
        """Test recording negative feedback with details"""
        success = self.feedback_system.record_feedback(
            session_id="test_session",
            query="Explain quantum computing",
            response="I don't know about quantum computing.",
            feedback_type="negative",
            details="Response was incomplete and not helpful",
            context_used=[],
            response_time_ms=50
        )
        
        self.assertTrue(success)
        
        # Check improvements table
        conn = sqlite3.connect(self.temp_db.name)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM improvements")
        count = cursor.fetchone()[0]
        self.assertGreater(count, 0)  # Should trigger improvement learning
        conn.close()
        
    def test_wilson_score_calculation(self):
        """Test Wilson score calculation for ranking"""
        # Test with different positive/negative ratios
        score1 = self.feedback_system._calculate_wilson_score(10, 0)  # All positive
        score2 = self.feedback_system._calculate_wilson_score(5, 5)   # Mixed
        score3 = self.feedback_system._calculate_wilson_score(0, 10)  # All negative
        
        self.assertGreater(score1, score2)
        self.assertGreater(score2, score3)
        self.assertAlmostEqual(score2, 0.5, delta=0.3)  # Should be around 0.5
        
    def test_response_quality_score(self):
        """Test getting quality score for responses"""
        response = "This is a test response."
        
        # Initially should be 0.5 (neutral)
        initial_score = self.feedback_system.get_response_quality_score(response)
        self.assertEqual(initial_score, 0.5)
        
        # Record positive feedback
        self.feedback_system.record_feedback(
            session_id="test",
            query="test query",
            response=response,
            feedback_type="positive"
        )
        
        # Score should increase
        new_score = self.feedback_system.get_response_quality_score(response)
        self.assertGreater(new_score, initial_score)
        
    def test_tag_extraction(self):
        """Test feedback tag extraction"""
        tags1 = self.feedback_system._extract_tags("The response was confusing and incomplete")
        self.assertIn('confusing', tags1)
        self.assertIn('incomplete', tags1)
        
        tags2 = self.feedback_system._extract_tags("Very helpful and accurate answer")
        self.assertIn('helpful', tags2)
        self.assertIn('accuracy', tags2)  # Checks for 'accuracy' in text
        
    def test_feedback_analytics(self):
        """Test analytics generation"""
        # Add some test feedback
        for i in range(5):
            self.feedback_system.record_feedback(
                session_id="test",
                query=f"Query {i}",
                response=f"Response {i}",
                feedback_type="positive" if i < 3 else "negative",
                response_time_ms=100 + i * 50
            )
        
        analytics = self.feedback_system.get_feedback_analytics(days=7)
        
        self.assertEqual(analytics['total_feedback'], 5)
        self.assertEqual(analytics['positive_feedback'], 3)
        self.assertEqual(analytics['negative_feedback'], 2)
        self.assertEqual(analytics['satisfaction_rate'], 60.0)
        
    def test_improvement_suggestions(self):
        """Test improvement suggestions generation"""
        # Add negative feedback for same query multiple times
        for i in range(3):
            self.feedback_system.record_feedback(
                session_id="test",
                query="What is Docker?",
                response="Docker response",
                feedback_type="negative",
                details="Response was wrong" if i == 0 else "Too slow",
                response_time_ms=6000 if i > 0 else 100
            )
        
        suggestions = self.feedback_system.get_improvement_suggestions(limit=5)
        
        self.assertGreater(len(suggestions), 0)
        self.assertEqual(suggestions[0]['query'], "What is Docker?")
        self.assertEqual(suggestions[0]['negative_count'], 3)
        self.assertIn('Optimize', suggestions[0]['suggested_action'])  # Due to slow response

class TestIntegration(unittest.TestCase):
    """Test integration between Knowledge Graph and Feedback System"""
    
    def test_feedback_affects_retrieval_ranking(self):
        """Test that feedback influences retrieval ranking"""
        # This would test the integration in the main app
        # Checking that responses with better feedback scores rank higher
        pass  # Implementation would require full app context
        
    def test_graph_entities_in_feedback_context(self):
        """Test that graph entities are tracked in feedback context"""
        # This would verify entities from graph appear in feedback metadata
        pass  # Implementation would require full app context

if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)