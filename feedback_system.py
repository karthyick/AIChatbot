"""Real-time Feedback System for AI Responses

This module provides a comprehensive feedback collection and learning system
for improving AI responses based on user feedback.
"""

import sqlite3
import json
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import hashlib
import numpy as np
from collections import defaultdict
import plotly.graph_objects as go
import plotly.express as px

class FeedbackSystem:
    """Manages user feedback collection and response improvement"""
    
    def __init__(self, db_path: str = "feedback.db"):
        """Initialize feedback system with database"""
        self.db_path = db_path
        self.init_database()
        
    def init_database(self):
        """Initialize SQLite database for feedback storage"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create feedback table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                query_text TEXT NOT NULL,
                query_hash TEXT NOT NULL,
                response_text TEXT NOT NULL,
                response_hash TEXT NOT NULL,
                context_used TEXT,
                feedback_type TEXT CHECK(feedback_type IN ('positive', 'negative')) NOT NULL,
                feedback_details TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                user_id TEXT,
                response_time_ms INTEGER,
                model_used TEXT,
                confidence_score REAL
            )
        """)
        
        # Create response quality scores table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS response_scores (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                response_hash TEXT UNIQUE NOT NULL,
                positive_count INTEGER DEFAULT 0,
                negative_count INTEGER DEFAULT 0,
                quality_score REAL DEFAULT 0.5,
                last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create learning improvements table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS improvements (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_pattern TEXT NOT NULL,
                improved_context TEXT NOT NULL,
                improvement_score REAL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                applied BOOLEAN DEFAULT FALSE
            )
        """)
        
        # Create feedback tags table for categorization
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feedback_tags (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                feedback_id INTEGER,
                tag TEXT NOT NULL,
                FOREIGN KEY (feedback_id) REFERENCES feedback (id)
            )
        """)
        
        # Create indices for better performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_query_hash ON feedback(query_hash)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_response_hash ON feedback(response_hash)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON feedback(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_feedback_type ON feedback(feedback_type)")
        
        conn.commit()
        conn.close()
        
    def record_feedback(self,
                       session_id: str,
                       query: str,
                       response: str,
                       feedback_type: str,
                       context_used: Optional[List[str]] = None,
                       details: Optional[str] = None,
                       response_time_ms: Optional[int] = None,
                       model_used: Optional[str] = None,
                       confidence_score: Optional[float] = None) -> bool:
        """Record user feedback for a response"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Generate hashes for deduplication
            query_hash = hashlib.md5(query.encode()).hexdigest()
            response_hash = hashlib.md5(response.encode()).hexdigest()
            
            # Convert context to JSON string
            context_json = json.dumps(context_used) if context_used else None
            
            # Insert feedback
            cursor.execute("""
                INSERT INTO feedback (
                    session_id, query_text, query_hash, response_text, 
                    response_hash, context_used, feedback_type, feedback_details,
                    response_time_ms, model_used, confidence_score
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (session_id, query, query_hash, response, response_hash,
                  context_json, feedback_type, details, response_time_ms,
                  model_used, confidence_score))
            
            feedback_id = cursor.lastrowid
            
            # Update response scores
            self._update_response_score(cursor, response_hash, feedback_type)
            
            # Extract and store tags from details if provided
            if details:
                tags = self._extract_tags(details)
                for tag in tags:
                    cursor.execute(
                        "INSERT INTO feedback_tags (feedback_id, tag) VALUES (?, ?)",
                        (feedback_id, tag)
                    )
            
            conn.commit()
            conn.close()
            
            # Trigger learning if negative feedback
            if feedback_type == 'negative':
                self._trigger_improvement_learning(query, response, context_used)
                
            return True
            
        except Exception as e:
            st.error(f"Error recording feedback: {str(e)}")
            return False
            
    def _update_response_score(self, cursor, response_hash: str, feedback_type: str):
        """Update quality score for a response"""
        # Check if response already has a score
        cursor.execute(
            "SELECT positive_count, negative_count FROM response_scores WHERE response_hash = ?",
            (response_hash,)
        )
        result = cursor.fetchone()
        
        if result:
            pos_count, neg_count = result
            if feedback_type == 'positive':
                pos_count += 1
            else:
                neg_count += 1
                
            # Calculate Wilson score for quality
            quality_score = self._calculate_wilson_score(pos_count, neg_count)
            
            cursor.execute("""
                UPDATE response_scores 
                SET positive_count = ?, negative_count = ?, 
                    quality_score = ?, last_updated = CURRENT_TIMESTAMP
                WHERE response_hash = ?
            """, (pos_count, neg_count, quality_score, response_hash))
        else:
            pos_count = 1 if feedback_type == 'positive' else 0
            neg_count = 1 if feedback_type == 'negative' else 0
            quality_score = self._calculate_wilson_score(pos_count, neg_count)
            
            cursor.execute("""
                INSERT INTO response_scores 
                (response_hash, positive_count, negative_count, quality_score)
                VALUES (?, ?, ?, ?)
            """, (response_hash, pos_count, neg_count, quality_score))
            
    def _calculate_wilson_score(self, positive: int, negative: int) -> float:
        """Calculate Wilson score interval for ranking"""
        n = positive + negative
        if n == 0:
            return 0.5
            
        z = 1.96  # 95% confidence interval
        p = positive / n
        
        denominator = 1 + (z**2) / n
        numerator = p + (z**2) / (2 * n) - z * np.sqrt(
            (p * (1 - p) + (z**2) / (4 * n)) / n
        )
        
        return numerator / denominator
        
    def _extract_tags(self, text: str) -> List[str]:
        """Extract tags from feedback text"""
        # Simple tag extraction - can be enhanced with NLP
        common_tags = [
            'accuracy', 'relevance', 'clarity', 'completeness',
            'speed', 'helpful', 'confusing', 'wrong', 'incomplete'
        ]
        
        text_lower = text.lower()
        tags = [tag for tag in common_tags if tag in text_lower]
        return tags
        
    def _trigger_improvement_learning(self, 
                                     query: str, 
                                     response: str,
                                     context_used: Optional[List[str]]):
        """Trigger improvement learning for negative feedback"""
        # This is where you'd implement actual ML improvement logic
        # For now, we'll store patterns for manual review
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Store improvement suggestion
            cursor.execute("""
                INSERT INTO improvements (query_pattern, improved_context, improvement_score)
                VALUES (?, ?, ?)
            """, (query[:100], json.dumps(context_used), 0.0))
            
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Error in improvement learning: {str(e)}")
            
    def get_response_quality_score(self, response: str) -> float:
        """Get quality score for a specific response"""
        response_hash = hashlib.md5(response.encode()).hexdigest()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT quality_score FROM response_scores WHERE response_hash = ?",
            (response_hash,)
        )
        result = cursor.fetchone()
        conn.close()
        
        return result[0] if result else 0.5
        
    def get_feedback_analytics(self, days: int = 7) -> Dict:
        """Get comprehensive feedback analytics"""
        conn = sqlite3.connect(self.db_path)
        
        # Calculate date threshold
        date_threshold = datetime.now() - timedelta(days=days)
        
        # Get overall statistics
        df = pd.read_sql_query("""
            SELECT * FROM feedback 
            WHERE timestamp >= ?
        """, conn, params=(date_threshold.isoformat(),))
        
        if df.empty:
            conn.close()
            return {
                'total_feedback': 0,
                'satisfaction_rate': 0,
                'daily_feedback': pd.DataFrame(),
                'top_issues': [],
                'response_times': {}
            }
        
        # Calculate metrics
        total = len(df)
        positive = len(df[df['feedback_type'] == 'positive'])
        satisfaction_rate = (positive / total * 100) if total > 0 else 0
        
        # Daily feedback trend
        df['date'] = pd.to_datetime(df['timestamp']).dt.date
        daily_feedback = df.groupby(['date', 'feedback_type']).size().unstack(fill_value=0)
        
        # Most common negative feedback queries
        negative_df = df[df['feedback_type'] == 'negative']
        top_issues = negative_df['query_text'].value_counts().head(5).to_dict()
        
        # Response time analysis
        response_times = {
            'mean': df['response_time_ms'].mean() if 'response_time_ms' in df else 0,
            'median': df['response_time_ms'].median() if 'response_time_ms' in df else 0,
            'p95': df['response_time_ms'].quantile(0.95) if 'response_time_ms' in df else 0
        }
        
        # Get tag distribution
        tag_query = """
            SELECT tag, COUNT(*) as count 
            FROM feedback_tags ft
            JOIN feedback f ON ft.feedback_id = f.id
            WHERE f.timestamp >= ?
            GROUP BY tag
            ORDER BY count DESC
            LIMIT 10
        """
        tags_df = pd.read_sql_query(tag_query, conn, params=(date_threshold.isoformat(),))
        
        conn.close()
        
        return {
            'total_feedback': total,
            'positive_feedback': positive,
            'negative_feedback': total - positive,
            'satisfaction_rate': satisfaction_rate,
            'daily_feedback': daily_feedback,
            'top_issues': top_issues,
            'response_times': response_times,
            'tag_distribution': tags_df.to_dict('records') if not tags_df.empty else []
        }
        
    def get_improvement_suggestions(self, limit: int = 10) -> List[Dict]:
        """Get improvement suggestions based on feedback patterns"""
        conn = sqlite3.connect(self.db_path)
        
        # Get most problematic queries
        query = """
            SELECT 
                query_text,
                COUNT(*) as negative_count,
                AVG(response_time_ms) as avg_response_time,
                GROUP_CONCAT(feedback_details, ' | ') as all_feedback
            FROM feedback
            WHERE feedback_type = 'negative'
            GROUP BY query_text
            ORDER BY negative_count DESC
            LIMIT ?
        """
        
        df = pd.read_sql_query(query, conn, params=(limit,))
        conn.close()
        
        suggestions = []
        for _, row in df.iterrows():
            suggestions.append({
                'query': row['query_text'],
                'negative_count': row['negative_count'],
                'avg_response_time': row['avg_response_time'],
                'feedback_summary': row['all_feedback'],
                'suggested_action': self._suggest_improvement_action(row)
            })
            
        return suggestions
        
    def _suggest_improvement_action(self, feedback_row) -> str:
        """Suggest improvement action based on feedback pattern"""
        if feedback_row['avg_response_time'] > 5000:
            return "Optimize response generation for better performance"
        elif 'wrong' in str(feedback_row['all_feedback']).lower():
            return "Review and update knowledge base for accuracy"
        elif 'incomplete' in str(feedback_row['all_feedback']).lower():
            return "Add more comprehensive information to knowledge base"
        elif 'confusing' in str(feedback_row['all_feedback']).lower():
            return "Simplify response generation and improve clarity"
        else:
            return "Review context retrieval and ranking algorithm"
            
    def export_feedback_data(self, format: str = 'csv') -> str:
        """Export feedback data for analysis"""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query("SELECT * FROM feedback", conn)
        conn.close()
        
        if format == 'csv':
            filename = f"feedback_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df.to_csv(filename, index=False)
            return filename
        elif format == 'json':
            filename = f"feedback_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            df.to_json(filename, orient='records', indent=2)
            return filename
        else:
            raise ValueError(f"Unsupported format: {format}")

def render_feedback_ui(response: str, query: str, session_id: str, 
                       context_used: List[str], feedback_system: FeedbackSystem):
    """Render feedback UI components in Streamlit"""
    col1, col2, col3 = st.columns([1, 1, 3])
    
    with col1:
        if st.button("👍", key=f"thumbs_up_{hash(response)}", help="This response was helpful"):
            success = feedback_system.record_feedback(
                session_id=session_id,
                query=query,
                response=response,
                feedback_type='positive',
                context_used=context_used
            )
            if success:
                st.success("✅ Thank you for your feedback!")
                st.balloons()
                
    with col2:
        if st.button("👎", key=f"thumbs_down_{hash(response)}", help="This response needs improvement"):
            with st.form(key=f"feedback_form_{hash(response)}"):
                feedback_details = st.text_area(
                    "What could be improved?",
                    placeholder="e.g., The answer was incomplete, confusing, or incorrect..."
                )
                
                improvement_tags = st.multiselect(
                    "Select issues (optional)",
                    ["Inaccurate", "Incomplete", "Confusing", "Irrelevant", 
                     "Too slow", "Too verbose", "Too brief", "Outdated"]
                )
                
                if st.form_submit_button("Submit Feedback"):
                    details = feedback_details
                    if improvement_tags:
                        details += f" Tags: {', '.join(improvement_tags)}"
                        
                    success = feedback_system.record_feedback(
                        session_id=session_id,
                        query=query,
                        response=response,
                        feedback_type='negative',
                        context_used=context_used,
                        details=details
                    )
                    
                    if success:
                        st.warning("📨 Thank you! We'll use your feedback to improve.")
                        
    with col3:
        # Show response quality indicator
        quality_score = feedback_system.get_response_quality_score(response)
        
        if quality_score > 0.7:
            st.success(f"⭐ High Quality Response (Score: {quality_score:.2f})")
        elif quality_score > 0.4:
            st.info(f"📊 Average Response (Score: {quality_score:.2f})")
        else:
            st.warning(f"⚠️ Low Quality Response (Score: {quality_score:.2f})")

def render_feedback_dashboard(feedback_system: FeedbackSystem):
    """Render comprehensive feedback analytics dashboard"""
    st.subheader("📊 Feedback Analytics Dashboard")
    
    # Time range selector
    col1, col2 = st.columns([1, 3])
    with col1:
        days = st.selectbox("Time Range", [7, 30, 90, 365], index=0)
        
    # Get analytics
    analytics = feedback_system.get_feedback_analytics(days)
    
    if analytics['total_feedback'] == 0:
        st.info("No feedback data available yet. Start collecting feedback to see analytics!")
        return
        
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Feedback", analytics['total_feedback'])
    col2.metric("Positive", analytics['positive_feedback'], 
                delta=f"+{analytics['positive_feedback']}" if analytics['positive_feedback'] > 0 else "0")
    col3.metric("Negative", analytics['negative_feedback'],
                delta=f"-{analytics['negative_feedback']}" if analytics['negative_feedback'] > 0 else "0")
    col4.metric("Satisfaction Rate", f"{analytics['satisfaction_rate']:.1f}%")
    
    # Daily feedback trend
    if not analytics['daily_feedback'].empty:
        st.write("### 📈 Feedback Trend")
        fig = go.Figure()
        
        if 'positive' in analytics['daily_feedback'].columns:
            fig.add_trace(go.Scatter(
                x=analytics['daily_feedback'].index,
                y=analytics['daily_feedback']['positive'],
                mode='lines+markers',
                name='Positive',
                line=dict(color='green', width=2)
            ))
            
        if 'negative' in analytics['daily_feedback'].columns:
            fig.add_trace(go.Scatter(
                x=analytics['daily_feedback'].index,
                y=analytics['daily_feedback']['negative'],
                mode='lines+markers',
                name='Negative',
                line=dict(color='red', width=2)
            ))
            
        fig.update_layout(
            title="Daily Feedback Count",
            xaxis_title="Date",
            yaxis_title="Count",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    # Top issues
    if analytics['top_issues']:
        st.write("### ⚠️ Top Issues (Negative Feedback)")
        issues_df = pd.DataFrame(
            analytics['top_issues'].items(),
            columns=['Query', 'Count']
        )
        st.dataframe(issues_df, use_container_width=True)
        
    # Tag distribution
    if analytics['tag_distribution']:
        st.write("### 🏷️ Feedback Tag Distribution")
        tags_df = pd.DataFrame(analytics['tag_distribution'])
        fig = px.bar(tags_df, x='tag', y='count', 
                     title="Most Common Feedback Tags")
        st.plotly_chart(fig, use_container_width=True)
        
    # Response time metrics
    if analytics['response_times'] and analytics['response_times']['mean'] > 0:
        st.write("### ⏱️ Response Time Metrics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Average", f"{analytics['response_times']['mean']:.0f} ms")
        col2.metric("Median", f"{analytics['response_times']['median']:.0f} ms")
        col3.metric("95th Percentile", f"{analytics['response_times']['p95']:.0f} ms")
        
    # Improvement suggestions
    st.write("### 💡 Improvement Suggestions")
    suggestions = feedback_system.get_improvement_suggestions(5)
    
    if suggestions:
        for i, suggestion in enumerate(suggestions, 1):
            with st.expander(f"Suggestion {i}: {suggestion['query'][:50]}..."):
                st.write(f"**Negative Count:** {suggestion['negative_count']}")
                st.write(f"**Avg Response Time:** {suggestion['avg_response_time']:.0f} ms")
                st.write(f"**Feedback Summary:** {suggestion['feedback_summary']}")
                st.write(f"**🎯 Suggested Action:** {suggestion['suggested_action']}")
                
    # Export functionality
    st.write("### 📥 Export Data")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("💾 Export to CSV"):
            filename = feedback_system.export_feedback_data('csv')
            st.success(f"Data exported to {filename}")
            
            with open(filename, 'rb') as f:
                st.download_button(
                    "Download CSV",
                    f,
                    file_name=filename,
                    mime='text/csv'
                )
                
    with col2:
        if st.button("📦 Export to JSON"):
            filename = feedback_system.export_feedback_data('json')
            st.success(f"Data exported to {filename}")
            
            with open(filename, 'rb') as f:
                st.download_button(
                    "Download JSON",
                    f,
                    file_name=filename,
                    mime='application/json'
                )