#!/usr/bin/env python
"""Setup script for ChatBot V2 Enhanced Features

This script automates the installation and configuration of the
Knowledge Graph Visualization and Feedback System features.
"""

import os
import sys
import subprocess
import platform
import argparse
from pathlib import Path

def check_python_version():
    """Check if Python version is 3.8+"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python 3.8+ required. Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version}")
    return True

def check_cuda_availability():
    """Check if CUDA is available"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
            return True
        else:
            print("⚠️ CUDA not available. CPU will be used (slower performance)")
            return False
    except ImportError:
        print("⚠️ PyTorch not installed yet")
        return False

def install_requirements():
    """Install required packages"""
    print("\n📦 Installing requirements...")
    
    requirements = [
        "streamlit",
        "langchain",
        "langchain-community",
        "chromadb",
        "transformers",
        "torch",
        "accelerate",
        "sentence-transformers",
        "redis",
        "google-search-results",
        "spacy",
        "nltk",
        "networkx",
        "scipy",
        "scikit-learn",
        "numpy",
        "pandas",
        "pyvis",
        "plotly"
    ]
    
    for package in requirements:
        print(f"Installing {package}...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", package],
            capture_output=True
        )
    
    print("✅ All packages installed")

def download_models():
    """Download required language models"""
    print("\n🤖 Downloading language models...")
    
    # Download spaCy model
    print("Downloading spaCy English model...")
    subprocess.run(
        [sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
        capture_output=True
    )
    
    # Download NLTK data
    print("Downloading NLTK data...")
    import nltk
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    
    print("✅ Language models downloaded")

def setup_directories():
    """Create necessary directories"""
    print("\n📁 Setting up directories...")
    
    directories = [
        "chroma_db",
        "logs",
        "exports",
        "backups"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"Created: {directory}/")
    
    print("✅ Directories created")

def create_config_file():
    """Create default configuration file"""
    print("\n⚙️ Creating configuration file...")
    
    config = """# ChatBot V2 Enhanced Configuration

# Model Settings
MODEL_NAME = "facebook/opt-1.3b"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEVICE = "cuda"  # or "cpu"

# Chunking Settings
CHUNK_SIZE = 500
CHUNK_OVERLAP = 300
NUM_CLUSTERS = 7

# Retrieval Settings
RETRIEVAL_K = 5

# Generation Settings
TEMPERATURE = 0.7
TOP_K = 50
TOP_P = 0.95
MAX_TOKENS = 300
REPETITION_PENALTY = 1.2

# Knowledge Graph Settings
GRAPH_FILTER_THRESHOLD = 1
GRAPH_PHYSICS_ENABLED = True
MAX_GRAPH_NODES = 10000

# Feedback System Settings
FEEDBACK_DB_PATH = "feedback.db"
WILSON_SCORE_Z = 1.96  # 95% confidence

# API Keys (update with your keys)
SERPAPI_KEY = "your_api_key_here"

# Paths
CHROMA_DB_PATH = "./chroma_db"
LOGS_PATH = "./logs"
EXPORTS_PATH = "./exports"
"""
    
    with open("config.py", "w") as f:
        f.write(config)
    
    print("✅ Configuration file created: config.py")
    print("⚠️ Remember to update API keys in config.py")

def test_installation():
    """Test if everything is installed correctly"""
    print("\n🧪 Testing installation...")
    
    tests_passed = True
    
    # Test imports
    try:
        import streamlit
        print("✅ Streamlit imported")
    except ImportError:
        print("❌ Failed to import Streamlit")
        tests_passed = False
    
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        print("✅ spaCy model loaded")
    except:
        print("❌ Failed to load spaCy model")
        tests_passed = False
    
    try:
        from knowledge_graph_viz import KnowledgeGraphVisualizer
        print("✅ Knowledge Graph module imported")
    except ImportError:
        print("❌ Failed to import Knowledge Graph module")
        tests_passed = False
    
    try:
        from feedback_system import FeedbackSystem
        print("✅ Feedback System module imported")
    except ImportError:
        print("❌ Failed to import Feedback System module")
        tests_passed = False
    
    try:
        import torch
        import transformers
        print("✅ PyTorch and Transformers imported")
    except ImportError:
        print("❌ Failed to import PyTorch/Transformers")
        tests_passed = False
    
    return tests_passed

def run_sample_test():
    """Run a simple test of the features"""
    print("\n🚀 Running sample test...")
    
    try:
        # Test Knowledge Graph
        import spacy
        from knowledge_graph_viz import KnowledgeGraphVisualizer
        
        nlp = spacy.load("en_core_web_sm")
        kg_viz = KnowledgeGraphVisualizer(nlp)
        
        test_text = "Microsoft acquired GitHub for $7.5 billion in 2018."
        graph = kg_viz.extract_enhanced_relationships(test_text, "test")
        
        print(f"✅ Knowledge Graph: {graph.number_of_nodes()} entities, {graph.number_of_edges()} relationships")
        
        # Test Feedback System
        from feedback_system import FeedbackSystem
        import tempfile
        
        with tempfile.NamedTemporaryFile(suffix='.db') as tmp:
            feedback_sys = FeedbackSystem(db_path=tmp.name)
            
            success = feedback_sys.record_feedback(
                session_id="test",
                query="Test query",
                response="Test response",
                feedback_type="positive"
            )
            
            if success:
                print("✅ Feedback System: Successfully recorded feedback")
            else:
                print("❌ Feedback System: Failed to record feedback")
        
        return True
        
    except Exception as e:
        print(f"❌ Sample test failed: {str(e)}")
        return False

def download_sample_data():
    """Create sample data for testing"""
    print("\n📝 Creating sample data...")
    
    sample_text = """Artificial Intelligence and Technology Companies

OpenAI, founded in 2015 by Elon Musk, Sam Altman, and others, has revolutionized 
the field of artificial intelligence with GPT models. The company is based in 
San Francisco, California.

Microsoft invested $10 billion in OpenAI in 2023, strengthening their partnership. 
Microsoft's CEO Satya Nadella has been a strong advocate for AI integration across 
their product suite including Azure, Office, and Bing.

Google, through its DeepMind subsidiary acquired in 2014 for $500 million, has been 
competing in the AI race with models like Bard and PaLM. Google's CEO Sundar Pichai 
announced major AI initiatives at Google I/O 2023.

Meta (formerly Facebook), led by Mark Zuckerberg, released their LLaMA models as 
open-source contributions to the AI community. Meta's AI Research lab (FAIR) is 
located in Menlo Park, California.

Amazon Web Services (AWS) provides cloud computing infrastructure that powers many 
AI applications. Andy Jassy, CEO of Amazon, has emphasized AI and machine learning 
as key growth areas.

Tesla, under Elon Musk's leadership, uses AI for autonomous driving technology. 
Their Full Self-Driving (FSD) system uses neural networks trained on millions of 
miles of driving data.

Apple, led by Tim Cook, has been more reserved in the AI race but uses machine 
learning extensively in products like Siri, Face ID, and computational photography 
in iPhones."""
    
    with open("sample_knowledge.txt", "w") as f:
        f.write(sample_text)
    
    print("✅ Created sample_knowledge.txt")
    print("   Use this file to test the knowledge upload feature")

def main():
    """Main setup function"""
    parser = argparse.ArgumentParser(description="Setup ChatBot V2 Enhanced Features")
    parser.add_argument("--skip-install", action="store_true", help="Skip package installation")
    parser.add_argument("--skip-models", action="store_true", help="Skip model downloads")
    parser.add_argument("--test-only", action="store_true", help="Only run tests")
    args = parser.parse_args()
    
    print("""
╔══════════════════════════════════════════════════════════╗
║     ChatBot V2 Enhanced Features Setup Script           ║
║                                                          ║
║  Features:                                               ║
║  • Dynamic Knowledge Graph Visualization                ║
║  • Real-time Feedback System                           ║
╚══════════════════════════════════════════════════════════╝
    """)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    if args.test_only:
        # Only run tests
        if test_installation():
            run_sample_test()
        sys.exit(0)
    
    # Check CUDA
    cuda_available = check_cuda_availability()
    
    # Install requirements
    if not args.skip_install:
        install_requirements()
    
    # Download models
    if not args.skip_models:
        download_models()
    
    # Setup directories
    setup_directories()
    
    # Create config file
    create_config_file()
    
    # Create sample data
    download_sample_data()
    
    # Test installation
    if test_installation():
        print("\n✅ Setup completed successfully!")
        
        # Run sample test
        run_sample_test()
        
        print("""
╔══════════════════════════════════════════════════════════╗
║                    NEXT STEPS                           ║
╠══════════════════════════════════════════════════════════╣
║  1. Update API keys in config.py                        ║
║  2. Run the application:                                ║
║     streamlit run app_enhanced.py                       ║
║  3. Upload sample_knowledge.txt to test                 ║
║  4. Explore the Knowledge Graph visualization           ║
║  5. Try the feedback system with test queries          ║
╚══════════════════════════════════════════════════════════╝
        """)
    else:
        print("\n❌ Setup failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()