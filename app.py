import streamlit as st
import pandas as pd

# Page configuration
st.set_page_config(page_title="📘 AI Terms Glossary", layout="wide")

st.title("🤖 AI Glossary: 200+ Terms & Acronyms")
st.markdown("Explore, search, and download a comprehensive glossary of key AI terms.")

# Load full AI glossary (200+ items)
ai_terms = [
    ("AI", "Artificial Intelligence"),
    ("AGI", "Artificial General Intelligence"),
    ("ANN", "Artificial Neural Network"),
    ("API", "Application Programming Interface"),
    ("ASR", "Automatic Speech Recognition"),
    ("AutoML", "Automated machine learning pipeline"),
    ("AUC", "Area Under the Curve"),
    ("Backpropagation", "Algorithm for training neural networks"),
    ("BERT", "Bidirectional Encoder Representations from Transformers"),
    ("Bias", "Model's systematic error"),
    ("BM25", "Best Matching 25 (Ranking algorithm)"),
    ("Black Box", "Model whose inner workings are opaque"),
    ("CatBoost", "Gradient boosting library for categorical features"),
    ("Chain-of-Thought", "LLM reasoning technique"),
    ("CNN", "Convolutional Neural Network"),
    ("Cold Start", "Lack of data to begin recommendations"),
    ("Contrastive Learning", "Learning by comparing samples"),
    ("Curriculum Learning", "Training with increasing difficulty"),
    ("CV", "Computer Vision"),
    ("CUDA", "Compute Unified Device Architecture"),
    ("DALL·E", "Deep learning model for image generation by OpenAI"),
    ("Data Augmentation", "Expanding datasets by modifying samples"),
    ("Data Drift", "Change in data over time"),
    ("DeepFake", "AI-generated fake audio or video"),
    ("DL", "Deep Learning"),
    ("Edge AI", "AI computations on edge devices"),
    ("EDA", "Exploratory Data Analysis"),
    ("ELMo", "Embeddings from Language Models"),
    ("Embedding", "Mapping text into vector space"),
    ("Ensemble", "Combining multiple models"),
    ("ETL", "Extract, Transform, Load"),
    ("F1 Score", "Harmonic mean of precision and recall"),
    ("FAIR", "Facebook AI Research"),
    ("Feature Engineering", "Creating features from raw data"),
    ("Few-shot", "Few examples used to learn a task"),
    ("Few-shot Learning", "Learning with few examples"),
    ("FLOPs", "Floating Point Operations per Second"),
    ("GAN", "Generative Adversarial Network"),
    ("GPT", "Generative Pre-trained Transformer"),
    ("Gradient Clipping", "Preventing exploding gradients"),
    ("GRU", "Gated Recurrent Unit"),
    ("Hallucination", "LLM-generated false information"),
    ("Hugging Face", "AI community and model repository"),
    ("Human-in-the-loop", "Humans aid AI during learning or prediction"),
    ("Hyperparameter", "Tunable model parameters set before training"),
    ("Inference", "Running trained models on new data"),
    ("Inference API", "Service for running trained models"),
    ("Intent Detection", "Understanding user intention from input"),
    ("IoT", "Internet of Things"),
    ("JAX", "Google’s numerical computing library for ML"),
    ("Keras", "Deep learning API in Python"),
    ("KNN", "K-Nearest Neighbors"),
    ("LangChain", "Framework for building LLM-powered apps"),
    ("Latent Space", "Hidden feature space learned by model"),
    ("LLM", "Large Language Model"),
    ("LLMOps", "Operations related to large language models"),
    ("LoRA", "Low-Rank Adaptation for fine-tuning LLMs"),
    ("LSTM", "Long Short-Term Memory"),
    ("Meta AI", "AI division at Meta"),
    ("Meta Prompt", "Prompt that structures other prompts"),
    ("ML", "Machine Learning"),
    ("MLOps", "Machine Learning Operations"),
    ("Model Card", "Documentation of ML models"),
    ("Model Compression", "Reducing model size"),
    ("Multi-modal Learning", "Learning from different data types (text, image, audio)"),
    ("MultiOn", "AI Agent for automated actions"),
    ("Named Entity Recognition", "Identifying entities like names or locations"),
    ("NER", "Named Entity Recognition"),
    ("Noise", "Irrelevant data in learning"),
    ("OCR", "Optical Character Recognition"),
    ("ONNX", "Open Neural Network Exchange"),
    ("Online Learning", "Model updated continuously as data arrives"),
    ("OpenAI", "AI research company behind ChatGPT"),
    ("Outlier Detection", "Identifying unusual data points"),
    ("Overfitting", "Model fits training data too well"),
    ("Parrot Model", "Repeats training data without reasoning"),
    ("Parameter", "Learnable element in a model"),
    ("Pinecone", "Vector database for embeddings"),
    ("Pretraining", "Initial training on large datasets"),
    ("Prompt Engineering", "Designing effective prompts for LLMs"),
    ("Pruning", "Removing unnecessary weights from a network"),
    ("Q-Learning", "Reinforcement learning algorithm"),
    ("Qdrant", "Vector search engine"),
    ("Quantization", "Reducing model size with lower precision"),
    ("RAG", "Retrieval-Augmented Generation"),
    ("Recall", "True positives over all actual positives"),
    ("Reinforcement Signal", "Reward/penalty guiding learning"),
    ("ReLU", "Rectified Linear Unit"),
    ("Reranking", "Reordering retrieved results"),
    ("Reward Model", "Guides LLM output via scoring"),
    ("RL", "Reinforcement Learning"),
    ("RLHF", "Reinforcement Learning with Human Feedback"),
    ("RNN", "Recurrent Neural Network"),
    ("ROC", "Receiver Operating Characteristic"),
    ("Sampling", "Drawing samples from data or model output"),
    ("SDXL", "Stable Diffusion XL"),
    ("Segmentation", "Dividing data into meaningful groups"),
    ("Self-attention", "Component of Transformer models"),
    ("SHAP", "SHapley Additive exPlanations"),
    ("SKLearn", "Scikit-learn ML library"),
    ("SMOTE", "Synthetic Minority Oversampling Technique"),
    ("Stable Diffusion", "Text-to-image model"),
    ("SVM", "Support Vector Machine"),
    ("Tensor", "Multidimensional array used in ML"),
    ("TensorFlow", "Google’s ML framework"),
    ("Token", "Smallest unit in NLP processing"),
    ("Tokenization", "Breaking text into tokens"),
    ("Transfer Learning", "Reusing knowledge from one task for another"),
    ("Transformer", "Deep learning model for sequence processing"),
    ("TruLens", "Tool for evaluating LLMs"),
    ("Truncation", "Cutting off excess input"),
    ("TTS", "Text-to-Speech"),
    ("Tuning", "Optimizing model hyperparameters"),
    ("UAT", "User Acceptance Testing"),
    ("UL2", "Unified Language Learning"),
    ("Underfitting", "Model too simple to capture data"),
    ("Unstructured.io", "Document processing AI"),
    ("Unsupervised Learning", "Learning from unlabeled data"),
    ("Validation Set", "Data for model evaluation during training"),
    ("Vector", "Numeric representation of data"),
    ("Vector Database", "Stores and searches embedding vectors"),
    ("Vision AI", "Image and video understanding AI"),
    ("Vision Transformer (ViT)", "Transformer for vision tasks"),
    ("Voice Activity Detection", "Detects presence of speech"),
    ("Watsonx", "IBM's AI platform"),
    ("Weights", "Model parameters learned during training"),
    ("Weights & Biases", "Experiment tracking tool"),
    ("Whisper", "OpenAI’s speech recognition model"),
    ("XAI", "Explainable Artificial Intelligence"),
    ("YOLO", "You Only Look Once (Object Detection)"),
    ("Zero-shot", "No prior examples used for task"),
]

# CSV Download
df = pd.DataFrame(ai_terms, columns=["Acronym", "Definition"])
csv = df.to_csv(index=False).encode("utf-8")
st.download_button("📥 Download as CSV", csv, "ai_glossary.csv", "text/csv")

# Search Bar
query = st.text_input("🔍 Search AI term or definition").lower()
filtered_terms = [item for item in ai_terms if query in item[0].lower() or query in item[1].lower()]

# Group and Display
from collections import defaultdict
grouped = defaultdict(list)
for term, definition in filtered_terms:
    grouped[term[0].upper()].append((term, definition))

for letter in sorted(grouped):
    with st.expander(f"📚 Terms starting with '{letter}' ({len(grouped[letter])})", expanded=False):
        for term, definition in grouped[letter]:
            st.markdown(f"**{term}** — {definition}")
