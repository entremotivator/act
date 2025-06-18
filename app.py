import streamlit as st
import pandas as pd

st.set_page_config(page_title="100 AI Terms & Acronyms", layout="wide")
st.title("🤖 100 AI Terms & Acronyms")

# Sample data: AI terms with definitions
ai_terms = [
    ("AI", "Artificial Intelligence"),
    ("AGI", "Artificial General Intelligence"),
    ("ANN", "Artificial Neural Network"),
    ("API", "Application Programming Interface"),
    ("ASR", "Automatic Speech Recognition"),
    ("BERT", "Bidirectional Encoder Representations from Transformers"),
    ("CNN", "Convolutional Neural Network"),
    ("CV", "Computer Vision"),
    ("DALL·E", "Deep learning model for image generation by OpenAI"),
    ("DL", "Deep Learning"),
    ("EDA", "Exploratory Data Analysis"),
    ("GAN", "Generative Adversarial Network"),
    ("GPT", "Generative Pre-trained Transformer"),
    ("IoT", "Internet of Things"),
    ("KNN", "K-Nearest Neighbors"),
    ("LSTM", "Long Short-Term Memory"),
    ("ML", "Machine Learning"),
    ("NLP", "Natural Language Processing"),
    ("OCR", "Optical Character Recognition"),
    ("RL", "Reinforcement Learning"),
    ("RNN", "Recurrent Neural Network"),
    ("ROC", "Receiver Operating Characteristic"),
    ("SVM", "Support Vector Machine"),
    ("TTS", "Text-to-Speech"),
    ("TF-IDF", "Term Frequency-Inverse Document Frequency"),
    ("UAT", "User Acceptance Testing"),
    ("XAI", "Explainable Artificial Intelligence"),
    ("YOLO", "You Only Look Once (Object Detection)"),
    ("AUC", "Area Under the Curve"),
    ("BM25", "Best Matching 25 (Ranking algorithm)"),
    ("CUDA", "Compute Unified Device Architecture"),
    ("EDA", "Exploratory Data Analysis"),
    ("ELMo", "Embeddings from Language Models"),
    ("ETL", "Extract, Transform, Load"),
    ("F1 Score", "Harmonic mean of precision and recall"),
    ("FAIR", "Facebook AI Research"),
    ("FLOPs", "Floating Point Operations per Second"),
    ("GRU", "Gated Recurrent Unit"),
    ("Hugging Face", "AI community and model repository"),
    ("Hyperparameter", "Tunable model parameters set before training"),
    ("Inference", "Running trained models on new data"),
    ("JAX", "Google’s numerical computing library for ML"),
    ("Keras", "Deep learning API in Python"),
    ("LangChain", "Framework for building LLM-powered apps"),
    ("LLM", "Large Language Model"),
    ("LoRA", "Low-Rank Adaptation for fine-tuning LLMs"),
    ("Meta AI", "AI division at Meta"),
    ("MLOps", "Machine Learning Operations"),
    ("MultiOn", "AI Agent for automated actions"),
    ("NER", "Named Entity Recognition"),
    ("ONNX", "Open Neural Network Exchange"),
    ("OpenAI", "AI research company behind ChatGPT"),
    ("Pinecone", "Vector database for embeddings"),
    ("Prompt Engineering", "Designing effective prompts for LLMs"),
    ("Q-Learning", "Reinforcement learning algorithm"),
    ("Qdrant", "Vector search engine"),
    ("ReLU", "Rectified Linear Unit"),
    ("RAG", "Retrieval-Augmented Generation"),
    ("SDXL", "Stable Diffusion XL"),
    ("Segmentation", "Dividing data into meaningful groups"),
    ("SHAP", "SHapley Additive exPlanations"),
    ("SKLearn", "Scikit-learn ML library"),
    ("SMOTE", "Synthetic Minority Oversampling Technique"),
    ("Stable Diffusion", "Text-to-image model"),
    ("Tensor", "Multidimensional array used in ML"),
    ("TensorFlow", "Google’s ML framework"),
    ("Token", "Smallest unit in NLP processing"),
    ("Transformer", "Deep learning model for sequence processing"),
    ("TruLens", "Tool for evaluating LLMs"),
    ("Tuning", "Optimizing model hyperparameters"),
    ("UL2", "Unified Language Learning"),
    ("Unstructured.io", "Document processing AI"),
    ("Vector", "Numeric representation of data"),
    ("Vector Database", "Stores and searches embedding vectors"),
    ("Vision AI", "Image and video understanding AI"),
    ("Watsonx", "IBM's AI platform"),
    ("Weights & Biases", "Experiment tracking tool"),
    ("Zero-shot", "No prior examples used for task"),
    ("Few-shot", "Few examples used to learn a task"),
    ("Chain-of-Thought", "LLM reasoning technique"),
    ("Embedding", "Mapping text into vector space"),
    ("Fine-tuning", "Training a pre-trained model for specific tasks"),
    ("Hallucination", "LLM-generated false information"),
    ("Inference API", "Service for running trained models"),
    ("LLMOps", "Operations related to large language models"),
    ("Meta Prompt", "Prompt that structures other prompts"),
    ("Model Card", "Documentation of ML models"),
    ("Noise", "Irrelevant data in learning"),
    ("Overfitting", "Model fits training data too well"),
    ("Parrot Model", "Repeats training data without reasoning"),
    ("Pretraining", "Initial training on large datasets"),
    ("Quantization", "Reducing model size with lower precision"),
    ("Recall", "True positives over all actual positives"),
    ("Reranking", "Reordering retrieved results"),
    ("Reward Model", "Guides LLM output via scoring"),
    ("RLHF", "Reinforcement Learning with Human Feedback"),
    ("Self-attention", "Component of Transformer models"),
    ("Tokenization", "Breaking text into tokens"),
    ("Truncation", "Cutting off excess input"),
    ("Unsupervised Learning", "Learning from unlabeled data"),
    ("Vision Transformer (ViT)", "Transformer for vision tasks"),
    ("Weights", "Model parameters learned during training"),
]

# Convert to DataFrame
df = pd.DataFrame(ai_terms, columns=["Term", "Definition"])

# Search filter
search = st.text_input("🔍 Search Terms or Acronyms").lower()
if search:
    df = df[df["Term"].str.lower().str.contains(search) | df["Definition"].str.lower().str.contains(search)]

# Display in columns
cols = st.columns(2)
half = len(df) // 2 + len(df) % 2
for i, row in enumerate(df.itertuples()):
    col = cols[0] if i < half else cols[1]
    with col:
        st.markdown(f"**{row.Term}**")
        st.caption(row.Definition)
