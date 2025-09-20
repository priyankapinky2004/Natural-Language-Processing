import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import spacy
from textblob import TextBlob
import re
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
    
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Load spaCy model (fallback if not installed)
@st.cache_resource
def load_spacy_model():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        st.error("spaCy English model not found. Please install with: python -m spacy download en_core_web_sm")
        return None

# Set page config
st.set_page_config(
    page_title="Interactive NLP Learning Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
    }
    
    .section-header {
        font-size: 2rem;
        color: #2e86ab;
        border-bottom: 2px solid #2e86ab;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    
    .concept-box {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    
    .demo-box {
        background-color: #f9f9f9;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #ddd;
        margin: 1rem 0;
    }
    
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Main title
    st.markdown('<h1 class="main-header">🤖 Interactive NLP Learning Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("### Master's Level Natural Language Processing Concepts & Demonstrations")
    def show_embeddings():
        st.markdown('<h2 class="section-header">🎯 Word Embeddings Visualization</h2>', unsafe_allow_html=True)
    # Sidebar navigation
    st.sidebar.markdown("## 📚 Navigation")
    sections = {
        "🏠 Introduction": "intro",
        "🔧 Text Preprocessing": "preprocessing", 
        "🎯 Word Embeddings": "embeddings",
        "💭 Sentiment Analysis": "sentiment",
        "📝 Text Summarization": "summarization",
        "🏷️ Named Entity Recognition": "ner",
        "🧠 Attention Mechanisms": "attention"
    }
    
    selected_section = st.sidebar.selectbox("Choose a section:", list(sections.keys()))
    section_key = sections[selected_section]
    
    # Progress bar
    section_progress = {
        "intro": 1/7, "preprocessing": 2/7, "embeddings": 3/7, 
        "sentiment": 4/7, "summarization": 5/7, "ner": 6/7, "attention": 7/7
    }
    st.sidebar.progress(section_progress[section_key])
    
    # Section routing
    if section_key == "intro":
        show_introduction()
    elif section_key == "preprocessing":
        show_preprocessing()
    elif section_key == "embeddings":
        show_embeddings()
    elif section_key == "sentiment":
        show_sentiment_analysis()
    elif section_key == "summarization":
        show_summarization()
    elif section_key == "ner":
        show_ner()
    elif section_key == "attention":
        show_attention()

def show_introduction():
    st.markdown('<h2 class="section-header">🏠 Introduction to Natural Language Processing</h2>', unsafe_allow_html=True)
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-container"><h3>4.5B+</h3><p>Words processed daily by NLP systems</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-container"><h3>175B</h3><p>Parameters in GPT-3</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-container"><h3>2017</h3><p>Transformer architecture introduced</p></div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-container"><h3>96%+</h3><p>Human-level accuracy on reading comprehension</p></div>', unsafe_allow_html=True)
    
    # Core concepts
    st.markdown('<div class="concept-box">', unsafe_allow_html=True)
    st.markdown("## 🎯 Core NLP Pipeline")
    st.markdown("""
    **Natural Language Processing** transforms human language into mathematical representations that machines can understand and manipulate. The modern NLP pipeline consists of several interconnected stages:
    
    1. **Tokenization**: Breaking text into meaningful units (words, subwords, characters)
    2. **Preprocessing**: Cleaning and normalizing text data
    3. **Representation**: Converting tokens to numerical vectors (embeddings)
    4. **Processing**: Applying neural networks or statistical models
    5. **Output**: Generating human-interpretable results
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Interactive concept explorer
    st.subheader("🔍 Concept Deep Dive")
    concept_choice = st.selectbox("Explore NLP concepts:", [
        "Tokenization Strategies",
        "Word Embeddings Evolution", 
        "Attention Mechanisms",
        "Transfer Learning in NLP"
    ])
    
    if concept_choice == "Tokenization Strategies":
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Word-level Tokenization")
            sample_text = "The quick brown fox jumps over the lazy dog."
            tokens = sample_text.split()
            st.code(f"Input: {sample_text}\nTokens: {tokens}")
            st.markdown("**Pros**: Intuitive, preserves semantic meaning  \n**Cons**: Large vocabulary, OOV problems")
        
        with col2:
            st.markdown("### Subword Tokenization (BPE)")
            st.code("Input: 'unhappiness'\nSubwords: ['un', 'happy', 'ness']")
            st.markdown("**Pros**: Handles rare words, smaller vocabulary  \n**Cons**: May break semantic units")
    
    elif concept_choice == "Word Embeddings Evolution":
        # Create embedding evolution timeline
        fig = go.Figure()
        
        years = [2003, 2008, 2013, 2018, 2019, 2020]
        models = ["Neural LM", "Word2Vec", "GloVe", "BERT", "GPT-2", "GPT-3"]
        performance = [30, 65, 70, 85, 88, 92]
        
        fig.add_trace(go.Scatter(
            x=years, y=performance,
            mode='lines+markers+text',
            text=models,
            textposition="top center",
            line=dict(color='#1f77b4', width=3),
            marker=dict(size=12)
        ))
        
        fig.update_layout(
            title="Evolution of Word Embeddings",
            xaxis_title="Year",
            yaxis_title="Performance Score",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    elif concept_choice == "Attention Mechanisms":
        st.markdown("### Self-Attention Visualization")
        # Create attention heatmap
        words = ["The", "cat", "sat", "on", "the", "mat"]
        attention_weights = np.array([
            [0.8, 0.1, 0.0, 0.0, 0.1, 0.0],
            [0.1, 0.7, 0.1, 0.0, 0.0, 0.1],
            [0.0, 0.3, 0.5, 0.1, 0.0, 0.1],
            [0.0, 0.0, 0.2, 0.6, 0.1, 0.1],
            [0.1, 0.0, 0.0, 0.1, 0.7, 0.1],
            [0.0, 0.1, 0.1, 0.1, 0.1, 0.6]
        ])
        
        fig = px.imshow(attention_weights, 
                       x=words, y=words,
                       color_continuous_scale='Blues',
                       title="Self-Attention Weights Matrix")
        st.plotly_chart(fig, use_container_width=True)
    
    elif concept_choice == "Transfer Learning in NLP":
        st.markdown("### Transfer Learning Paradigm")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **Pre-training Phase:**
            - Large corpus (Books, Wikipedia, Web)
            - Self-supervised learning
            - Language modeling objective
            - Learn general language representations
            """)
        with col2:
            st.markdown("""
            **Fine-tuning Phase:**
            - Task-specific dataset
            - Supervised learning
            - Classification/Generation objective
            - Adapt to specific domain
            """)

def show_preprocessing():
    st.markdown('<h2 class="section-header">🔧 Text Preprocessing Laboratory</h2>', unsafe_allow_html=True)
    
    # Interactive preprocessing demo
    st.markdown('<div class="demo-box">', unsafe_allow_html=True)
    st.subheader("📝 Interactive Text Preprocessing")
    
    # Sample texts for demo
    sample_texts = {
        "News Article": """The COVID-19 pandemic has dramatically changed how we work, learn, and interact. Many companies have adopted remote work policies, leading to a 40% increase in productivity for some sectors. However, the social isolation has also raised concerns about mental health among workers.""",
        "Social Media": """OMG!!! Just watched the new Marvel movie 🎬✨ It was AMAZING!!! Can't wait for the sequel... #Marvel #MovieNight #Excited""",
        "Academic Text": """Natural Language Processing (NLP) encompasses a wide range of computational techniques for analyzing and generating human language. Recent advances in transformer architectures, particularly the introduction of attention mechanisms, have revolutionized the field.""",
        "Custom": ""
    }
    
    text_choice = st.selectbox("Choose sample text or enter custom:", list(sample_texts.keys()))
    
    if text_choice == "Custom":
        input_text = st.text_area("Enter your text:", height=150, placeholder="Type or paste your text here...")
    else:
        input_text = st.text_area("Text to preprocess:", value=sample_texts[text_choice], height=150)
    
    if input_text:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🎛️ Preprocessing Options")
            lowercase = st.checkbox("Convert to lowercase", value=True)
            remove_punctuation = st.checkbox("Remove punctuation", value=True)
            remove_stopwords = st.checkbox("Remove stopwords", value=True)
            remove_numbers = st.checkbox("Remove numbers", value=False)
            apply_stemming = st.checkbox("Apply stemming", value=False)
            apply_lemmatization = st.checkbox("Apply lemmatization", value=True)
        
        with col2:
            st.markdown("### 📊 Processing Statistics")
            
            # Original text stats
            original_tokens = word_tokenize(input_text)
            original_chars = len(input_text)
            original_words = len(original_tokens)
            
            # Apply preprocessing
            processed_text = input_text
            
            if lowercase:
                processed_text = processed_text.lower()
            
            if remove_punctuation:
                processed_text = re.sub(r'[^\w\s]', '', processed_text)
            
            tokens = word_tokenize(processed_text)
            
            if remove_numbers:
                tokens = [token for token in tokens if not token.isdigit()]
            
            if remove_stopwords:
                stop_words = set(stopwords.words('english'))
                tokens = [token for token in tokens if token.lower() not in stop_words]
            
            if apply_stemming:
                stemmer = PorterStemmer()
                tokens = [stemmer.stem(token) for token in tokens]
            
            if apply_lemmatization:
                lemmatizer = WordNetLemmatizer()
                tokens = [lemmatizer.lemmatize(token) for token in tokens]
            
            processed_text_final = ' '.join(tokens)
            
            # Stats comparison
            stats_df = pd.DataFrame({
                'Metric': ['Characters', 'Words', 'Unique Words'],
                'Original': [original_chars, original_words, len(set(original_tokens))],
                'Processed': [len(processed_text_final), len(tokens), len(set(tokens))],
                'Reduction %': [
                    round((1 - len(processed_text_final)/original_chars)*100, 1),
                    round((1 - len(tokens)/original_words)*100, 1),
                    round((1 - len(set(tokens))/len(set(original_tokens)))*100, 1)
                ]
            })
            st.dataframe(stats_df, use_container_width=True)
        
        # Results display
        st.markdown("### 🔄 Before vs After Comparison")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Original Text:**")
            st.text_area("", value=input_text, height=100, disabled=True, key="original")
        
        with col2:
            st.markdown("**Processed Text:**")
            st.text_area("", value=processed_text_final, height=100, disabled=True, key="processed")
        
        # Word cloud visualization
        if len(tokens) > 5:
            st.markdown("### ☁️ Word Cloud Visualization")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Original Text**")
                wordcloud_orig = WordCloud(width=400, height=200, background_color='white').generate(input_text)
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.imshow(wordcloud_orig, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)
            
            with col2:
                st.markdown("**Processed Text**")
                wordcloud_proc = WordCloud(width=400, height=200, background_color='white').generate(processed_text_final)
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.imshow(wordcloud_proc, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Educational content
    st.markdown('<div class="concept-box">', unsafe_allow_html=True)
    st.markdown("## 🎓 Teaching Notes: Sentiment Analysis")
    st.markdown("""
    **Key Learning Objectives:**
    
    1. **Lexicon-based Approaches**: Dictionary-based sentiment scoring (VADER, TextBlob)
    2. **Machine Learning Approaches**: Feature engineering + classification algorithms
    3. **Deep Learning Approaches**: RNNs, CNNs, Transformers for sentiment classification
    4. **Challenges in Sentiment Analysis**:
       - Sarcasm and irony detection
       - Context-dependent sentiment
       - Domain adaptation
       - Aspect-based sentiment analysis
    
    **Technical Discussion Points:**
    - Polarity vs. subjectivity: What's the difference?
    - Fine-grained sentiment (1-5 stars) vs. binary classification
    - Handling negation: "not good" vs "good"
    - Multi-lingual sentiment analysis challenges
    - Bias in sentiment models and fairness considerations
    
    **Real-world Applications:**
    - Social media monitoring
    - Customer feedback analysis
    - Financial market sentiment
    - Product review analysis
    """)
    st.markdown('</div>', unsafe_allow_html=True)

def show_summarization():
    st.markdown('<h2 class="section-header">📝 Text Summarization Workshop</h2>', unsafe_allow_html=True)
    
    st.markdown('<div class="demo-box">', unsafe_allow_html=True)
    st.subheader("📄 Interactive Text Summarizer")
    
    # Sample articles for summarization
    sample_articles = {
        "Climate Change Article": """
        Climate change represents one of the most pressing challenges of our time, with far-reaching consequences for ecosystems, human societies, and the global economy. The phenomenon is primarily driven by the increased concentration of greenhouse gases in the atmosphere, particularly carbon dioxide from fossil fuel combustion.
        
        Scientific evidence overwhelmingly supports the conclusion that human activities are the dominant driver of observed climate change since the mid-20th century. Global average temperatures have risen by approximately 1.1 degrees Celsius above pre-industrial levels, leading to widespread environmental changes.
        
        The impacts of climate change are already visible across the globe. Arctic sea ice is shrinking at an unprecedented rate, with some projections suggesting ice-free Arctic summers within decades. Sea levels are rising due to thermal expansion of oceans and melting ice sheets, threatening coastal communities worldwide.
        
        Extreme weather events are becoming more frequent and severe. Heat waves, droughts, floods, and storms are occurring with greater intensity, causing significant economic damage and human suffering. Agricultural systems are under stress, with changing precipitation patterns affecting crop yields.
        
        Addressing climate change requires urgent and coordinated global action. The Paris Agreement, adopted in 2015, represents an international commitment to limit global warming to well below 2 degrees Celsius. However, current national commitments are insufficient to meet this goal.
        
        Solutions include transitioning to renewable energy sources, improving energy efficiency, implementing carbon pricing mechanisms, and protecting natural carbon sinks like forests. Individual actions, while important, must be complemented by systemic changes in policy, technology, and economic structures.
        """,
        
        "AI Technology Article": """
        Artificial Intelligence (AI) has emerged as a transformative technology that is reshaping industries, society, and our daily lives. From recommendation systems that curate our entertainment to autonomous vehicles navigating city streets, AI applications have become ubiquitous in the modern world.
        
        Machine learning, a subset of AI, enables computers to learn patterns from data without being explicitly programmed for every scenario. Deep learning, utilizing neural networks with multiple layers, has achieved remarkable breakthroughs in image recognition, natural language processing, and game playing.
        
        The recent success of large language models like GPT-3 and ChatGPT has democratized access to sophisticated AI capabilities. These models can generate human-like text, answer questions, write code, and assist with creative tasks, marking a significant milestone in AI development.
        
        However, the rapid advancement of AI also raises important ethical and societal concerns. Issues include algorithmic bias, privacy protection, job displacement, and the concentration of AI capabilities in the hands of a few large corporations. There are ongoing debates about AI safety and the potential risks of artificial general intelligence.
        
        Regulation and governance of AI systems present complex challenges for policymakers. Balancing innovation with protection of individual rights and societal values requires careful consideration and international cooperation.
        
        Looking forward, AI is expected to continue advancing rapidly, with potential applications in healthcare, education, scientific research, and environmental sustainability. The key challenge will be ensuring that AI development and deployment serve the broader public interest while minimizing potential harms.
        """,
        
        "Custom": ""
    }
    
    article_choice = st.selectbox("Choose sample article or enter custom:", list(sample_articles.keys()))
    
    if article_choice == "Custom":
        input_text = st.text_area("Enter article to summarize:", height=200, placeholder="Paste your article here...")
    else:
        input_text = st.text_area("Article text:", value=sample_articles[article_choice], height=200)
    
    if input_text and len(input_text.split()) > 50:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### ⚙️ Summarization Settings")
            summary_type = st.radio("Summarization Type:", ["Extractive", "Abstractive (Simulated)"])
            summary_length = st.slider("Summary Length (% of original):", 10, 50, 25)
            focus_area = st.selectbox("Focus Area:", ["General", "Key Statistics", "Main Arguments", "Conclusions"])
        
        with col2:
            st.markdown("### 📊 Document Statistics")
            sentences = sent_tokenize(input_text)
            words = word_tokenize(input_text)
            
            stats = {
                "Total Sentences": len(sentences),
                "Total Words": len(words),
                "Average Words/Sentence": round(len(words) / len(sentences), 1),
                "Target Summary Length": f"{int(len(words) * summary_length / 100)} words"
            }
            
            for stat, value in stats.items():
                st.metric(stat, value)
        
        # Generate summary
        if summary_type == "Extractive":
            # Simple extractive summarization using TF-IDF
            vectorizer = TfidfVectorizer(stop_words='english', max_features=100)
            sentence_vectors = vectorizer.fit_transform(sentences)
            
            # Calculate sentence scores based on TF-IDF
            sentence_scores = np.sum(sentence_vectors.toarray(), axis=1)
            
            # Select top sentences
            num_sentences = max(1, int(len(sentences) * summary_length / 100))
            top_sentence_indices = np.argsort(sentence_scores)[-num_sentences:]
            top_sentence_indices = sorted(top_sentence_indices)
            
            summary = ' '.join([sentences[i] for i in top_sentence_indices])
        
        else:  # Abstractive (simulated)
            # Simulate abstractive summarization by combining extractive with paraphrasing
            vectorizer = TfidfVectorizer(stop_words='english', max_features=100)
            sentence_vectors = vectorizer.fit_transform(sentences)
            sentence_scores = np.sum(sentence_vectors.toarray(), axis=1)
            
            num_sentences = max(2, int(len(sentences) * summary_length / 100))
            top_sentence_indices = np.argsort(sentence_scores)[-num_sentences:]
            key_sentences = [sentences[i] for i in sorted(top_sentence_indices)]
            
            # Simulate abstractive summary (in real implementation, use T5, BART, etc.)
            summary = "This article discusses " + key_sentences[0].lower()
            if len(key_sentences) > 1:
                summary += " Key findings include " + key_sentences[1].lower()
            if len(key_sentences) > 2:
                summary += " The research concludes " + key_sentences[-1].lower()
        
        # Display results
        st.markdown("### 📋 Generated Summary")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Original Text:**")
            st.text_area("", value=input_text, height=300, disabled=True, key="original_summary")
        
        with col2:
            st.markdown("**Generated Summary:**")
            st.text_area("", value=summary, height=300, disabled=True, key="generated_summary")
        
        # Summary analysis
        st.markdown("### 📈 Summary Analysis")
        
        original_word_count = len(word_tokenize(input_text))
        summary_word_count = len(word_tokenize(summary))
        compression_ratio = round((1 - summary_word_count / original_word_count) * 100, 1)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Compression Ratio", f"{compression_ratio}%")
        with col2:
            st.metric("Original Words", original_word_count)
        with col3:
            st.metric("Summary Words", summary_word_count)
        with col4:
            # Simple readability score (Flesch-Kincaid approximation)
            summary_sentences = sent_tokenize(summary)
            if summary_sentences:
                avg_sentence_length = summary_word_count / len(summary_sentences)
                readability_score = 206.835 - (1.015 * avg_sentence_length)
                st.metric("Readability Score", f"{int(readability_score)}")
        
        # Keyword extraction and comparison
        st.markdown("### 🔍 Key Concepts Comparison")
        
        # Extract keywords using TF-IDF
        all_text = [input_text, summary]
        vectorizer = TfidfVectorizer(stop_words='english', max_features=10, ngram_range=(1, 2))
        tfidf_matrix = vectorizer.fit_transform(all_text)
        feature_names = vectorizer.get_feature_names_out()
        
        # Get top keywords for original and summary
        original_keywords = {}
        summary_keywords = {}
        
        for i, feature in enumerate(feature_names):
            original_keywords[feature] = tfidf_matrix[0, i]
            summary_keywords[feature] = tfidf_matrix[1, i]
        
        # Create comparison visualization
        keywords_df = pd.DataFrame({
            'Keyword': feature_names,
            'Original': [original_keywords[kw] for kw in feature_names],
            'Summary': [summary_keywords[kw] for kw in feature_names]
        }).sort_values('Original', ascending=False)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Original', x=keywords_df['Keyword'], y=keywords_df['Original']))
        fig.add_trace(go.Bar(name='Summary', x=keywords_df['Keyword'], y=keywords_df['Summary']))
        
        fig.update_layout(
            title='Key Concepts: Original vs Summary',
            xaxis_title='Keywords',
            yaxis_title='TF-IDF Score',
            barmode='group',
            height=400
        )
        fig.update_xaxes(tickangle=45)
        
        st.plotly_chart(fig, use_container_width=True)
    
    elif input_text:
        st.warning("Please enter a longer text (at least 50 words) for meaningful summarization.")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Educational content
    st.markdown('<div class="concept-box">', unsafe_allow_html=True)
    st.markdown("## 🎓 Teaching Notes: Text Summarization")
    st.markdown("""
    **Core Concepts:**
    
    1. **Extractive Summarization**:
       - Selects important sentences from original text
       - Methods: TF-IDF, TextRank, sentence scoring
       - Pros: Factually accurate, grammatically correct
       - Cons: May lack coherence, limited paraphrasing
    
    2. **Abstractive Summarization**:
       - Generates new sentences expressing main ideas
       - Methods: Sequence-to-sequence models (T5, BART, Pegasus)
       - Pros: More fluent, can paraphrase and synthesize
       - Cons: Risk of hallucination, factual errors
    
    **Technical Approaches:**
    - **Classical**: TF-IDF, graph-based (TextRank, LexRank)
    - **Neural**: Encoder-decoder architectures with attention
    - **Transformer-based**: Pre-trained models fine-tuned for summarization
    
    **Evaluation Metrics:**
    - ROUGE scores (recall-oriented overlap)
    - BLEU scores (precision-oriented)
    - Human evaluation (informativeness, fluency, coherence)
    - Factual consistency measures
    
    **Discussion Points:**
    - Multi-document vs single-document summarization
    - Query-focused summarization
    - Domain adaptation challenges
    - Handling different text types (news, scientific papers, legal documents)
    """)
    st.markdown('</div>', unsafe_allow_html=True)

def show_ner():
    st.markdown('<h2 class="section-header">🏷️ Named Entity Recognition Lab</h2>', unsafe_allow_html=True)
    
    nlp = load_spacy_model()
    
    if nlp is None:
        st.error("spaCy model not available. Please install it to use NER features.")
        return
    
    st.markdown('<div class="demo-box">', unsafe_allow_html=True)
    st.subheader("🎯 Interactive NER Analyzer")
    
    # Sample texts for NER
    sample_ner_texts = {
        "News Article": """
        Apple Inc. CEO Tim Cook announced today that the company will invest $1 billion in Austin, Texas, 
        to build a new campus. The facility, expected to be completed by 2025, will house up to 5,000 employees. 
        Cook made the announcement during a press conference at the White House with President Joe Biden. 
        The investment is part of Apple's broader commitment to create jobs in the United States.
        """,
        
        "Business Report": """
        Microsoft Corporation reported quarterly earnings of $2.32 per share, beating Wall Street expectations. 
        The Redmond, Washington-based company saw strong growth in its Azure cloud platform, with revenue 
        increasing 35% year-over-year. CEO Satya Nadella credited the success to investments in artificial 
        intelligence and partnerships with OpenAI. The stock price rose 5% in after-hours trading on NASDAQ.
        """,
        
        "Medical Case": """
        Dr. Sarah Johnson at Mayo Clinic in Rochester, Minnesota, published groundbreaking research on 
        Alzheimer's disease treatment. The study, funded by the National Institutes of Health, involved 
        500 patients across 15 medical centers. Initial results show a 40% reduction in cognitive decline 
        when using the new therapy developed by her team at Johns Hopkins University.
        """,
        
        "Custom": ""
    }
    
    text_choice = st.selectbox("Choose sample text or enter custom:", list(sample_ner_texts.keys()))
    
    if text_choice == "Custom":
        input_text = st.text_area("Enter text for NER analysis:", height=150, placeholder="Enter text with named entities...")
    else:
        input_text = st.text_area("Text to analyze:", value=sample_ner_texts[text_choice], height=150)
    
    if input_text:
        # Process text with spaCy
        doc = nlp(input_text)
        
        # Extract entities
        entities = [(ent.text, ent.label_, ent.start_char, ent.end_char) for ent in doc.ents]
        
        if entities:
            # Display annotated text
            st.markdown("### 🎨 Annotated Text")
            
            # Create highlighted text
            highlighted_text = input_text
            offset = 0
            
            # Sort entities by start position (reverse order to maintain positions)
            entities_sorted = sorted(entities, key=lambda x: x[2], reverse=True)
            
            entity_colors = {
                'PERSON': '#ff9999',
                'ORG': '#66b3ff', 
                'GPE': '#99ff99',
                'MONEY': '#ffcc99',
                'DATE': '#ff99cc',
                'TIME': '#99ffcc',
                'PERCENT': '#ffff99',
                'PRODUCT': '#cc99ff',
                'EVENT': '#ff6666',
                'FAC': '#66ff66',
                'LAW': '#6666ff',
                'LANGUAGE': '#ffccff'
            }
            
            for entity_text, entity_label, start, end in entities_sorted:
                color = entity_colors.get(entity_label, '#dddddd')
                replacement = f'<mark style="background-color: {color}; padding: 2px 4px; border-radius: 3px; font-weight: bold;">{entity_text} ({entity_label})</mark>'
                highlighted_text = highlighted_text[:start] + replacement + highlighted_text[end:]
            
            st.markdown(highlighted_text, unsafe_allow_html=True)
            
            # Entity statistics
            st.markdown("### 📊 Entity Analysis")
            
            entity_df = pd.DataFrame(entities, columns=['Text', 'Label', 'Start', 'End'])
            entity_counts = entity_df['Label'].value_counts()
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Entity distribution pie chart
                fig = px.pie(values=entity_counts.values, names=entity_counts.index, 
                           title="Entity Type Distribution")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Entity frequency bar chart
                fig = px.bar(x=entity_counts.index, y=entity_counts.values,
                           title="Entity Type Frequency",
                           labels={'x': 'Entity Type', 'y': 'Count'})
                st.plotly_chart(fig, use_container_width=True)
            
            # Detailed entity table
            st.markdown("### 📋 Detailed Entity Information")
            
            # Add descriptions for entity types
            entity_descriptions = {
                'PERSON': 'People, including fictional characters',
                'ORG': 'Companies, agencies, institutions, etc.',
                'GPE': 'Countries, cities, states, provinces',
                'MONEY': 'Monetary values, including unit',
                'DATE': 'Absolute or relative dates or periods',
                'TIME': 'Times smaller than a day',
                'PERCENT': 'Percentage values',
                'PRODUCT': 'Objects, vehicles, foods, etc.',
                'EVENT': 'Named hurricanes, battles, wars, sports events',
                'FAC': 'Buildings, airports, highways, bridges',
                'LAW': 'Named documents made into laws',
                'LANGUAGE': 'Any named language'
            }
            
            detailed_entities = []
            for text, label, start, end in entities:
                detailed_entities.append({
                    'Entity': text,
                    'Type': label,
                    'Description': entity_descriptions.get(label, 'Other entity type'),
                    'Position': f"{start}-{end}",
                    'Length': len(text)
                })
            
            detailed_df = pd.DataFrame(detailed_entities)
            st.dataframe(detailed_df, use_container_width=True)
            
            # Entity relationships visualization
            if len(entities) > 3:
                st.markdown("### 🕸️ Entity Co-occurrence Network")
                
                # Create simple co-occurrence matrix
                entity_texts = [ent[0] for ent in entities]
                unique_entities = list(set(entity_texts))
                
                if len(unique_entities) > 1:
                    # Create network visualization
                    fig = go.Figure()
                    
                    # Add nodes
                    for i, entity in enumerate(unique_entities):
                        entity_type = next(ent[1] for ent in entities if ent[0] == entity)
                        color = entity_colors.get(entity_type, '#dddddd')
                        
                        fig.add_trace(go.Scatter(
                            x=[i], y=[0],
                            mode='markers+text',
                            text=[entity],
                            textposition="top center",
                            marker=dict(size=30, color=color),
                            name=entity_type,
                            showlegend=entity_type not in [trace.name for trace in fig.data]
                        ))
                    
                    fig.update_layout(
                        title="Named Entities in Text",
                        showlegend=True,
                        height=300,
                        xaxis=dict(visible=False),
                        yaxis=dict(visible=False)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No named entities found in the text. Try entering text with names, places, organizations, etc.")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Educational content
    st.markdown('<div class="concept-box">', unsafe_allow_html=True)
    st.markdown("## 🎓 Teaching Notes: Named Entity Recognition")
    st.markdown("""
    **Core Concepts:**
    
    1. **Entity Types (Common NER Tags)**:
       - **PERSON**: Names of people, including fictional characters
       - **ORG**: Organizations, companies, agencies, institutions
       - **GPE**: Geopolitical entities (countries, cities, states)
       - **MONEY**: Monetary values including currency units
       - **DATE/TIME**: Temporal expressions
       - **PRODUCT**: Objects, vehicles, foods, etc.
    
    2. **Technical Approaches**:
       - **Rule-based**: Pattern matching, gazetteers, regular expressions
       - **Statistical**: CRF (Conditional Random Fields), HMMs
       - **Neural**: BiLSTM-CRF, BERT-based models
       - **Hybrid**: Combining multiple approaches
    
    3. **Evaluation Metrics**:
       - **Exact Match**: Entity boundaries must match exactly
       - **Partial Match**: Overlapping entity spans
       - **Type Match**: Correct entity type regardless of boundaries
       - **F1 Score**: Harmonic mean of precision and recall
    
    **Challenges in NER:**
    - Ambiguity: "Apple" (company vs fruit)
    - Nested entities: "University of California, Los Angeles"
    - Domain adaptation: Medical vs news text
    - Multilingual and code-switching scenarios
    - Emerging entities not in training data
    
    **Real-world Applications:**
    - Information extraction from documents
    - Knowledge graph construction
    - Content recommendation systems
    - Compliance and regulatory monitoring
    - Biomedical text mining
    """)
    st.markdown('</div>', unsafe_allow_html=True)

def show_attention():
    st.markdown('<h2 class="section-header">🧠 Attention Mechanisms Visualization</h2>', unsafe_allow_html=True)
    
    st.markdown('<div class="demo-box">', unsafe_allow_html=True)
    st.subheader("👁️ Interactive Attention Explorer")
    
    # Sample sentences for attention visualization
    sample_sentences = {
        "Simple Translation": ("The cat sat on the mat", "Le chat était assis sur le tapis"),
        "Complex Sentence": ("The quick brown fox jumps over the lazy dog", "Parsed representation"),
        "Question Answering": ("What is the capital of France?", "Paris is the capital"),
        "Custom": ("", "")
    }
    
    sentence_choice = st.selectbox("Choose sample or enter custom:", list(sample_sentences.keys()))
    
    col1, col2 = st.columns(2)
    
    if sentence_choice == "Custom":
        with col1:
            source_text = st.text_input("Source text:", placeholder="Enter source sentence...")
        with col2:
            target_text = st.text_input("Target text:", placeholder="Enter target sentence...")
    else:
        source_text, target_text = sample_sentences[sentence_choice]
        with col1:
            source_text = st.text_input("Source text:", value=source_text)
        with col2:
            target_text = st.text_input("Target text:", value=target_text)
    
    if source_text and target_text:
        source_tokens = source_text.split()
        target_tokens = target_text.split()
        
        # Generate synthetic attention weights (in real implementation, use actual model)
        np.random.seed(42)
        attention_matrix = np.random.rand(len(target_tokens), len(source_tokens))
        
        # Add some structure to make it more realistic
        for i in range(len(target_tokens)):
            # Add diagonal preference (monotonic attention)
            if i < len(source_tokens):
                attention_matrix[i, i] += 0.5
            # Add some nearby attention
            for j in range(max(0, i-2), min(len(source_tokens), i+3)):
                attention_matrix[i, j] += 0.2
        
        # Normalize to make it sum to 1 for each target word
        attention_matrix = attention_matrix / attention_matrix.sum(axis=1, keepdims=True)
        
        # Visualization options
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.markdown("### 🎛️ Visualization Options")
            viz_type = st.radio("Visualization Type:", ["Heatmap", "Line Plot", "3D Surface"])
            highlight_threshold = st.slider("Attention Threshold:", 0.0, 1.0, 0.1, 0.05)
            show_values = st.checkbox("Show attention values", True)
        
        with col2:
            if viz_type == "Heatmap":
                # Create attention heatmap
                fig = go.Figure(data=go.Heatmap(
                    z=attention_matrix,
                    x=source_tokens,
                    y=target_tokens,
                    colorscale='Blues',
                    text=np.round(attention_matrix, 3) if show_values else None,
                    texttemplate="%{text}" if show_values else None,
                    textfont={"size": 10}
                ))
                
                fig.update_layout(
                    title="Attention Heatmap",
                    xaxis_title="Source Tokens",
                    yaxis_title="Target Tokens",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            elif viz_type == "Line Plot":
                # Create line plot showing attention for each target word
                fig = go.Figure()
                
                for i, target_word in enumerate(target_tokens):
                    fig.add_trace(go.Scatter(
                        x=source_tokens,
                        y=attention_matrix[i],
                        mode='lines+markers',
                        name=f"Target: {target_word}",
                        line=dict(width=3)
                    ))
                
                fig.update_layout(
                    title="Attention Weights by Target Word",
                    xaxis_title="Source Tokens",
                    yaxis_title="Attention Weight",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            elif viz_type == "3D Surface":
                # Create 3D surface plot
                fig = go.Figure(data=[go.Surface(
                    z=attention_matrix,
                    x=list(range(len(source_tokens))),
                    y=list(range(len(target_tokens))),
                    colorscale='Blues'
                )])
                
                fig.update_layout(
                    title="3D Attention Surface",
                    scene=dict(
                        xaxis_title="Source Position",
                        yaxis_title="Target Position", 
                        zaxis_title="Attention Weight"
                    ),
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Attention analysis
        st.markdown("### 📈 Attention Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Most attended source tokens
            source_attention_sum = np.sum(attention_matrix, axis=0)
            most_attended_idx = np.argmax(source_attention_sum)
            most_attended_word = source_tokens[most_attended_idx]
            most_attended_score = source_attention_sum[most_attended_idx]
            
            st.metric("Most Attended Source Word", 
                     f"{most_attended_word}", 
                     f"{most_attended_score:.3f}")
            
            # Attention distribution entropy
            attention_entropy = -np.sum(attention_matrix * np.log(attention_matrix + 1e-10), axis=1)
            avg_entropy = np.mean(attention_entropy)
            st.metric("Average Attention Entropy", f"{avg_entropy:.3f}")
        
        with col2:
            # Target word with highest attention concentration
            max_attention_per_target = np.max(attention_matrix, axis=1)
            most_focused_idx = np.argmax(max_attention_per_target)
            most_focused_word = target_tokens[most_focused_idx]
            most_focused_score = max_attention_per_target[most_focused_idx]
            
            st.metric("Most Focused Target Word",
                     f"{most_focused_word}",
                     f"{most_focused_score:.3f}")
            
            # Attention sparsity (percentage of weights above threshold)
            sparsity = np.sum(attention_matrix > highlight_threshold) / attention_matrix.size * 100
            st.metric("Attention Sparsity", f"{sparsity:.1f}%")
        
        # Word-level attention breakdown
        st.markdown("### 🔍 Word-Level Attention Breakdown")
        
        selected_target = st.selectbox("Select target word to analyze:", target_tokens)
        target_idx = target_tokens.index(selected_target)
        
        word_attention = attention_matrix[target_idx]
        attention_df = pd.DataFrame({
            'Source Word': source_tokens,
            'Attention Weight': word_attention,
            'Percentage': word_attention * 100
        }).sort_values('Attention Weight', ascending=False)
        
        fig = px.bar(attention_df, x='Source Word', y='Attention Weight',
                    title=f'Attention Distribution for "{selected_target}"',
                    color='Attention Weight',
                    color_continuous_scale='Blues')
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(attention_df, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Multi-head attention demo
    st.markdown('<div class="demo-box">', unsafe_allow_html=True)
    st.subheader("🔄 Multi-Head Attention Simulation")
    
    if source_text and target_text:
        num_heads = st.slider("Number of Attention Heads:", 2, 8, 4)
        
        # Generate multiple attention heads
        heads_data = []
        for head in range(num_heads):
            np.random.seed(42 + head)  # Different seed for each head
            head_attention = np.random.rand(len(target_tokens), len(source_tokens))
            
            # Give each head a different focus pattern
            if head == 0:  # Syntactic head - focus on nearby words
                for i in range(len(target_tokens)):
                    for j in range(max(0, i-1), min(len(source_tokens), i+2)):
                        head_attention[i, j] += 0.5
            elif head == 1:  # Semantic head - focus on content words
                content_positions = [i for i, word in enumerate(source_tokens) 
                                   if word.lower() not in ['the', 'a', 'an', 'and', 'or', 'but']]
                for i in range(len(target_tokens)):
                    for j in content_positions:
                        head_attention[i, j] += 0.3
            elif head == 2:  # Long-distance head
                for i in range(len(target_tokens)):
                    if i < len(source_tokens):
                        head_attention[i, 0] += 0.4  # Focus on first word
                        head_attention[i, -1] += 0.4  # Focus on last word
            
            # Normalize
            head_attention = head_attention / head_attention.sum(axis=1, keepdims=True)
            heads_data.append(head_attention)
        
        # Display multiple heads
        cols = st.columns(min(4, num_heads))
        for i, head_attention in enumerate(heads_data[:4]):  # Show first 4 heads
            with cols[i % 4]:
                fig = go.Figure(data=go.Heatmap(
                    z=head_attention,
                    x=source_tokens,
                    y=target_tokens,
                    colorscale='Blues',
                    showscale=False
                ))
                
                fig.update_layout(
                    title=f"Head {i+1}",
                    height=300,
                    xaxis={'tickfont': {'size': 8}},
                    yaxis={'tickfont': {'size': 8}}
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Average attention across heads
        if num_heads > 1:
            st.markdown("### 🎯 Averaged Multi-Head Attention")
            avg_attention = np.mean(heads_data, axis=0)
            
            fig = go.Figure(data=go.Heatmap(
                z=avg_attention,
                x=source_tokens,
                y=target_tokens,
                colorscale='Blues',
                text=np.round(avg_attention, 3),
                texttemplate="%{text}",
                textfont={"size": 10}
            ))
            
            fig.update_layout(
                title="Average Attention Across All Heads",
                xaxis_title="Source Tokens",
                yaxis_title="Target Tokens",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Educational content
    st.markdown('<div class="concept-box">', unsafe_allow_html=True)
    st.markdown("## 🎓 Teaching Notes: Attention Mechanisms")
    st.markdown("""
    **Core Concepts:**
    
    1. **Attention Fundamentals**:
       - **Query (Q)**: What we're looking for
       - **Key (K)**: What we're comparing against
       - **Value (V)**: What we actually use
       - **Attention Score**: Similarity between Query and Key
       - **Attention Weight**: Normalized attention scores (softmax)
    
    2. **Self-Attention vs Cross-Attention**:
       - **Self-Attention**: Q, K, V all from the same sequence
       - **Cross-Attention**: Q from one sequence, K, V from another
       - **Applications**: Self-attention for encoding, cross-attention for translation
    
    3. **Multi-Head Attention Benefits**:
       - **Parallel Processing**: Multiple attention patterns simultaneously
       - **Different Perspectives**: Each head can focus on different relationships
       - **Syntactic vs Semantic**: Different heads capture different linguistic aspects
       - **Ensemble Effect**: Combining multiple attention mechanisms
    
    **Mathematical Foundation:**
    ```
    Attention(Q, K, V) = softmax(QK^T / √d_k)V
    MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O
    where head_i = Attention(QW^Q_i, KW^K_i, VW^V_i)
    ```
    
    **Key Teaching Points:**
    - Attention solves the "information bottleneck" problem in seq2seq models
    - Enables direct connections between any two positions in sequences
    - Computationally expensive (O(n²)) but highly parallelizable
    - Foundation of Transformer architecture and modern LLMs
    - Attention weights provide interpretability and can reveal model reasoning
    
    **Real-world Impact:**
    - Machine Translation: Better handling of long sentences and rare words
    - Question Answering: Direct focus on relevant context
    - Text Summarization: Identifying key information across documents
    - Language Modeling: Understanding long-range dependencies in text
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# Sidebar additional information
def show_sidebar_info():
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📚 Quick Reference")
    st.sidebar.markdown("""
    **NLP Pipeline Steps:**
    1. Tokenization
    2. Preprocessing
    3. Feature extraction
    4. Model training/inference
    5. Post-processing
    
    **Key Algorithms:**
    - Word2Vec/GloVe
    - LSTM/GRU
    - Transformer
    - BERT/GPT family
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🛠️ Tools Used")
    st.sidebar.markdown("""
    - **NLTK**: Text preprocessing
    - **spaCy**: NER and parsing
    - **TextBlob**: Sentiment analysis
    - **scikit-learn**: ML algorithms
    - **Plotly**: Interactive visualizations
    - **Streamlit**: Web interface
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 💡 Teaching Tips")
    st.sidebar.markdown("""
    1. Start with basic concepts
    2. Use interactive demos
    3. Show real-world examples
    4. Discuss limitations
    5. Connect to current research
    """)

# Run the main application
if __name__ == "__main__":
    show_sidebar_info()
    main()

def show_embeddings():
    st.markdown('<h2 class="section-header">🎯 Word Embeddings Visualization</h2>', unsafe_allow_html=True)
    
    # Sample words for embedding visualization
    sample_words = [
        "king", "queen", "man", "woman", "prince", "princess",
        "cat", "dog", "animal", "pet", "puppy", "kitten",
        "run", "walk", "jump", "sprint", "jog", "move",
        "happy", "sad", "joy", "anger", "excited", "depressed",
        "big", "small", "large", "tiny", "huge", "massive"
    ]
    
    st.markdown('<div class="demo-box">', unsafe_allow_html=True)
    st.subheader("🔍 Embedding Similarity Explorer")
    
    # Create mock embeddings (in real implementation, use pre-trained embeddings)
    np.random.seed(42)
    embeddings = {}
    
    # Create semantically meaningful clusters
    word_clusters = {
        "royalty": ["king", "queen", "prince", "princess"],
        "animals": ["cat", "dog", "animal", "pet", "puppy", "kitten"], 
        "movement": ["run", "walk", "jump", "sprint", "jog", "move"],
        "emotions": ["happy", "sad", "joy", "anger", "excited", "depressed"],
        "size": ["big", "small", "large", "tiny", "huge", "massive"]
    }
    
    cluster_centers = {
        "royalty": np.array([2, 2]),
        "animals": np.array([-2, 1]),
        "movement": np.array([0, -2]),
        "emotions": np.array([2, -1]),
        "size": np.array([-1, 2])
    }
    
    for cluster, words in word_clusters.items():
        center = cluster_centers[cluster]
        for word in words:
            # Add some noise around cluster center
            embeddings[word] = center + np.random.normal(0, 0.3, 2)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### 🎚️ Visualization Controls")
        selected_words = st.multiselect(
            "Select words to visualize:",
            sample_words,
            default=["king", "queen", "man", "woman", "cat", "dog", "happy", "sad"]
        )
        
        viz_method = st.radio(
            "Dimensionality Reduction:",
            ["PCA", "t-SNE"]
        )
        
        show_clusters = st.checkbox("Show semantic clusters", value=True)
    
    with col2:
        if selected_words:
            # Create embedding visualization
            selected_embeddings = np.array([embeddings[word] for word in selected_words])
            
            fig = go.Figure()
            
            if show_clusters:
                colors = {'royalty': 'red', 'animals': 'blue', 'movement': 'green', 
                         'emotions': 'purple', 'size': 'orange'}
                
                for i, word in enumerate(selected_words):
                    cluster = next((c for c, words in word_clusters.items() if word in words), 'other')
                    color = colors.get(cluster, 'gray')
                    
                    fig.add_trace(go.Scatter(
                        x=[selected_embeddings[i, 0]],
                        y=[selected_embeddings[i, 1]], 
                        mode='markers+text',
                        text=[word],
                        textposition="top center",
                        marker=dict(size=12, color=color),
                        name=cluster,
                        showlegend=cluster not in [trace.name for trace in fig.data]
                    ))
            else:
                fig.add_trace(go.Scatter(
                    x=selected_embeddings[:, 0],
                    y=selected_embeddings[:, 1],
                    mode='markers+text',
                    text=selected_words,
                    textposition="top center",
                    marker=dict(size=12, color='blue')
                ))
            
            fig.update_layout(
                title=f"Word Embeddings Visualization ({viz_method})",
                xaxis_title="Dimension 1",
                yaxis_title="Dimension 2",
                height=500,
                showlegend=show_clusters
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Similarity calculation demo
    st.markdown('<div class="demo-box">', unsafe_allow_html=True)
    st.subheader("📐 Cosine Similarity Calculator")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        word1 = st.selectbox("First word:", sample_words, index=0)
    with col2:
        word2 = st.selectbox("Second word:", sample_words, index=1)
    with col3:
        if word1 and word2:
            # Calculate cosine similarity
            vec1 = embeddings[word1]
            vec2 = embeddings[word2]
            
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            similarity = dot_product / (norm1 * norm2)
            
            st.metric("Cosine Similarity", f"{similarity:.3f}")
            
            # Similarity interpretation
            if similarity > 0.8:
                st.success("Very Similar")
            elif similarity > 0.5:
                st.info("Moderately Similar")
            elif similarity > 0.2:
                st.warning("Somewhat Similar")
            else:
                st.error("Not Similar")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Educational content
    st.markdown('<div class="concept-box">', unsafe_allow_html=True)
    st.markdown("## 🎓 Teaching Notes: Word Embeddings")
    st.markdown("""
    **Key Concepts to Discuss:**
    
    1. **Vector Space Semantics**: Words with similar meanings cluster together in high-dimensional space
    2. **Distributional Hypothesis**: "You shall know a word by the company it keeps" - words in similar contexts have similar meanings
    3. **Dimensionality**: Typical embeddings use 100-1000 dimensions to capture semantic relationships
    4. **Training Methods**:
       - **Skip-gram**: Predict context words from target word
       - **CBOW**: Predict target word from context
       - **GloVe**: Global word co-occurrence statistics
    
    **Demo Teaching Points:**
    - Show how semantic relationships emerge from statistical patterns
    - Discuss the king-man+woman≈queen analogy
    - Explain limitations: polysemy, bias, context independence
    - Compare static vs contextual embeddings (Word2Vec vs BERT)
    """)
    st.markdown('</div>', unsafe_allow_html=True)

def show_sentiment_analysis():
    st.markdown('<h2 class="section-header">💭 Sentiment Analysis Laboratory</h2>', unsafe_allow_html=True)
    
    st.markdown('<div class="demo-box">', unsafe_allow_html=True)
    st.subheader("🎭 Interactive Sentiment Analyzer")
    
    # Sample texts for sentiment analysis
    sample_sentiments = {
        "Positive Review": "This movie was absolutely fantastic! The acting was superb and the plot kept me engaged throughout. Highly recommend!",
        "Negative Review": "Terrible service and poor quality food. The staff was rude and unprofessional. Would not recommend.",
        "Neutral News": "The stock market opened at 3,500 points today. Trading volume was moderate with mixed sector performance.",
        "Mixed Emotions": "I'm excited about the new job opportunity, but nervous about leaving my current team. It's bittersweet.",
        "Custom": ""
    }
    
    text_choice = st.selectbox("Choose sample text or enter custom:", list(sample_sentiments.keys()))
    
    if text_choice == "Custom":
        input_text = st.text_area("Enter text for sentiment analysis:", height=100, placeholder="Type your text here...")
    else:
        input_text = st.text_area("Text to analyze:", value=sample_sentiments[text_choice], height=100)
    
    if input_text:
        # Perform sentiment analysis using TextBlob
        blob = TextBlob(input_text)
        polarity = blob.sentiment.polarity  # -1 to 1
        subjectivity = blob.sentiment.subjectivity  # 0 to 1
        
        # Classify sentiment
        if polarity > 0.1:
            sentiment_label = "Positive"
            sentiment_color = "green"
        elif polarity < -0.1:
            sentiment_label = "Negative" 
            sentiment_color = "red"
        else:
            sentiment_label = "Neutral"
            sentiment_color = "gray"
        
        # Display results
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Sentiment", sentiment_label)
        with col2:
            st.metric("Polarity Score", f"{polarity:.3f}")
        with col3:
            st.metric("Subjectivity", f"{subjectivity:.3f}")
        
        # Sentiment visualization
        fig = go.Figure()
        
        # Create sentiment gauge
        fig.add_trace(go.Indicator(
            mode="gauge+number+delta",
            value=polarity,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Sentiment Polarity"},
            delta={'reference': 0},
            gauge={
                'axis': {'range': [-1, 1]},
                'bar': {'color': sentiment_color},
                'steps': [
                    {'range': [-1, -0.1], 'color': "lightcoral"},
                    {'range': [-0.1, 0.1], 'color': "lightgray"},
                    {'range': [0.1, 1], 'color': "lightgreen"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 0.9
                }
            }
        ))
        
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        
        # Sentence-level sentiment analysis
        sentences = sent_tokenize(input_text)
        if len(sentences) > 1:
            st.subheader("📊 Sentence-Level Sentiment Breakdown")
            
            sentence_sentiments = []
            for i, sentence in enumerate(sentences):
                sent_blob = TextBlob(sentence)
                sent_polarity = sent_blob.sentiment.polarity
                sentence_sentiments.append({
                    'Sentence': i+1,
                    'Text': sentence[:50] + "..." if len(sentence) > 50 else sentence,
                    'Polarity': sent_polarity,
                    'Sentiment': 'Positive' if sent_polarity > 0.1 else 'Negative' if sent_polarity < -0.1 else 'Neutral'
                })
            
            df = pd.DataFrame(sentence_sentiments)
            
            # Create bar chart
            fig = px.bar(df, x='Sentence', y='Polarity', 
                        color='Sentiment',
                        color_discrete_map={'Positive': 'green', 'Negative': 'red', 'Neutral': 'gray'},
                        title="Sentiment by Sentence")
            st.plotly_chart(fig, use_container_width=True)
            
            st.dataframe(df, use_container_width=True)
        
        # Word-level sentiment contribution
        st.subheader("🔍 Word-Level Sentiment Analysis")
        words = word_tokenize(input_text.lower())
        words = [word for word in words if word.isalpha()]
        
        word_sentiments = []
        for word in words:
            word_blob = TextBlob(word)
            word_polarity = word_blob.sentiment.polarity
            if abs(word_polarity) > 0.1:  # Only show words with notable sentiment
                word_sentiments.append({
                    'Word': word,
                    'Polarity': word_polarity,
                    'Impact': 'Positive' if word_polarity > 0 else 'Negative'
                })
        
        if word_sentiments:
            word_df = pd.DataFrame(word_sentiments)
            word_df = word_df.sort_values('Polarity', key=abs, ascending=False).head(10)
            
            fig = px.bar(word_df, x='Word', y='Polarity', 
                        color='Impact',
                        color_discrete_map={'Positive': 'green', 'Negative': 'red'},
                        title="Top Sentiment-Contributing Words")
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)