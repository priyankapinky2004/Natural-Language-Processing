import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import LabelBinarizer
import time
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Deep Learning Master Class",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .section-header {
        font-size: 2rem;
        font-weight: bold;
        color: #2c3e50;
        border-bottom: 3px solid #3498db;
        padding-bottom: 0.5rem;
        margin: 2rem 0 1rem 0;
    }
    
    .concept-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        color: white;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #3498db;
        margin: 0.5rem 0;
    }
    
    .code-section {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'trained_model' not in st.session_state:
    st.session_state.trained_model = None
if 'training_history' not in st.session_state:
    st.session_state.training_history = None
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False

# Sidebar navigation
st.sidebar.title("🧠 Deep Learning Dashboard")
st.sidebar.markdown("---")

sections = {
    "🏠 Introduction": "intro",
    "🧬 Neural Network Architecture": "architecture", 
    "📊 Training Visualization": "training",
    "🎯 Interactive Demo": "demo",
    "📈 Model Evaluation": "evaluation",
    "🌍 Real-World Applications": "applications",
    "📝 Code Examples": "code"
}

selected_section = st.sidebar.selectbox(
    "Choose a section:",
    options=list(sections.keys()),
    index=0
)

current_section = sections[selected_section]

# Main header
st.markdown('<h1 class="main-header">🧠 Deep Learning Master Class</h1>', unsafe_allow_html=True)
st.markdown("*An Interactive Journey into Neural Networks and Deep Learning*")
st.markdown("---")

# Helper functions
def create_neural_network_diagram():
    """Create an interactive neural network visualization"""
    fig = go.Figure()
    
    # Define layers
    layers = [4, 6, 6, 3]  # Input, Hidden1, Hidden2, Output
    layer_names = ['Input\nLayer', 'Hidden\nLayer 1', 'Hidden\nLayer 2', 'Output\nLayer']
    
    # Colors for different layers
    colors = ['#3498db', '#e74c3c', '#f39c12', '#2ecc71']
    
    # Node positions
    max_nodes = max(layers)
    x_positions = [0, 2, 4, 6]
    
    for layer_idx, (num_nodes, x_pos, color, name) in enumerate(zip(layers, x_positions, colors, layer_names)):
        y_start = (max_nodes - num_nodes) / 2
        y_positions = [y_start + i for i in range(num_nodes)]
        
        # Add nodes
        fig.add_trace(go.Scatter(
            x=[x_pos] * num_nodes,
            y=y_positions,
            mode='markers',
            marker=dict(size=20, color=color, line=dict(width=2, color='white')),
            name=name,
            text=[f'{name}\nNode {i+1}' for i in range(num_nodes)],
            hovertemplate='%{text}<extra></extra>'
        ))
        
        # Add connections to next layer
        if layer_idx < len(layers) - 1:
            next_layer_nodes = layers[layer_idx + 1]
            next_y_start = (max_nodes - next_layer_nodes) / 2
            next_y_positions = [next_y_start + i for i in range(next_layer_nodes)]
            
            for y1 in y_positions:
                for y2 in next_y_positions:
                    fig.add_trace(go.Scatter(
                        x=[x_pos, x_positions[layer_idx + 1]],
                        y=[y1, y2],
                        mode='lines',
                        line=dict(width=1, color='rgba(128,128,128,0.3)'),
                        showlegend=False,
                        hoverinfo='skip'
                    ))
    
    fig.update_layout(
        title="Neural Network Architecture",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=400,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def create_activation_functions_plot():
    """Create visualization of common activation functions"""
    x = np.linspace(-5, 5, 100)
    
    # Define activation functions
    sigmoid = 1 / (1 + np.exp(-x))
    tanh = np.tanh(x)
    relu = np.maximum(0, x)
    leaky_relu = np.where(x > 0, x, x * 0.01)
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['Sigmoid', 'Tanh', 'ReLU', 'Leaky ReLU']
    )
    
    # Add traces
    fig.add_trace(go.Scatter(x=x, y=sigmoid, name='Sigmoid', line=dict(color='#3498db')), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=tanh, name='Tanh', line=dict(color='#e74c3c')), row=1, col=2)
    fig.add_trace(go.Scatter(x=x, y=relu, name='ReLU', line=dict(color='#f39c12')), row=2, col=1)
    fig.add_trace(go.Scatter(x=x, y=leaky_relu, name='Leaky ReLU', line=dict(color='#2ecc71')), row=2, col=2)
    
    fig.update_layout(
        title="Common Activation Functions",
        height=500,
        showlegend=False
    )
    
    return fig

def load_and_preprocess_data(dataset_choice):
    """Load and preprocess the selected dataset"""
    if dataset_choice == "MNIST":
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        x_train = x_train.reshape(-1, 28*28).astype('float32') / 255.0
        x_test = x_test.reshape(-1, 28*28).astype('float32') / 255.0
        num_classes = 10
        
    elif dataset_choice == "Fashion-MNIST":
        (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
        x_train = x_train.reshape(-1, 28*28).astype('float32') / 255.0
        x_test = x_test.reshape(-1, 28*28).astype('float32') / 255.0
        num_classes = 10
        
    elif dataset_choice == "CIFAR-10":
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
        x_train = x_train.reshape(-1, 32*32*3).astype('float32') / 255.0
        x_test = x_test.reshape(-1, 32*32*3).astype('float32') / 255.0
        y_train = y_train.flatten()
        y_test = y_test.flatten()
        num_classes = 10
    
    # Convert to categorical
    y_train_cat = keras.utils.to_categorical(y_train, num_classes)
    y_test_cat = keras.utils.to_categorical(y_test, num_classes)
    
    return (x_train, y_train, x_test, y_test), (y_train_cat, y_test_cat), num_classes

def create_model(architecture, input_shape, num_classes):
    """Create a neural network model based on architecture choice"""
    model = keras.Sequential()
    
    if architecture == "Simple Dense":
        model.add(keras.layers.Dense(128, activation='relu', input_shape=(input_shape,)))
        model.add(keras.layers.Dropout(0.2))
        model.add(keras.layers.Dense(64, activation='relu'))
        model.add(keras.layers.Dropout(0.2))
        model.add(keras.layers.Dense(num_classes, activation='softmax'))
        
    elif architecture == "Deep Dense":
        model.add(keras.layers.Dense(256, activation='relu', input_shape=(input_shape,)))
        model.add(keras.layers.Dropout(0.3))
        model.add(keras.layers.Dense(128, activation='relu'))
        model.add(keras.layers.Dropout(0.2))
        model.add(keras.layers.Dense(64, activation='relu'))
        model.add(keras.layers.Dropout(0.2))
        model.add(keras.layers.Dense(num_classes, activation='softmax'))
        
    elif architecture == "CNN":
        # Reshape for CNN (assuming square images)
        if input_shape == 784:  # 28x28
            model.add(keras.layers.Reshape((28, 28, 1), input_shape=(784,)))
        elif input_shape == 3072:  # 32x32x3
            model.add(keras.layers.Reshape((32, 32, 3), input_shape=(3072,)))
            
        model.add(keras.layers.Conv2D(32, (3, 3), activation='relu'))
        model.add(keras.layers.MaxPooling2D((2, 2)))
        model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(keras.layers.MaxPooling2D((2, 2)))
        model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(64, activation='relu'))
        model.add(keras.layers.Dropout(0.2))
        model.add(keras.layers.Dense(num_classes, activation='softmax'))
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Section: Introduction
if current_section == "intro":
    st.markdown('<h2 class="section-header">🏠 Introduction to Deep Learning</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### Welcome to Deep Learning! 🚀
        
        Deep Learning is a subset of machine learning that uses artificial neural networks 
        to model and understand complex patterns in data. It's inspired by the human brain's 
        neural network structure.
        
        **Key Concepts:**
        """)
        
        concepts = {
            "🧠 Artificial Neural Networks": "Mathematical models inspired by biological neural networks",
            "🔗 Layers": "Stacked transformations that process data progressively",
            "⚡ Activation Functions": "Non-linear functions that enable complex pattern learning",
            "📉 Backpropagation": "Algorithm for training networks by propagating errors backwards",
            "🎯 Loss Functions": "Measures how far predictions are from actual values"
        }
        
        for concept, description in concepts.items():
            st.markdown(f"""
            <div class="concept-card">
                <h4>{concept}</h4>
                <p>{description}</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 📊 Deep Learning Timeline")
        timeline_data = {
            'Year': [1943, 1969, 1986, 2006, 2012, 2017, 2020],
            'Milestone': [
                'McCulloch-Pitts Neuron',
                'Perceptron Limitations',
                'Backpropagation',
                'Deep Belief Networks',
                'AlexNet (ImageNet)',
                'Transformers',
                'GPT-3'
            ]
        }
        
        fig = px.scatter(timeline_data, x='Year', y=[1]*7, 
                        hover_data=['Milestone'], 
                        title="Major DL Milestones")
        fig.update_traces(marker=dict(size=15, color='#3498db'))
        fig.update_layout(height=300, yaxis=dict(visible=False))
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Architecture Overview
    st.markdown("### 🏗️ Popular Deep Learning Architectures")
    
    arch_col1, arch_col2, arch_col3 = st.columns(3)
    
    with arch_col1:
        st.markdown("""
        **🔷 Convolutional Neural Networks (CNNs)**
        - Best for: Image processing, Computer Vision
        - Key feature: Convolution and pooling layers
        - Applications: Image classification, object detection
        """)
    
    with arch_col2:
        st.markdown("""
        **🔄 Recurrent Neural Networks (RNNs)**
        - Best for: Sequential data, Time series
        - Key feature: Memory of previous inputs
        - Applications: Language modeling, speech recognition
        """)
    
    with arch_col3:
        st.markdown("""
        **🎯 Transformers**
        - Best for: Natural Language Processing
        - Key feature: Attention mechanism
        - Applications: Translation, text generation, ChatGPT
        """)

# Section: Neural Network Architecture
elif current_section == "architecture":
    st.markdown('<h2 class="section-header">🧬 Neural Network Architecture</h2>', unsafe_allow_html=True)
    
    # Interactive Neural Network Diagram
    st.markdown("### 🔗 Interactive Neural Network Structure")
    fig = create_neural_network_diagram()
    st.plotly_chart(fig, use_container_width=True)
    
    # Explanation columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🧠 How Neural Networks Work
        
        1. **Input Layer**: Receives the raw data (pixels, features, etc.)
        2. **Hidden Layers**: Process and transform the data through weights and activations
        3. **Output Layer**: Produces the final prediction or classification
        
        Each connection has a **weight** that determines the strength of the signal.
        Each neuron applies an **activation function** to introduce non-linearity.
        """)
        
        # Layer details
        st.markdown("#### 📋 Layer Types")
        layer_info = {
            "Dense/Fully Connected": "Every neuron connects to every neuron in the next layer",
            "Convolutional": "Applies filters to detect local patterns (edges, shapes)",
            "Pooling": "Reduces spatial dimensions while keeping important features",
            "Dropout": "Randomly sets some neurons to zero during training (prevents overfitting)",
            "Batch Normalization": "Normalizes inputs to each layer for faster training"
        }
        
        for layer_type, description in layer_info.items():
            st.markdown(f"**{layer_type}**: {description}")
    
    with col2:
        st.markdown("### ⚡ Activation Functions")
        fig_activation = create_activation_functions_plot()
        st.plotly_chart(fig_activation, use_container_width=True)
        
        st.markdown("""
        **Why Activation Functions?**
        - Without them, neural networks would be linear (just matrix multiplication)
        - They introduce non-linearity, enabling complex pattern learning
        - Different functions have different properties and use cases
        """)

# Section: Training Visualization
elif current_section == "training":
    st.markdown('<h2 class="section-header">📊 Training Process Visualization</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    ### 🎯 Understanding Neural Network Training
    
    Training a neural network involves:
    1. **Forward Pass**: Data flows through network to make predictions
    2. **Loss Calculation**: Compare predictions with actual labels
    3. **Backward Pass**: Calculate gradients using backpropagation
    4. **Weight Update**: Adjust weights using optimizer (SGD, Adam, etc.)
    """)
    
    # Training simulation
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### 🎮 Training Simulator")
        
        if st.button("🚀 Simulate Training Process"):
            # Create animated training visualization
            epochs = 50
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Simulate training metrics
            losses = []
            accuracies = []
            
            loss_chart = st.empty()
            
            for epoch in range(epochs):
                # Simulate decreasing loss and increasing accuracy
                loss = 2.5 * np.exp(-epoch/15) + 0.1 + np.random.normal(0, 0.05)
                accuracy = (1 - np.exp(-epoch/10)) * 0.95 + np.random.normal(0, 0.02)
                
                losses.append(max(0, loss))
                accuracies.append(min(1, max(0, accuracy)))
                
                # Update progress
                progress = (epoch + 1) / epochs
                progress_bar.progress(progress)
                status_text.text(f'Epoch {epoch+1}/{epochs} - Loss: {loss:.4f} - Accuracy: {accuracy:.4f}')
                
                # Update chart
                if epoch % 5 == 0:  # Update chart every 5 epochs
                    fig = make_subplots(specs=[[{"secondary_y": True}]])
                    
                    fig.add_trace(
                        go.Scatter(x=list(range(len(losses))), y=losses, name="Loss", line=dict(color='#e74c3c')),
                        secondary_y=False,
                    )
                    
                    fig.add_trace(
                        go.Scatter(x=list(range(len(accuracies))), y=accuracies, name="Accuracy", line=dict(color='#2ecc71')),
                        secondary_y=True,
                    )
                    
                    fig.update_xaxes(title_text="Epoch")
                    fig.update_yaxes(title_text="Loss", secondary_y=False, color='#e74c3c')
                    fig.update_yaxes(title_text="Accuracy", secondary_y=True, color='#2ecc71')
                    fig.update_layout(title="Training Progress", height=300)
                    
                    loss_chart.plotly_chart(fig, use_container_width=True)
                
                time.sleep(0.1)  # Small delay for animation effect
    
    with col2:
        st.markdown("#### 📚 Key Training Concepts")
        
        training_concepts = {
            "🎯 Loss Function": {
                "description": "Measures prediction error",
                "examples": "Cross-entropy (classification), MSE (regression)"
            },
            "📈 Optimizer": {
                "description": "Algorithm for updating weights",
                "examples": "SGD, Adam, RMSprop, AdaGrad"
            },
            "📊 Learning Rate": {
                "description": "Controls step size in weight updates",
                "examples": "0.001 (typical), 0.1 (high), 0.0001 (low)"
            },
            "🔄 Batch Size": {
                "description": "Number of samples processed together",
                "examples": "32, 64, 128, 256"
            },
            "🎪 Epochs": {
                "description": "Complete passes through training data",
                "examples": "10-100 (typical), 1000+ (large datasets)"
            }
        }
        
        for concept, details in training_concepts.items():
            with st.expander(concept):
                st.markdown(f"**Description**: {details['description']}")
                st.markdown(f"**Examples**: {details['examples']}")
        
        # Gradient descent visualization
        st.markdown("#### 🎢 Gradient Descent Visualization")
        
        # Create simple quadratic function for gradient descent demo
        x = np.linspace(-5, 5, 100)
        y = x**2 + 2*x + 1
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='Loss Function', line=dict(color='#3498db')))
        
        # Add gradient descent path
        learning_rates = [0.1, 0.3, 0.5]
        colors = ['#e74c3c', '#f39c12', '#2ecc71']
        
        for lr, color in zip(learning_rates, colors):
            x_gd = [3]  # Starting point
            y_gd = [x_gd[0]**2 + 2*x_gd[0] + 1]
            
            for _ in range(20):
                gradient = 2*x_gd[-1] + 2
                x_new = x_gd[-1] - lr * gradient
                y_new = x_new**2 + 2*x_new + 1
                x_gd.append(x_new)
                y_gd.append(y_new)
                
                if abs(gradient) < 0.01:
                    break
            
            fig.add_trace(go.Scatter(
                x=x_gd, y=y_gd, mode='lines+markers',
                name=f'LR={lr}', line=dict(color=color)
            ))
        
        fig.update_layout(
            title="Gradient Descent with Different Learning Rates",
            xaxis_title="Parameter Value",
            yaxis_title="Loss",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)

# Section: Interactive Demo
elif current_section == "demo":
    st.markdown('<h2 class="section-header">🎯 Interactive Deep Learning Demo</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    ### 🎮 Train Your Own Neural Network!
    
    This interactive demo lets you train a neural network on popular datasets.
    Watch how different architectures and hyperparameters affect performance!
    """)
    
    # Configuration columns
    config_col1, config_col2, config_col3 = st.columns(3)
    
    with config_col1:
        st.markdown("#### 📊 Dataset Selection")
        dataset_choice = st.selectbox(
            "Choose Dataset:",
            ["MNIST", "Fashion-MNIST", "CIFAR-10"],
            help="MNIST: Handwritten digits, Fashion-MNIST: Clothing items, CIFAR-10: Objects"
        )
        
        st.markdown("#### 🏗️ Architecture")
        architecture_choice = st.selectbox(
            "Choose Architecture:",
            ["Simple Dense", "Deep Dense", "CNN"],
            help="Dense: Fully connected layers, CNN: Convolutional network"
        )
    
    with config_col2:
        st.markdown("#### ⚙️ Hyperparameters")
        epochs = st.slider("Epochs:", 1, 20, 5)
        batch_size = st.selectbox("Batch Size:", [32, 64, 128, 256], index=1)
        learning_rate = st.selectbox("Learning Rate:", [0.001, 0.01, 0.1], index=0)
        
        st.markdown("#### 🎯 Training Options")
        validation_split = st.slider("Validation Split:", 0.1, 0.3, 0.2)
        
    with config_col3:
        st.markdown("#### 🚀 Training Control")
        
        if st.button("🏋️ Train Model", type="primary"):
            with st.spinner("Loading and preprocessing data..."):
                # Load data
                (x_train, y_train, x_test, y_test), (y_train_cat, y_test_cat), num_classes = load_and_preprocess_data(dataset_choice)
                
                # Create model
                input_shape = x_train.shape[1]
                model = create_model(architecture_choice, input_shape, num_classes)
                
                # Display model summary
                st.markdown("#### 🔍 Model Architecture")
                model_summary = []
                model.summary(print_fn=lambda x: model_summary.append(x))
                st.text('\n'.join(model_summary))
                
                # Train model
                st.markdown("#### 🎯 Training Progress")
                progress_bar = st.progress(0)
                
                # Create placeholder for live training metrics
                metrics_placeholder = st.empty()
                chart_placeholder = st.empty()
                
                # Custom callback for live updates
                class LiveTrainingCallback(keras.callbacks.Callback):
                    def __init__(self):
                        self.losses = []
                        self.accuracies = []
                        self.val_losses = []
                        self.val_accuracies = []
                    
                    def on_epoch_end(self, epoch, logs=None):
                        self.losses.append(logs.get('loss'))
                        self.accuracies.append(logs.get('accuracy'))
                        self.val_losses.append(logs.get('val_loss'))
                        self.val_accuracies.append(logs.get('val_accuracy'))
                        
                        # Update progress bar
                        progress = (epoch + 1) / epochs
                        progress_bar.progress(progress)
                        
                        # Update metrics display
                        with metrics_placeholder.container():
                            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                            
                            with metric_col1:
                                st.metric("Loss", f"{logs.get('loss', 0):.4f}")
                            with metric_col2:
                                st.metric("Accuracy", f"{logs.get('accuracy', 0):.4f}")
                            with metric_col3:
                                st.metric("Val Loss", f"{logs.get('val_loss', 0):.4f}")
                            with metric_col4:
                                st.metric("Val Accuracy", f"{logs.get('val_accuracy', 0):.4f}")
                        
                        # Update training chart
                        if epoch > 0:  # Skip first epoch for better visualization
                            fig = make_subplots(
                                rows=1, cols=2,
                                subplot_titles=['Loss', 'Accuracy']
                            )
                            
                            epochs_range = list(range(1, len(self.losses) + 1))
                            
                            # Loss plot
                            fig.add_trace(
                                go.Scatter(x=epochs_range, y=self.losses, name='Training Loss', line=dict(color='#e74c3c')),
                                row=1, col=1
                            )
                            fig.add_trace(
                                go.Scatter(x=epochs_range, y=self.val_losses, name='Validation Loss', line=dict(color='#c0392b')),
                                row=1, col=1
                            )
                            
                            # Accuracy plot
                            fig.add_trace(
                                go.Scatter(x=epochs_range, y=self.accuracies, name='Training Accuracy', line=dict(color='#2ecc71')),
                                row=1, col=2
                            )
                            fig.add_trace(
                                go.Scatter(x=epochs_range, y=self.val_accuracies, name='Validation Accuracy', line=dict(color='#27ae60')),
                                row=1, col=2
                            )
                            
                            fig.update_layout(height=400, showlegend=True)
                            fig.update_xaxes(title_text="Epoch")
                            fig.update_yaxes(title_text="Loss", row=1, col=1)
                            fig.update_yaxes(title_text="Accuracy", row=1, col=2)
                            
                            chart_placeholder.plotly_chart(fig, use_container_width=True)
                
                # Initialize callback
                callback = LiveTrainingCallback()
                
                # Compile model with specified learning rate
                model.compile(
                    optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                    loss='categorical_crossentropy',
                    metrics=['accuracy']
                )
                
                # Train the model
                history = model.fit(
                    x_train, y_train_cat,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_split=validation_split,
                    callbacks=[callback],
                    verbose=0
                )
                
                # Store results in session state
                st.session_state.trained_model = model
                st.session_state.training_history = history
                st.session_state.model_trained = True
                st.session_state.test_data = (x_test, y_test, y_test_cat)
                st.session_state.dataset_choice = dataset_choice
                
                st.success("🎉 Model trained successfully!")
        
        # Model prediction demo
        if st.session_state.model_trained:
            st.markdown("#### 🔮 Test Predictions")
            
            if st.button("🎲 Random Prediction"):
                x_test, y_test, y_test_cat = st.session_state.test_data
                
                # Get random sample
                idx = np.random.randint(0, len(x_test))
                sample = x_test[idx:idx+1]
                true_label = y_test[idx]
                
                # Make prediction
                prediction = st.session_state.trained_model.predict(sample, verbose=0)
                predicted_label = np.argmax(prediction)
                confidence = np.max(prediction) * 100
                
                # Display results
                pred_col1, pred_col2 = st.columns(2)
                
                with pred_col1:
                    # Visualize the sample
                    if st.session_state.dataset_choice == "MNIST":
                        image = sample.reshape(28, 28)
                        fig = px.imshow(image, color_continuous_scale='gray', title="Input Image")
                    elif st.session_state.dataset_choice == "Fashion-MNIST":
                        image = sample.reshape(28, 28)
                        fig = px.imshow(image, color_continuous_scale='gray', title="Input Image")
                    else:  # CIFAR-10
                        image = sample.reshape(32, 32, 3)
                        fig = px.imshow(image, title="Input Image")
                    
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
        
        # Success stories
        st.markdown("#### 🏆 Notable Success Stories")
        
        success_col1, success_col2, success_col3 = st.columns(3)
        
        with success_col1:
            st.markdown("""
            **🔬 Medical Diagnosis**
            - Skin cancer detection (dermatology)
            - Diabetic retinopathy screening
            - COVID-19 X-ray analysis
            - Drug discovery acceleration
            """)
        
        with success_col2:
            st.markdown("""
            **🌾 Agriculture**
            - Crop disease identification
            - Yield prediction
            - Pest detection
            - Precision farming
            """)
        
        with success_col3:
            st.markdown("""
            **🏭 Manufacturing**
            - Defect detection
            - Quality assurance
            - Predictive maintenance
            - Supply chain optimization
            """)
    
    with app_tabs[1]:  # Natural Language Processing
        st.markdown("### 🗣️ Natural Language Processing")
        
        nlp_col1, nlp_col2 = st.columns(2)
        
        with nlp_col1:
            st.markdown("""
            **💬 Conversational AI**
            - ChatGPT, Claude, Bard
            - Customer service chatbots
            - Virtual assistants
            - Language tutoring systems
            
            **🌐 Machine Translation**
            - Google Translate
            - Real-time conversation translation
            - Document translation
            - Cross-lingual search
            
            **📝 Text Generation**
            - Content creation
            - Code generation
            - Creative writing assistance
            - Email composition
            """)
        
        with nlp_col2:
            # NLP Evolution Timeline
            st.markdown("#### 📈 NLP Evolution")
            
            nlp_evolution = {
                'Year': [2013, 2017, 2018, 2019, 2020, 2022, 2023],
                'Model': ['Word2Vec', 'Transformer', 'BERT', 'GPT-2', 'GPT-3', 'ChatGPT', 'GPT-4'],
                'Parameters': [0.1, 0.1, 0.3, 1.5, 175, 175, 1000]  # In billions
            }
            
            fig = px.bar(nlp_evolution, x='Model', y='Parameters',
                        title="Growth in NLP Model Size",
                        color='Parameters',
                        color_continuous_scale='viridis')
            
            fig.update_layout(
                xaxis_title="Model",
                yaxis_title="Parameters (Billions)",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)

# Section: Code Examples
elif current_section == "code":
    st.markdown('<h2 class="section-header">📝 Code Examples & Implementation</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    ### 💻 Complete Deep Learning Implementation Guide
    
    This section provides comprehensive code examples for building, training, and deploying deep learning models.
    """)
    
    code_tabs = st.tabs([
        "🏗️ Basic Neural Network", 
        "🖼️ CNN Implementation", 
        "🔄 RNN/LSTM", 
        "⚡ Transfer Learning"
    ])
    
    with code_tabs[0]:  # Basic Neural Network
        st.markdown("### 🏗️ Building a Basic Neural Network")
        
        st.code("""
# Basic Neural Network with TensorFlow/Keras
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# 1. Data Preparation
def prepare_data():
    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    # Normalize pixel values to [0, 1]
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Reshape data for dense layers
    x_train = x_train.reshape(-1, 28*28)
    x_test = x_test.reshape(-1, 28*28)
    
    # Convert labels to categorical
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    
    return (x_train, y_train), (x_test, y_test)

# 2. Model Architecture
def create_neural_network():
    model = keras.Sequential([
        # Input layer (28*28 = 784 features)
        keras.layers.Dense(128, activation='relu', input_shape=(784,)),
        keras.layers.Dropout(0.2),  # Prevent overfitting
        
        # Hidden layer
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.2),
        
        # Output layer (10 classes for digits 0-9)
        keras.layers.Dense(10, activation='softmax')
    ])
    
    return model

# 3. Training
def train_model():
    # Prepare data
    (x_train, y_train), (x_test, y_test) = prepare_data()
    
    # Create model
    model = create_neural_network()
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train model
    history = model.fit(
        x_train, y_train,
        batch_size=32,
        epochs=10,
        validation_split=0.1,
        verbose=1
    )
    
    # Evaluate model
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    return model, history

# Run the complete pipeline
model, history = train_model()
        """, language="python")
        
        st.markdown("""
        **Key Concepts:**
        
        1. **Data Normalization**: Scaling pixel values helps training stability
        2. **Dropout**: Prevents overfitting by randomly setting neurons to 0
        3. **Activation Functions**: ReLU for hidden layers, Softmax for output
        4. **Loss Function**: Categorical crossentropy for multi-class classification
        5. **Optimizer**: Adam adapts learning rate automatically
        """)
    
    with code_tabs[1]:  # CNN Implementation
        st.markdown("### 🖼️ Convolutional Neural Network (CNN)")
        
        st.code("""
# CNN Implementation for Image Classification
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# 1. CNN Architecture
def create_cnn_model():
    model = keras.Sequential([
        # First Convolutional Block
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Second Convolutional Block
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Third Convolutional Block
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.25),
        
        # Flatten and Dense Layers
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')  # 10 classes
    ])
    
    return model

# 2. Data Preparation with Augmentation
def prepare_cifar10_data():
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    
    # Normalize pixel values
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Convert labels to categorical
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    
    return (x_train, y_train), (x_test, y_test)

# 3. Training with Data Augmentation
def train_cnn_with_augmentation():
    (x_train, y_train), (x_test, y_test) = prepare_cifar10_data()
    
    model = create_cnn_model()
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Data augmentation
    datagen = keras.preprocessing.image.ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )
    
    datagen.fit(x_train)
    
    history = model.fit(
        datagen.flow(x_train, y_train, batch_size=32),
        steps_per_epoch=len(x_train) // 32,
        epochs=50,
        validation_data=(x_test, y_test)
    )
    
    return model, history

model, history = train_cnn_with_augmentation()
        """, language="python")
    
    with code_tabs[2]:  # RNN/LSTM
        st.markdown("### 🔄 Recurrent Neural Networks (RNN/LSTM)")
        
        st.code("""
# LSTM Implementation for Time Series/Text
import tensorflow as tf
from tensorflow import keras
import numpy as np

# 1. Time Series LSTM
def create_lstm_model(input_shape):
    model = keras.Sequential([
        # LSTM layers
        keras.layers.LSTM(50, return_sequences=True, input_shape=input_shape),
        keras.layers.Dropout(0.2),
        
        keras.layers.LSTM(50, return_sequences=True),
        keras.layers.Dropout(0.2),
        
        keras.layers.LSTM(50),
        keras.layers.Dropout(0.2),
        
        # Dense layers
        keras.layers.Dense(25, activation='relu'),
        keras.layers.Dense(1)  # Single output for regression
    ])
    
    return model

# 2. Text Classification LSTM
def create_text_classifier():
    vocab_size = 10000
    embedding_dim = 100
    max_length = 100
    
    model = keras.Sequential([
        # Embedding layer
        keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        
        # LSTM layers
        keras.layers.LSTM(64, dropout=0.3, recurrent_dropout=0.3, return_sequences=True),
        keras.layers.LSTM(32, dropout=0.3, recurrent_dropout=0.3),
        
        # Dense layers
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(1, activation='sigmoid')  # Binary classification
    ])
    
    return model

# 3. Bidirectional LSTM
def create_bidirectional_lstm():
    model = keras.Sequential([
        # Bidirectional LSTM layers
        keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=True), 
                                 input_shape=(60, 1)),
        keras.layers.Dropout(0.2),
        
        keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=True)),
        keras.layers.Dropout(0.2),
        
        keras.layers.Bidirectional(keras.layers.LSTM(32)),
        keras.layers.Dropout(0.2),
        
        # Output layer
        keras.layers.Dense(1)
    ])
    
    return model

# Create models
time_series_model = create_lstm_model((60, 1))
text_model = create_text_classifier()
bi_lstm_model = create_bidirectional_lstm()
        """, language="python")
    
    with code_tabs[3]:  # Transfer Learning
        st.markdown("### ⚡ Transfer Learning Implementation")
        
        st.code("""
# Transfer Learning with Pre-trained Models
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import ResNet50, VGG16

# 1. Basic Transfer Learning
def create_transfer_model(base_model_name='ResNet50', num_classes=10):
    # Load pre-trained model
    if base_model_name == 'ResNet50':
        base_model = ResNet50(weights='imagenet', include_top=False, 
                            input_shape=(224, 224, 3))
    elif base_model_name == 'VGG16':
        base_model = VGG16(weights='imagenet', include_top=False, 
                         input_shape=(224, 224, 3))
    
    # Freeze base model layers
    base_model.trainable = False
    
    # Add custom classification head
    model = keras.Sequential([
        base_model,
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    return model, base_model

# 2. Fine-tuning Strategy
def fine_tune_model(model, base_model, learning_rate=1e-5):
    # Unfreeze top layers of base model
    base_model.trainable = True
    
    # Fine-tune from this layer onwards
    fine_tune_at = len(base_model.layers) - 20  # Last 20 layers
    
    # Freeze all layers except the last few
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
    
    # Use a lower learning rate for fine-tuning
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# 3. Progressive Training
def progressive_transfer_learning(train_data, val_data, num_classes=10):
    # Stage 1: Train only classification head
    print("Stage 1: Training classification head...")
    model, base_model = create_transfer_model(num_classes=num_classes)
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Initial training with frozen base
    history1 = model.fit(
        train_data,
        epochs=10,
        validation_data=val_data
    )
    
    # Stage 2: Fine-tune with lower learning rate
    print("Stage 2: Fine-tuning...")
    model = fine_tune_model(model, base_model, learning_rate=1e-5)
    
    # Fine-tuning
    history2 = model.fit(
        train_data,
        epochs=10,
        validation_data=val_data
    )
    
    return model, (history1, history2)

# Create transfer learning model
model, base_model = create_transfer_model(num_classes=10)
print("Transfer learning model created!")
        """, language="python")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #7f8c8d; padding: 20px;'>
    <h3>🎓 Deep Learning Master Class Complete!</h3>
    <p>You've explored neural networks, architectures, training, evaluation, and real-world applications.</p>
    <p>Keep learning and building amazing AI systems! 🚀</p>
</div>
""", unsafe_allow_html=True)
                
                pred_col2.markdown(f"**True Label:** {true_label}")
                pred_col2.markdown(f"**Predicted Label:** {predicted_label}")
                pred_col2.markdown(f"**Confidence:** {confidence:.1f}%")
                
                if true_label == predicted_label:
                    pred_col2.success("✅ Correct Prediction!")
                else:
                    pred_col2.error("❌ Incorrect Prediction")
                
                # Show prediction probabilities
                prob_data = pd.DataFrame({
                    'Class': range(len(prediction[0])),
                    'Probability': prediction[0]
                })
                
                fig = px.bar(prob_data, x='Class', y='Probability', 
                           title="Prediction Probabilities")
                fig.update_layout(height=300)
                pred_col2.plotly_chart(fig, use_container_width=True)

# Section: Model Evaluation
elif current_section == "evaluation":
    st.markdown('<h2 class="section-header">📈 Model Evaluation & Metrics</h2>', unsafe_allow_html=True)
    
    if not st.session_state.model_trained:
        st.warning("⚠️ Please train a model first in the Interactive Demo section!")
        st.markdown("### 📚 Evaluation Metrics Overview")
        
        # Show evaluation concepts even without trained model
        eval_concepts = {
            "🎯 Accuracy": "Percentage of correct predictions",
            "📊 Precision": "True Positives / (True Positives + False Positives)",
            "📈 Recall": "True Positives / (True Positives + False Negatives)",
            "🎪 F1-Score": "Harmonic mean of Precision and Recall",
            "📉 Loss": "Measure of prediction error",
            "🔄 Confusion Matrix": "Table showing actual vs predicted classifications",
            "📈 ROC Curve": "Trade-off between True Positive Rate and False Positive Rate"
        }
        
        for metric, description in eval_concepts.items():
            st.markdown(f"**{metric}**: {description}")
    
    else:
        # Comprehensive evaluation of trained model
        x_test, y_test, y_test_cat = st.session_state.test_data
        model = st.session_state.trained_model
        
        # Make predictions
        with st.spinner("Evaluating model performance..."):
            y_pred = model.predict(x_test, verbose=0)
            y_pred_classes = np.argmax(y_pred, axis=1)
            
            # Calculate metrics
            test_loss, test_accuracy = model.evaluate(x_test, y_test_cat, verbose=0)
        
        # Display key metrics
        st.markdown("### 📊 Overall Performance")
        
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        
        with metric_col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>🎯 Test Accuracy</h3>
                <h2>{test_accuracy * 100:.2f}%</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with metric_col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>📉 Test Loss</h3>
                <h2>{test_loss:.4f}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with metric_col3:
            precision = len(y_test[y_test == y_pred_classes]) / len(y_test)
            st.markdown(f"""
            <div class="metric-card">
                <h3>📊 Precision</h3>
                <h2>{precision * 100:.2f}%</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with metric_col4:
            num_classes = len(np.unique(y_test))
            st.markdown(f"""
            <div class="metric-card">
                <h3>🏷️ Classes</h3>
                <h2>{num_classes}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        # Detailed analysis
        analysis_col1, analysis_col2 = st.columns(2)
        
        with analysis_col1:
            st.markdown("### 🔄 Confusion Matrix")
            
            # Calculate confusion matrix
            cm = confusion_matrix(y_test, y_pred_classes)
            
            # Create heatmap
            fig = px.imshow(cm, 
                          text_auto=True, 
                          aspect="auto",
                          color_continuous_scale='Blues',
                          title="Confusion Matrix")
            fig.update_layout(
                xaxis_title="Predicted Label",
                yaxis_title="True Label",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Interpretation
            st.markdown("""
            **How to read:**
            - Diagonal elements: Correct predictions
            - Off-diagonal: Misclassifications
            - Darker colors: Higher values
            """)
        
        with analysis_col2:
            st.markdown("### 📈 Training History")
            
            if st.session_state.training_history:
                history = st.session_state.training_history.history
                
                fig = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=['Model Loss', 'Model Accuracy']
                )
                
                epochs_range = list(range(1, len(history['loss']) + 1))
                
                # Loss subplot
                fig.add_trace(
                    go.Scatter(x=epochs_range, y=history['loss'], name='Training Loss'),
                    row=1, col=1
                )
                if 'val_loss' in history:
                    fig.add_trace(
                        go.Scatter(x=epochs_range, y=history['val_loss'], name='Validation Loss'),
                        row=1, col=1
                    )
                
                # Accuracy subplot
                fig.add_trace(
                    go.Scatter(x=epochs_range, y=history['accuracy'], name='Training Accuracy'),
                    row=2, col=1
                )
                if 'val_accuracy' in history:
                    fig.add_trace(
                        go.Scatter(x=epochs_range, y=history['val_accuracy'], name='Validation Accuracy'),
                        row=2, col=1
                    )
                
                fig.update_xaxes(title_text="Epoch", row=2, col=1)
                fig.update_yaxes(title_text="Loss", row=1, col=1)
                fig.update_yaxes(title_text="Accuracy", row=2, col=1)
                fig.update_layout(height=500)
                
                st.plotly_chart(fig, use_container_width=True)
        
        # ROC Curve (for multiclass)
        st.markdown("### 📈 ROC Curves (One-vs-Rest)")
        
        # Convert to binary format for ROC calculation
        lb = LabelBinarizer()
        y_test_binary = lb.fit_transform(y_test)
        
        # If binary classification, reshape
        if y_test_binary.shape[1] == 1:
            y_test_binary = np.hstack([1-y_test_binary, y_test_binary])
        
        fig = go.Figure()
        
        # Calculate ROC for each class
        for i in range(min(y_test_binary.shape[1], 10)):  # Limit to 10 classes for visibility
            if y_test_binary.shape[1] > i:
                fpr, tpr, _ = roc_curve(y_test_binary[:, i], y_pred[:, i])
                roc_auc = auc(fpr, tpr)
                
                fig.add_trace(go.Scatter(
                    x=fpr, y=tpr,
                    name=f'Class {i} (AUC = {roc_auc:.2f})',
                    mode='lines'
                ))
        
        # Add diagonal line
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            line=dict(dash='dash', color='gray'),
            name='Random Classifier'
        ))
        
        fig.update_layout(
            title='ROC Curves for Each Class',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Classification Report
        st.markdown("### 📋 Detailed Classification Report")
        
        report = classification_report(y_test, y_pred_classes, output_dict=True)
        
        # Convert to DataFrame for better visualization
        report_df = pd.DataFrame(report).transpose()
        report_df = report_df.round(3)
        
        # Style the dataframe
        st.dataframe(
            report_df,
            use_container_width=True
        )

# Section: Real-World Applications
elif current_section == "applications":
    st.markdown('<h2 class="section-header">🌍 Real-World Applications</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    ### 🚀 Deep Learning in Action
    
    Deep Learning has revolutionized numerous industries and applications. 
    Let's explore some of the most impactful use cases!
    """)
    
    # Application categories
    app_tabs = st.tabs([
        "🖼️ Computer Vision", 
        "🗣️ Natural Language", 
        "🎵 Audio & Speech", 
        "🎮 Gaming & Robotics",
        "🏥 Healthcare",
        "🚗 Autonomous Systems"
    ])
    
    with app_tabs[0]:  # Computer Vision
        st.markdown("### 🖼️ Computer Vision Applications")
        
        cv_col1, cv_col2 = st.columns(2)
        
        with cv_col1:
            st.markdown("""
            **📱 Image Classification**
            - Photo tagging and organization
            - Medical image diagnosis
            - Quality control in manufacturing
            - Content moderation
            
            **🎯 Object Detection**
            - Autonomous vehicle navigation
            - Security and surveillance
            - Retail inventory management
            - Sports analytics
            
            **🎨 Image Generation**
            - Art creation and style transfer
            - Photo enhancement and restoration
            - Synthetic data generation
            - Fashion design assistance
            """)
        
        with cv_col2:
            # Create a sample computer vision workflow
            st.markdown("#### 🔍 CV Pipeline Example")
            
            cv_pipeline = {
                'Stage': ['Input Image', 'Preprocessing', 'Feature Extraction', 'Classification', 'Output'],
                'Description': [
                    'Raw pixel data',
                    'Resize, normalize',
                    'Convolution layers',
                    'Dense layers',
                    'Predicted class'
                ],
                'Size': [224*224*3, 224*224*3, 7*7*512, 1000, 1]
            }
            
            pipeline_df = pd.DataFrame(cv_pipeline)
            
            fig = go.Figure(data=go.Scatter(
                x=pipeline_df['Stage'],
                y=pipeline_df['Size'],
                mode='markers+lines',
                marker=dict(size=15, color='#3498db'),
                line=dict(width=3)
            ))
            
            fig.update_layout(
                title="Computer Vision Processing Pipeline",
                xaxis_title="Processing Stage",
                yaxis_title="Data Size",
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)