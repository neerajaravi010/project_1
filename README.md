# Emotion Classification with Multiple Machine Learning Models
A comprehensive emotion classification project implementing and comparing various machine learning and deep learning models for text-based emotion recognition
## Project Overview
This project implements multiple state-of-the-art machine learning models to classify text into six different emotion categories: sadness, joy, love, anger, fear, and surprise. The project provides a comparative analysis of different approaches including traditional machine learning, deep learning, and transformer-based models
## Dataset
The project uses three CSV files: 
1. training.csv - Training dataset
2. validation.csv - Validation dataset
3. test.csv - Test dataset
## Dataset Structure:
1. Input: Text data (emotional expressions/sentences)
2. Output: Emotion labels 05 corresponding to the six emotions)
3. Classes: 6 emotion categories
 0 Sadness
 1 Joy
 2 Love
 3 Anger
 4 Fear
 5 Surprise
## Models Implemented
1. **BERT Transformer Model**
 * Pre-trained BERT-base-uncased model
 * Fine-tuned for sequence classification
 * Uses BertTokenizer for text preprocessing
 * Implements padding and sequence truncation (max_len=64
2. **Capsule Network with CNN**
 * Custom capsule network architecture
 * Combines Conv1D layers with capsule routing
 * Implements spatial dropout and layer normalization
 * Hyperparameter tuning with cross-validation
3. **Graph Convolutional Network GCN**
 * PyTorch Geometric implementation
 * Two-layer GCN with ReLU activation
 * Dropout regularization
 * TFIDF feature extraction for graph node features
4. **XGBoost Classifier**
 * Gradient boosting implementation
 * TFIDF vectorization pipeline
 * Grid search optimization for hyperparameters
 * Cross-validation for model selection
5. **Random Forest Classifier**
 * Ensemble learning approach
 * TFIDF feature extraction
 * Hyperparameter tuning with GridSearchCV
 * Weighted evaluation metrics
## Technical Stack
**Core Libraries**
 * PyTorch & PyTorch Geometric: For GCN implementation
 * TensorFlow/Keras: For CNN and Transformer models
 * Transformers Hugging Face): For BERT implementation
 * Scikit-learn: For traditional ML models and preprocessing
 * XGBoost: For gradient boosting
 * NLTK: For text preprocessing (stemming)

 **Data Processing**
 * Pandas: Data manipulation and analysis
 * NumPy: Numerical computations
 * CountVectorizer & TF IDF: Text feature extraction
 * BERT Tokenizer: Advanced text tokenization
 
 **Visualization & Analysis**
 * Matplotlib & Seaborn: Data visualization
 * Label distribution analysis
 * Performance metrics visualization
## Model Evaluation
 Each model is evaluated using comprehensive metrics:
 * Accuracy: Overall classification accuracy
 * Precision: Weighted precision across classes
 * Recall: Weighted recall across classes
 * F1Score: Weighted F1-score for balanced evaluation

**Cross-Validation Strategy**
 * Stratified KFold: Ensures balanced class distribution
 * Grid Search: Hyperparameter optimization
 * Multiple folds: Robust performance estimation

##  Installation & Setup
 **Prerequisites**
 pip install torch torch_geometric
 pip install tensorflow transformers
 pip install scikit-learn xgboost
 pip install pandas numpy matplotlib seaborn
 pip install nltk
 **Required Files**
 Ensure you have the following CSV files in your project directory:
 * training.csv
 * validation.csv
 * test.csv

**Usage**
1. Clone the repository
2. Install dependencies using the requirements above
3. Place your dataset files in the project directory
4. Run the Jupyter notebook Neeraja_Ravi_10632860.ipynb
5. Execute cells sequentially to train and evaluate all models

 **Key Features**
 * Automated preprocessing pipeline
 * Hyperparameter optimization for all models
 * Cross-validation for robust evaluation
 * Comparative analysis across different approaches
 * Visualization of data distribution and results

 **Model Architecture Details**
  `Pre-trained: bert-base-uncased
  Max sequence length: 64 tokens
  Fine-tuning: 5 epochs
  Optimizer: Adam
  Loss: Sparse categorical crossentropy`

