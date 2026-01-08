# 🚀 Real-Time Competitor Strategy Tracker for E-Commerce  
### Infosys Springboard | AI Internship Project

---

## 📌 Project Overview

The **Real-Time Competitor Strategy Tracker** is an AI-driven market intelligence system designed to monitor and analyze competitor pricing, product availability, customer sentiment, and promotional strategies across e-commerce platforms in real time.

By automating data collection and analysis, the system enables businesses to make **data-driven pricing and strategic decisions**, reduce manual monitoring effort, and respond quickly to dynamic market conditions.

This project was developed as part of the **Artificial Intelligence Internship at Infosys Springboard**.

---

## 🎯 Problem Statement

E-commerce markets are highly dynamic, with frequent changes in competitor prices and promotions.  
Manual competitor tracking is:
- Time-consuming  
- Error-prone  
- Not scalable  

There is a need for an **automated, real-time competitor intelligence system** that delivers actionable insights for strategic decision-making.

---

## 🏁 Project Objectives

- Automate competitor data collection  
- Track pricing, offers, and product availability  
- Analyze customer sentiment and market trends  
- Enable dynamic pricing strategies  
- Support real-time, data-driven business decisions  

---

# 🚀 Milestone 1: Infrastructure, Tooling & Foundations

### 🎯 Objective
Establish a stable development environment and validate the mathematical foundations of deep learning by transitioning from low-level numerical implementations to industry-standard frameworks.

---

## 🛠️ Tech Stack & Tools

### Computational Engines
- **NumPy** – Implemented neural network logic from scratch  
- **CuPy** – GPU-accelerated matrix computations using NVIDIA CUDA  

### Deep Learning Frameworks
- **PyTorch** – Dynamic computational graphs with `nn.Module`  
- **TensorFlow / Keras** – High-level model development and benchmarking  

### Evaluation & Visualization
- **Matplotlib** – Training and validation loss/accuracy visualization  
- **Scikit-learn** – Performance metrics (`accuracy_score`)  

---

## 🧠 Model Development

### From-Scratch Neural Network
- Implemented a `GateNeuralNetwork` class  
- Manual backpropagation using the chain rule  
- Weight change tracking (`FWC`, `MWC`, `LWC`) to validate learning  

### Framework-Based MLP (TensorFlow/Keras)
- **Input Layer**: 28×28 → 784-dimensional vector  
- **Hidden Layers**: Dense (128, 64) with Sigmoid activation  
- **Output Layer**: 10 neurons with Softmax  
- **Optimizer**: RMSprop  
- **Loss Function**: Binary Crossentropy  

---

## 📊 Results & Insights

- >91% accuracy achieved in the first epoch  
- Final validation accuracy of ~98.06%  
- Optimal generalization observed around Epoch 25  

---

## 📦 Deliverables
- GPU-enabled development environment  
- Automated MNIST preprocessing pipeline  
- Trained model weights exported in `.npy` format  

---

# 🕷️ Milestone 2: Web Scraping & Data Aggregation

### 🎯 Objective
Build a real-world data pipeline to extract, clean, and structure unstructured competitor data from e-commerce platforms.

---

## 🛠️ Tech Stack

### Web Automation & Scraping
- **Playwright** – JavaScript-heavy page automation  
- **BeautifulSoup4 & lxml** – HTML parsing and deep scraping  

### Data Processing
- **Pandas** – Data cleaning, aggregation, CSV/JSON export  
- **Regex (re)** – Price, rating, and text normalization  

### Intelligence Layer
- **Transformers (RoBERTa)** – Headline sentiment analysis  

---

## 🧠 Engineering Highlights

- Master–detail crawling architecture  
- Dynamic pagination handling  
- Ethical scraping with rate-limiting  
- Fault-tolerant execution using error handling  

---

## 📊 Outputs
- `books.csv` with 1,000+ structured records  
- Top 5 trending keywords from live news feeds  
- Sentiment polarity index ranging from -1 to +1  

---

# 📊 Milestone 3: AI Sentiment Analysis & Semantic Modeling

### 🎯 Objective
Extract customer intelligence using deep NLP and semantic similarity techniques to rank products based on quality and market appeal.

---

## 🛠️ Techniques Used
- **RoBERTa (cardiffnlp)** for contextual sentiment analysis  
- **Jaccard Distance** for lexical diversity  
- **TF-IDF + Cosine Similarity** for semantic alignment  

---

## 📈 Popularity Index Formula

| Feature | Weight | Description |
|------|------|------|
| Sentiment Score | 40% | Emotional appeal |
| Cosine Similarity | 40% | Semantic depth |
| Jaccard Distance | 20% | Information richness |

---

## 🧪 Results
- Generated `milestone3_popularity_report.csv`  
- Strong semantic clustering observed  
- RoBERTa outperformed traditional NLP models on complex themes  

---

## ▶️ Execution
```bash
pip install transformers torch scikit-learn pandas nltk
python milestone3_analysis.py
