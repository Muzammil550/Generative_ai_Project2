# Generative_ai
"Depression Tweet Classification using Neural Network & Random Forest"


📌 Overview
This project aims to classify tweets as depressed or not depressed using Sentence Transformers (MiniLM) for text embeddings and two classification models:

A Neural Network (PyTorch)
A Random Forest Classifier (Sklearn)
🛠 Features
Preprocesses tweet text using NLTK (stopword removal, lemmatization)
Converts text to embeddings using Sentence-BERT (MiniLM)
Implements a PyTorch-based Neural Network classifier
Implements a Random Forest classifier for comparison
Saves the best-performing Neural Network model
🚀 Installation
1️⃣ Clone the Repository
bash
Copy
Edit
git clone https://github.com/your-username/depression-tweet-classifier.git
cd depression-tweet-classifier
2️⃣ Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
🔹 Usage
Training the Model
Run the following command to train both models:

bash
Copy
Edit
python Depression_Tweet_NN.py
The best Neural Network model will be saved as best_model.pth.
📊 Results
The trained models are evaluated using:

Accuracy
Classification Report (Precision, Recall, F1-score)
📜 Dataset
The dataset should contain tweets labeled as 0 (Not Depressed) or 1 (Depressed). Ensure your CSV file has the following columns:

arduino
Copy
Edit
text,label
"I feel so alone and sad.",1
"Life is amazing!",0
Note: The dataset file should be named depression_tweets.csv and placed in the project directory.

🔥 Technologies Used
Python
PyTorch
Scikit-Learn
NLTK
Sentence Transformers
Pandas & NumPy
📌 To-Do
 Implement more advanced deep learning models (e.g., LSTMs, GRUs)
 Experiment with different text embedding models
 Fine-tune hyperparameters for better accuracy
📝 License
This project is licensed under the MIT License.
