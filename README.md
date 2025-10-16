# sentiment-finetune

This project fine-tunes transformer-based models (**DistilBERT**) to classify social media posts into three sentiment categories — **positive**, **negative**, and **neutral**.  
It compares **full fine-tuning** and **parameter-efficient fine-tuning (LoRA)** to evaluate model performance.

---

## Repository Structure
```
sentiment-finetune/
├── src/
│ ├── run.py # Main script to train and evaluate models
│ └── utils.py # Data loading, preprocessing, and metrics
├── data/
│ └── README.txt # Placeholder (dataset should be downloaded separately)
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Dataset

This project uses the **[Social Media Sentiments Analysis Dataset](https://www.kaggle.com/datasets/kashishparmar02/social-media-sentiments-analysis-dataset)** from Kaggle.

### To set up the dataset:
1. Download the dataset manually from the Kaggle link above.  
2. Rename the file to:
sentimentdataset.csv
3. Place it in the `data/` directory:
```
sentiment-finetune/
├── data/
│ └── sentimentdataset.csv
```

## Installation

1. Clone the repository:
git clone https://github.com/your-username/sentiment-finetune.git
2. Run:
```
cd sentiment-finetune
pip install -r requirements.txt
python src/run.py
```
## Model training process
Loads and preprocesses the dataset

Fine-tunes a transformer model (DistilBERT)

Optionally applies LoRA (Low-Rank Adaptation)

Evaluates the model on test data

Outputs metrics including Accuracy, F1, Precision, and Recall
```
Example Output:
===== Summary Table =====
           Method    Accuracy      F1    Precision   Recall
Full Fine-Tuning     0.727        0.641    0.814      0.639
LoRA                 0.409        0.194    0.136      0.333
```
All dependencies are listed in requirements.txt.
Key packages include:

transformers

datasets

scikit-learn

pandas

numpy

torch
