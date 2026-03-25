"""
Application Flask pour classifier les SMS en SPAM/HAM
"""

from flask import Flask, render_template, request, jsonify
import pickle
import re
import os
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

app = Flask(__name__)

# Variables globales pour le modèle
model = None
vectorizer = None

# Télécharger les ressources NLTK nécessaires
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('punkt_tab')

def preprocessing(text):
    """
    Preprocessing avec NLTK (compatible Python 3.14)
    Approche simple sans punkt_tab
    """
    # 1. Minuscules
    text = text.lower()
    
    # 2. Remplacer nombres et caractères spéciaux
    text = re.sub(r"[^a-zA-Z\s]", " ", str(text))
    
    # 3. Tokenization simple (split)
    tokens = text.split()
    
    # 4. Supprimer stop words et tokens courts
    stop_words = set(stopwords.words('english'))
    tokens = [
        token for token in tokens
        if token not in stop_words and len(token) >= 3
    ]
    
    # 5. Reconstruire
    return " ".join(tokens)

def load_or_train_model():
    """
    Charger le modèle existant ou l'entraîner s'il n'existe pas
    """
    global model, vectorizer
    
    print("Initialisation du modèle...")
    
    model_path = 'model.pkl'
    vectorizer_path = 'vectorizer.pkl'
    
    # Vérifier si les modèles sauvegardés existent
    if os.path.exists(model_path) and os.path.exists(vectorizer_path):
        print(" Chargement des modèles existants...")
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)
        print(" Modèles chargés avec succès!")
        return True
    
    else:
        print(" Entraînement du modèle (première fois)...")
        print("   Téléchargement du dataset...")
        
        # Télécharger et charger les données
        import urllib.request
        import zipfile
        
        try:
            # Télécharger le dataset
            url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip'
            print(f" Téléchargement depuis {url}...")
            urllib.request.urlretrieve(url, 'spam_data.zip')
            print(" Téléchargement terminé")
            
            # Extraire
            with zipfile.ZipFile('spam_data.zip', 'r') as zip_ref:
                zip_ref.extractall()
            
            # Charger
            df = pd.read_csv('SMSSpamCollection', sep='\t', names=['label', 'message'])
            print(f" Dataset chargé: {len(df)} SMS")
            
            # Preprocessing
            print(" Preprocessing en cours...")
            df['message_clean'] = df['message'].apply(preprocessing)
            
            # Train/Test split
            X = df['message_clean']
            y = df['label'].map({'ham': 0, 'spam': 1})
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Vectorization
            print("  Vectorization TF-IDF...")
            vectorizer = TfidfVectorizer(max_features=5000)
            X_train_tfidf = vectorizer.fit_transform(X_train)
            
            # Training
            print("  Entraînement Naive Bayes...")
            model = MultinomialNB()
            model.fit(X_train_tfidf, y_train)
            
            # Sauvegarder
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            with open(vectorizer_path, 'wb') as f:
                pickle.dump(vectorizer, f)
            
            print(" Modèle entraîné et sauvegardé!")
            return True
            
        except Exception as e:
            print(f" Erreur lors de l'entraînement: {e}")
            import traceback
            traceback.print_exc()
            return False

@app.route('/')
def index():
    """Route principale - affiche la page HTML"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """API - Prédire si un SMS est SPAM ou HAM"""
    try:
        data = request.get_json()
        message = data.get('message', '').strip()
        
        if not message:
            return jsonify({'error': 'Message vide'}), 400
        
        # Preprocessing
        cleaned = preprocessing(message)
        
        if not cleaned:
            return jsonify({'error': 'Message invalide après nettoyage'}), 400
        
        # Vectorization
        vectorized = vectorizer.transform([cleaned])
        
        # Prédiction
        prediction = model.predict(vectorized)[0]
        probability = model.predict_proba(vectorized)[0]
        
        label = "SPAM" if prediction == 1 else "HAM"
        confidence_spam = float(probability[1])
        confidence_ham = float(probability[0])
        
        return jsonify({
            'message': message,
            'label': label,
            'confidence_spam': confidence_spam,
            'confidence_ham': confidence_ham,
            'success': True
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check - vérifier que l'API fonctionne"""
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'vectorizer_loaded': vectorizer is not None
    })

if __name__ == '__main__':
    # Initialiser le modèle
    if load_or_train_model():
        print("\n" + "="*50)
        print("Serveur en cours de démarrage...")
        print("   http://localhost:5000")
        print("="*50 + "\n")
        app.run(debug=True, port=5000)
    else:
        print("Impossible de démarrer le serveur sans le modèle")
