"""
Использование обученных нейронных сетей для предсказания тональности и стиля текста.
"""
import joblib
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

MODELS_DIR = Path(__file__).parent / "models"
MAX_LEN = 50
EMBEDDING_DIM = 128


class TextClassifier(nn.Module):
    """Нейронная сеть для классификации текста"""
    def __init__(self, vocab_size, embedding_dim, num_classes, max_len=50):
        super(TextClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.max_len = max_len
        self.fc1 = nn.Linear(embedding_dim, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(32, num_classes)
    
    def forward(self, x):
        x = self.embedding(x)
        x = x.mean(dim=1)  # Глобальное усреднение
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def load_model_files(task_type: str):
    """Загрузить модель, словарь и LabelEncoder из файлов"""
    model_path = MODELS_DIR / f"{task_type}_model.pth"
    vocab_path = MODELS_DIR / f"{task_type}_vocab.pkl"
    encoder_path = MODELS_DIR / f"{task_type}_encoder.pkl"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Модель не найдена: {model_path}. Сначала запустите train.py")
    if not vocab_path.exists():
        raise FileNotFoundError(f"Словарь не найден: {vocab_path}. Сначала запустите train.py")
    if not encoder_path.exists():
        raise FileNotFoundError(f"LabelEncoder не найден: {encoder_path}. Сначала запустите train.py")
    
    # Загружаем словарь и encoder
    word_to_idx = joblib.load(vocab_path)
    encoder = joblib.load(encoder_path)
    
    # Создаём модель и загружаем веса
    vocab_size = len(word_to_idx)
    num_classes = len(encoder.classes_)
    
    model = TextClassifier(vocab_size, EMBEDDING_DIM, num_classes)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    return model, word_to_idx, encoder


def text_to_sequence(text, word_to_idx, max_len):
    """Преобразовать текст в последовательность индексов"""
    words = text.lower().split()
    sequence = [word_to_idx.get(word, 0) for word in words[:max_len]]
    sequence = sequence + [0] * (max_len - len(sequence))
    return torch.LongTensor([sequence])


def predict_sentiment(text: str, model=None, word_to_idx=None, encoder=None):
    """
    Предсказать тональность текста (positive/negative).
    
    Args:
        text: текст для анализа
        model: модель (если None, загрузится автоматически)
        word_to_idx: словарь (если None, загрузится автоматически)
        encoder: LabelEncoder (если None, загрузится автоматически)
    
    Returns:
        Предсказанная метка (строка)
    """
    if model is None or word_to_idx is None or encoder is None:
        model, word_to_idx, encoder = load_model_files("sentiment")
    
    # Преобразуем текст в последовательность
    sequence = text_to_sequence(text, word_to_idx, MAX_LEN)
    
    # Предсказываем
    with torch.no_grad():
        outputs = model(sequence)
        probs = torch.softmax(outputs, dim=1)
        prediction_encoded = torch.argmax(outputs, dim=1).item()
        confidence = probs[0][prediction_encoded].item()
    
    # Декодируем обратно в строку
    prediction = encoder.inverse_transform([prediction_encoded])[0]
    
    return prediction


def predict_style(text: str, model=None, word_to_idx=None, encoder=None):
    """
    Предсказать стиль текста (colloquial/journalistic/literary).
    
    Args:
        text: текст для анализа
        model: модель (если None, загрузится автоматически)
        word_to_idx: словарь (если None, загрузится автоматически)
        encoder: LabelEncoder (если None, загрузится автоматически)
    
    Returns:
        Предсказанная метка (строка)
    """
    if model is None or word_to_idx is None or encoder is None:
        model, word_to_idx, encoder = load_model_files("style")
    
    # Преобразуем текст в последовательность
    sequence = text_to_sequence(text, word_to_idx, MAX_LEN)
    
    # Предсказываем
    with torch.no_grad():
        outputs = model(sequence)
        probs = torch.softmax(outputs, dim=1)
        prediction_encoded = torch.argmax(outputs, dim=1).item()
        confidence = probs[0][prediction_encoded].item()
    
    # Декодируем обратно в строку
    prediction = encoder.inverse_transform([prediction_encoded])[0]
    
    return prediction


def analyze_text(text: str):
    """
    Полный анализ текста: тональность + стиль.
    
    Args:
        text: текст для анализа
    
    Returns:
        dict с результатами
    """
    sentiment_model, sentiment_vocab, sentiment_encoder = load_model_files("sentiment")
    style_model, style_vocab, style_encoder = load_model_files("style")
    
    sentiment = predict_sentiment(text, sentiment_model, sentiment_vocab, sentiment_encoder)
    style = predict_style(text, style_model, style_vocab, style_encoder)
    
    return {
        "text": text,
        "sentiment": sentiment,
        "style": style
    }


def main():
    """Пример использования"""
    print("=" * 60)
    print("Анализ текста: тональность и стиль")
    print("=" * 60)
    
    # Примеры текстов
    test_texts = [
        "Это отличный фильм, мне очень понравилось!",
        "Привет, как дела? Давай встретимся завтра.",
        "По данным статистики, уровень безработицы снизился.",
        "Величественные горы возвышались над долиной.",
        "Ужасное обслуживание, никогда больше не приду.",
        "избегайте использования эмодзи",
    ]
    
    for text in test_texts:
        result = analyze_text(text)
        print(f"\nТекст: {result['text']}")
        print(f"   Тональность: {result['sentiment']}")
        print(f"   Стиль: {result['style']}")


if __name__ == "__main__":
    main()
