"""
Обучение нейронных сетей для классификации текста:
1. Тональность (sentiment): positive/negative
2. Стиль (style): colloquial/journalistic/literary

Использует semi-supervised learning (pseudo-labeling) с нейронной сетью на PyTorch.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import re
import joblib

# Пути к данным
DATA_DIR = Path(__file__).parent / "data"
MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

# Параметры модели
MAX_WORDS = 1000  # Максимальное количество слов в словаре
MAX_LEN = 50      # Максимальная длина последовательности
EMBEDDING_DIM = 128  # Размерность embedding слоя
BATCH_SIZE = 4
EPOCHS = 200  # Больше эпох для простой модели
LEARNING_RATE = 0.001


class TextDataset(Dataset):
    """Dataset для текстов"""
    def __init__(self, texts, labels, word_to_idx, max_len):
        self.texts = texts
        self.labels = labels
        self.word_to_idx = word_to_idx
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx] if self.labels is not None else -1
        
        # Преобразуем текст в последовательность индексов
        words = text.lower().split()
        sequence = [self.word_to_idx.get(word, 0) for word in words[:self.max_len]]
        sequence = sequence + [0] * (self.max_len - len(sequence))
        
        return torch.LongTensor(sequence), torch.LongTensor([label])[0]


class TextClassifier(nn.Module):
    """Нейронная сеть для классификации текста (упрощённая для маленьких данных)"""
    def __init__(self, vocab_size, embedding_dim, num_classes, max_len=50):
        super(TextClassifier, self).__init__()
        
        # Слой 1: Embedding - преобразует индексы слов в плотные векторы
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Слой 2: Глобальное усреднение (проще чем LSTM для маленьких данных)
        # Берём среднее всех embedding векторов в последовательности
        self.max_len = max_len
        
        # Слой 3: Dense - полносвязный слой с активацией ReLU
        self.fc1 = nn.Linear(embedding_dim, 32)
        self.relu = nn.ReLU()
        
        # Слой 4: Dropout - регуляризация
        self.dropout = nn.Dropout(0.3)
        
        # Слой 5: Выходной слой - вероятности классов
        self.fc2 = nn.Linear(32, num_classes)
    
    def forward(self, x):
        # x shape: (batch_size, seq_len)
        x = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        # Усредняем по последовательности (глобальное усреднение)
        x = x.mean(dim=1)  # (batch_size, embedding_dim)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def build_vocab(texts, max_words):
    """Создать словарь из текстов"""
    word_counts = Counter()
    for text in texts:
        words = text.lower().split()
        word_counts.update(words)
    
    # Берём самые частые слова
    most_common = word_counts.most_common(max_words - 1)  # -1 для <UNK>
    word_to_idx = {'<UNK>': 0}
    for word, count in most_common:
        word_to_idx[word] = len(word_to_idx)
    
    return word_to_idx


def load_data(task_type: str):
    """
    Загрузить размеченные и неразмеченные данные.
    
    Args:
        task_type: 'sentiment' или 'style'
    
    Returns:
        X_labeled, y_labeled, X_unlabeled
    """
    # Загружаем размеченные данные
    labeled_file = DATA_DIR / f"{task_type}_labeled.csv"
    df_labeled = pd.read_csv(labeled_file, quotechar='"', escapechar='\\')
    X_labeled = df_labeled["text"].values.tolist()
    y_labeled = df_labeled.iloc[:, 1].values
    
    # Загружаем неразмеченные данные
    unlabeled_file = DATA_DIR / f"{task_type}_unlabeled.csv"
    df_unlabeled = pd.read_csv(unlabeled_file, quotechar='"', escapechar='\\')
    X_unlabeled = df_unlabeled["text"].values.tolist()
    
    print(f"\nДанные для задачи '{task_type}':")
    print(f"   Размеченных примеров: {len(X_labeled)}")
    print(f"   Неразмеченных примеров: {len(X_unlabeled)}")
    print(f"   Классы: {np.unique(y_labeled)}")
    
    return X_labeled, y_labeled, X_unlabeled


def train_semi_supervised_neural_network(X_labeled, y_labeled, X_unlabeled, task_type: str):
    """
    Обучить нейронную сеть с использованием semi-supervised learning (pseudo-labeling).
    
    Args:
        X_labeled: размеченные тексты
        y_labeled: метки для размеченных текстов
        X_unlabeled: неразмеченные тексты
        task_type: тип задачи ('sentiment' или 'style')
    
    Returns:
        Обученная модель, word_to_idx и LabelEncoder
    """
    print(f"\nОбучение нейронной сети для '{task_type}'...")
    
    # Кодируем строковые метки в числа
    label_encoder = LabelEncoder()
    y_labeled_encoded = label_encoder.fit_transform(y_labeled)
    num_classes = len(label_encoder.classes_)
    
    print(f"   Классы: {label_encoder.classes_}")
    print(f"   Количество классов: {num_classes}")
    
    # Объединяем все тексты для создания словаря
    X_all = X_labeled + X_unlabeled
    
    # Создаём словарь
    print("   Создаём словарь...")
    word_to_idx = build_vocab(X_all, MAX_WORDS)
    vocab_size = len(word_to_idx)
    
    print(f"   Размер словаря: {vocab_size}")
    
    # Создаём нейронную сеть
    print("   Создаём нейронную сеть...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   Устройство: {device}")
    
 
    model = TextClassifier(vocab_size, EMBEDDING_DIM, num_classes, max_len=MAX_LEN).to(device)
    
    print("\nАрхитектура нейронной сети:")
    print(model)
    
    # Подсчитываем параметры
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n   Всего параметров: {total_params:,}")
    print(f"   Обучаемых параметров: {trainable_params:,}")
    
    # Создаём датасеты
    labeled_dataset = TextDataset(X_labeled, y_labeled_encoded, word_to_idx, MAX_LEN)
    labeled_loader = DataLoader(labeled_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Оптимизатор и функция потерь
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Вычисляем веса классов для несбалансированных данных
    from collections import Counter
    class_counts = Counter(y_labeled_encoded)
    total = len(y_labeled_encoded)
    class_weights = torch.FloatTensor([total / (num_classes * class_counts[i]) for i in range(num_classes)]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    print(f"   Веса классов: {class_weights.cpu().numpy()}")
    
    # Этап 1: Обучение на размеченных данных
    print("\nЭтап 1: Обучение на размеченных данных...")
    model.train()
    
    best_loss = float('inf')
    patience = 20 
    patience_counter = 0
    
    for epoch in range(EPOCHS):
        total_loss = 0
        correct = 0
        total_samples = 0
        
        for sequences, labels in labeled_loader:
            sequences = sequences.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Подсчитываем точность
            predictions = torch.argmax(outputs, dim=1)
            correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)
        
        avg_loss = total_loss / len(labeled_loader)
        accuracy = correct / total_samples
        
        if (epoch + 1) % 5 == 0:
            # Проверяем распределение предсказаний
            model.eval()
            with torch.no_grad():
                sample_preds = []
                for sequences, labels in labeled_loader:
                    sequences = sequences.to(device)
                    outputs = model(sequences)
                    predictions = torch.argmax(outputs, dim=1)
                    sample_preds.extend(predictions.cpu().numpy())
                pred_dist = Counter(sample_preds)
            model.train()
            
            print(f"   Epoch {epoch + 1}/{EPOCHS}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.3f}")
            print(f"      Распределение предсказаний: {dict(pred_dist)}")
        
        # Early stopping (останавливаемся только если loss не улучшается)
        if avg_loss < best_loss - 0.001:  # Небольшой запас для стабильности
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"   Early stopping на эпохе {epoch + 1}")
                break
    
    # Этап 2: Pseudo-labeling (semi-supervised)
    print("\nЭтап 2: Pseudo-labeling (semi-supervised learning)...")
    
    model.eval()
    unlabeled_dataset = TextDataset(X_unlabeled, None, word_to_idx, MAX_LEN)
    unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Предсказываем метки для неразмеченных данных
    all_predictions = []
    all_probs = []
    
    with torch.no_grad():
        for sequences, _ in unlabeled_loader:
            sequences = sequences.to(device)
            outputs = model(sequences)
            probs = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(probs, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_probs.extend(torch.max(probs, dim=1)[0].cpu().numpy())
    
    all_probs = np.array(all_probs)
    all_predictions = np.array(all_predictions)
    
    # Берём только высокоуверенные предсказания (повысили порог до 0.85)
    confidence_threshold = 0.85
    confident_mask = all_probs >= confidence_threshold
    
    if np.sum(confident_mask) > 0:
        X_pseudo = [X_unlabeled[i] for i in range(len(X_unlabeled)) if confident_mask[i]]
        y_pseudo = all_predictions[confident_mask]
        
        print(f"   Найдено {np.sum(confident_mask)} высокоуверенных предсказаний (порог: {confidence_threshold})")
        
        # Ограничиваем количество pseudo-labeled данных (не больше чем labeled)
        max_pseudo = min(len(X_labeled), len(X_pseudo))
        X_pseudo_limited = X_pseudo[:max_pseudo]
        y_pseudo_limited = y_pseudo[:max_pseudo]
        
        # Объединяем размеченные и псевдо-размеченные данные
        X_combined = X_labeled + X_pseudo_limited
        y_combined = np.concatenate([y_labeled_encoded, y_pseudo_limited])
        
        # Создаём новый датасет
        combined_dataset = TextDataset(X_combined, y_combined, word_to_idx, MAX_LEN)
        combined_loader = DataLoader(combined_dataset, batch_size=BATCH_SIZE, shuffle=True)
        
        # Переобучаем модель на объединённых данных с меньшим learning rate
        print(f"   Переобучение на размеченных + {len(X_pseudo_limited)} псевдо-размеченных данных...")
        
        # Уменьшаем learning rate для более осторожного обучения
        optimizer_fine = optim.Adam(model.parameters(), lr=LEARNING_RATE * 0.1)
        model.train()
        
        for epoch in range(EPOCHS // 3):  # Ещё меньше эпох для второго этапа
            total_loss = 0
            for sequences, labels in combined_loader:
                sequences = sequences.to(device)
                labels = labels.to(device)
                
                optimizer_fine.zero_grad()
                outputs = model(sequences)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer_fine.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / len(combined_loader)
                print(f"   Epoch {epoch + 1}/{EPOCHS // 3}, Loss: {avg_loss:.4f}")
    else:
        print(f"   Не найдено достаточно уверенных предсказаний (порог: {confidence_threshold})")
        print("   Продолжаем с только размеченными данными")
    
    # Оцениваем на размеченных данных
    model.eval()
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for sequences, labels in labeled_loader:
            sequences = sequences.to(device)
            outputs = model(sequences)
            probs = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(outputs, dim=1)
            all_preds.extend(predictions.cpu().numpy())
            all_probs.extend(torch.max(probs, dim=1)[0].cpu().numpy())
    
    y_pred = label_encoder.inverse_transform(all_preds)
    accuracy = accuracy_score(y_labeled, y_pred)
    avg_confidence = np.mean(all_probs)
    
    print(f"\nТочность на размеченных данных: {accuracy:.3f}")
    print(f"Средняя уверенность предсказаний: {avg_confidence:.3f}")
    
    # Проверяем распределение предсказаний
    from collections import Counter
    pred_dist = Counter(y_pred)
    print(f"\nРаспределение предсказаний:")
    for cls, count in pred_dist.items():
        print(f"  {cls}: {count}")
    
    print(f"\nОтчёт по классификации:")
    print(classification_report(y_labeled, y_pred))
    
    return model, word_to_idx, label_encoder


def save_model(model, word_to_idx, label_encoder, task_type: str):
    """Сохранить модель, словарь и LabelEncoder в файлы"""
    model_path = MODELS_DIR / f"{task_type}_model.pth"
    vocab_path = MODELS_DIR / f"{task_type}_vocab.pkl"
    encoder_path = MODELS_DIR / f"{task_type}_encoder.pkl"
    
    # Сохраняем модель PyTorch
    torch.save(model.state_dict(), model_path)
    
    # Сохраняем словарь и encoder через joblib
    joblib.dump(word_to_idx, vocab_path)
    joblib.dump(label_encoder, encoder_path)
    
    print(f"Модель сохранена: {model_path}")
    print(f"Словарь сохранён: {vocab_path}")
    print(f"LabelEncoder сохранён: {encoder_path}")


def main():
    """Главная функция обучения"""
    print("=" * 60)
    print("Обучение нейронных сетей классификации текста")
    print("   Используется semi-supervised learning (pseudo-labeling)")
    print("   Фреймворк: PyTorch")
    print("=" * 60)
    
    # Обучаем модель для тональности
    X_labeled_sent, y_labeled_sent, X_unlabeled_sent = load_data("sentiment")
    model_sentiment, vocab_sentiment, encoder_sentiment = train_semi_supervised_neural_network(
        X_labeled_sent, y_labeled_sent, X_unlabeled_sent, "sentiment"
    )
    save_model(model_sentiment, vocab_sentiment, encoder_sentiment, "sentiment")
    
    print("\n" + "=" * 60)
    
    # Обучаем модель для стиля
    X_labeled_style, y_labeled_style, X_unlabeled_style = load_data("style")
    model_style, vocab_style, encoder_style = train_semi_supervised_neural_network(
        X_labeled_style, y_labeled_style, X_unlabeled_style, "style"
    )
    save_model(model_style, vocab_style, encoder_style, "style")
    
    print("\n" + "=" * 60)
    print("Обучение завершено!")
    print(f"Модели сохранены в: {MODELS_DIR}")


if __name__ == "__main__":
    main()
