"""
Скрипт для проверки качества данных.
"""
import pandas as pd
from pathlib import Path
from collections import Counter

DATA_DIR = Path(__file__).parent / "data"


def analyze_data(task_type: str):
    """Анализ данных для задачи"""
    print(f"\n{'='*60}")
    print(f"Анализ данных для задачи: {task_type}")
    print(f"{'='*60}")
    
    # Загружаем размеченные данные
    labeled_file = DATA_DIR / f"{task_type}_labeled.csv"
    df_labeled = pd.read_csv(labeled_file, quotechar='"', escapechar='\\')
    
    print(f"\nРазмеченные данные:")
    print(f"  Всего примеров: {len(df_labeled)}")
    
    # Баланс классов
    label_col = df_labeled.columns[1]
    class_counts = Counter(df_labeled[label_col])
    print(f"\n  Баланс классов:")
    for cls, count in class_counts.items():
        percentage = 100 * count / len(df_labeled)
        print(f"    {cls}: {count} ({percentage:.1f}%)")
    
    # Проверка на дубликаты
    duplicates = df_labeled.duplicated(subset=['text']).sum()
    print(f"\n  Дубликаты текстов: {duplicates}")
    
    # Средняя длина текстов
    text_lengths = df_labeled['text'].str.len()
    print(f"\n  Длина текстов:")
    print(f"    Средняя: {text_lengths.mean():.1f} символов")
    print(f"    Мин: {text_lengths.min()} символов")
    print(f"    Макс: {text_lengths.max()} символов")
    
    # Количество слов
    word_counts = df_labeled['text'].str.split().str.len()
    print(f"\n  Количество слов:")
    print(f"    Среднее: {word_counts.mean():.1f} слов")
    print(f"    Мин: {word_counts.min()} слов")
    print(f"    Макс: {word_counts.max()} слов")
    
    # Примеры текстов для каждого класса
    print(f"\n  Примеры текстов по классам:")
    for cls in class_counts.keys():
        examples = df_labeled[df_labeled[label_col] == cls]['text'].head(3).tolist()
        print(f"\n    {cls}:")
        for ex in examples:
            print(f"      - {ex[:60]}...")
    
    # Загружаем неразмеченные данные
    unlabeled_file = DATA_DIR / f"{task_type}_unlabeled.csv"
    df_unlabeled = pd.read_csv(unlabeled_file, quotechar='"', escapechar='\\')
    
    print(f"\n\nНеразмеченные данные:")
    print(f"  Всего примеров: {len(df_unlabeled)}")
    
    # Средняя длина
    text_lengths_un = df_unlabeled['text'].str.len()
    print(f"  Средняя длина: {text_lengths_un.mean():.1f} символов")
    
    # Количество слов
    word_counts_un = df_unlabeled['text'].str.split().str.len()
    print(f"  Среднее количество слов: {word_counts_un.mean():.1f}")
    
    # Проверка на пересечение с размеченными данными
    labeled_texts = set(df_labeled['text'].str.lower())
    unlabeled_texts = set(df_unlabeled['text'].str.lower())
    intersection = labeled_texts & unlabeled_texts
    print(f"\n  Пересечение с размеченными данными: {len(intersection)} текстов")
    
    # Примеры неразмеченных текстов
    print(f"\n  Примеры неразмеченных текстов:")
    for ex in df_unlabeled['text'].head(5).tolist():
        print(f"    - {ex[:60]}...")


if __name__ == "__main__":
    analyze_data("sentiment")
    analyze_data("style")
