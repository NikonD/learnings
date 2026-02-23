"""
Скрипт для тестирования качества моделей на известных примерах.
"""
from predict import predict_sentiment, predict_style

# Тестовые примеры с ожидаемыми результатами
test_cases = [
    # (текст, ожидаемый sentiment, ожидаемый style)
    ("Это отличный фильм, мне очень понравилось!", "positive", "colloquial"),
    ("Привет, как дела? Давай встретимся завтра.", "positive", "colloquial"),
    ("По данным статистики, уровень безработицы снизился.", "positive", "journalistic"),
    ("Величественные горы возвышались над долиной.", "positive", "literary"),
    ("Ужасное обслуживание, никогда больше не приду.", "negative", "colloquial"),
    ("Слушай, а ты не знаешь, где ближайший магазин?", None, "colloquial"),
    ("Эксперты отмечают положительную динамику развития.", "positive", "journalistic"),
    ("Время текло медленно, словно густой мёд.", None, "literary"),
]

print("=" * 70)
print("Тестирование моделей")
print("=" * 70)

correct_sentiment = 0
correct_style = 0
total_sentiment = 0
total_style = 0

for text, expected_sentiment, expected_style in test_cases:
    predicted_sentiment = predict_sentiment(text)
    predicted_style = predict_style(text)
    
    print(f"\nТекст: {text}")
    print(f"   Sentiment: {predicted_sentiment} (ожидалось: {expected_sentiment})")
    print(f"   Style: {predicted_style} (ожидалось: {expected_style})")
    
    if expected_sentiment:
        total_sentiment += 1
        if predicted_sentiment == expected_sentiment:
            correct_sentiment += 1
            print(f"   Sentiment правильный")
        else:
            print(f"   Sentiment неправильный")
    
    if expected_style:
        total_style += 1
        if predicted_style == expected_style:
            correct_style += 1
            print(f"   Style правильный")
        else:
            print(f"   Style неправильный")

print("\n" + "=" * 70)
print("Результаты:")
if total_sentiment > 0:
    print(f"   Sentiment: {correct_sentiment}/{total_sentiment} ({100*correct_sentiment/total_sentiment:.1f}%)")
if total_style > 0:
    print(f"   Style: {correct_style}/{total_style} ({100*correct_style/total_style:.1f}%)")
print("=" * 70)
