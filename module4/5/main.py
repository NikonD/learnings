import numpy as np
import matplotlib.pyplot as plt

# 2D семантическое пространство
# x: живое(-) → техника(+)
# y: негатив(-) → позитив(+)

embeddings = {
    "кошка": np.array([-0.9, 0.4]),
    "собака": np.array([-0.8, 0.6]),
    "робот": np.array([0.9, 0.1]),
    "вирус": np.array([-0.2, -0.9]),
    "милый": np.array([-0.3, 0.9]),
    "сломанный": np.array([0.6, -0.8]),
}

# Фразы
phrase1 = embeddings["милый"] + embeddings["кошка"]
phrase2 = embeddings["милый"] + embeddings["собака"]
phrase3 = embeddings["сломанный"] + embeddings["робот"]

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

print("Similarity (милая кошка, милая собака):",
      cosine_similarity(phrase1, phrase2))

print("Similarity (милая кошка, сломанный робот):",
      cosine_similarity(phrase1, phrase3))

# Визуализация
for word, vec in embeddings.items():
    plt.scatter(vec[0], vec[1])
    plt.text(vec[0]+0.02, vec[1]+0.02, word)

plt.axhline(0)
plt.axvline(0)
plt.title("Семантическое пространство")
plt.savefig("semantic_space.png", dpi=150, bbox_inches='tight')
print("График сохранен в файл: semantic_space.png")
plt.close()