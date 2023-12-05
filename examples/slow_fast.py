import random
import time
import torch


N_TENSORS = 10
N_DIM = 10


if __name__ == "__main__":
    now = time.time()
    print(f"Starting script, {now=}")

    with open("/usr/share/dict/words", "r") as fd:
        documents = fd.readlines()

    print(f"{len(documents)=}")
    max_documents = len(documents)
    documents = [
        documents[random.randint(0, max_documents)].strip() for _ in range(N_TENSORS)
    ]

    tensors = [torch.randn(N_DIM) for _ in range(N_TENSORS)]

    query = torch.randn(N_DIM)

    print("for-loop approach:")

    scores = []
    for idx in range(N_TENSORS):
        score = tensors[idx] @ query
        scores.append(score)

    argmaxes = sorted(range(N_TENSORS), key=lambda idx: scores[idx], reverse=True)[:3]
    best_for_loop = [documents[idx] for idx in argmaxes]

    print(f"{best_for_loop=}")
