import random
import time
import torch


N_TENSORS = 10000
N_DIM = 100
N_ITERS = 20


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

    start = time.time()

    for _ in range(N_ITERS):
        scores = []
        for idx in range(N_TENSORS):
            score = tensors[idx] @ query
            scores.append(score)

        argmaxes = sorted(range(N_TENSORS), key=lambda idx: scores[idx], reverse=True)
        argmaxes = argmaxes[:3]
        best_for_loop = [documents[idx] for idx in argmaxes]

    end = time.time()
    avg_time = (end - start) / N_ITERS
    print(f"{avg_time=}")

    print("matrix approach:")

    M = torch.stack(tensors)

    start = time.time()

    for _ in range(N_ITERS):
        result = M @ (query.view(-1, 1))
        argmaxes = torch.topk(result, k=3, dim=0)
        best_matrix_docs = [
            documents[idx] for idx in argmaxes.indices.view(-1).tolist()
        ]

    end = time.time()

    avg_matrix_time = (end - start) / N_ITERS

    print(f"{avg_matrix_time=}")
