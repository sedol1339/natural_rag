Loading the dataset:

```
from natural_rag.dataset import RAGDataset
dataset = RAGDataset.load_from_dir('datasets/bl')
print(dataset)
```

```
RAGDataset(752 documents, 109 questions, 97 questions with answers)
```

```
print(dataset.report_stats())
```

```
Total documents: 752
Shortest documents in symbols: [59, 59, 59, 59, 59]
Longest documents in symbols: [73440, 45060, 42362, 25090, 23764]
Symbols: 2104247
Words: 309256
Pages (assuming 1800 chars/page): 1169
Total questions: 109
Total questions with answers: 97
Min relevant docs per question: 0
Max relevant docs per question: 13
N docs without questions: 610
```