# Requirements

```
pip install neomodel pydantic PyYAML
```

# Loading the dataset

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

# Installing Neo4j

```
# 1. Installation and starting
wget -O - https://debian.neo4j.com/neotechnology.gpg.key | sudo apt-key add -
echo 'deb https://debian.neo4j.com stable latest' | sudo tee /etc/apt/sources.list.d/neo4j.list
sudo apt-get update
sudo apt-get install neo4j
sudo systemctl enable neo4j
sudo systemctl start neo4j
sudo systemctl status neo4j

# 1. Setting password
cypher-shell
# enter default user: neo4j, pass: neo4j, set pass to: rag12345

# 2. Installing APOC
# version table: https://github.com/neo4j/apoc/releases
cd /var/lib/neo4j/plugins
sudo wget https://github.com/neo4j/apoc/releases/download/5.15.0/apoc-5.15.0-core.jar
sudo chown neo4j:neo4j /var/lib/neo4j/plugins/*.jar
sudo chmod 755 /var/lib/neo4j/plugins/*.jar

# In /etc/neo4j/neo4j.conf add lines:
dbms.security.procedures.unrestricted=apoc.*
dbms.security.procedures.allowlist=apoc.*
server.default_listen_address=0.0.0.0

# Save changes in the file and restart neo4j:
sudo systemctl restart neo4j

# Check if Neo4j shell with APOC is working:
cypher-shell -u neo4j -p ragu1234
# Issue the command: RETURN apoc.version();
# Ctrl-C

# Set env vars:
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=rag12345
```

# Save the dataset to Neo4j

```
import os
import sys

from neomodel import config

sys.path.append('/home/oleg/rag_workspace/natural_rag')
from natural_rag.data_neo4j import ingest_rag_dataset_to_neo4j
from natural_rag.dataset import RAGDataset

dataset = RAGDataset.load_from_dir('natural_rag/datasets/bl', load_sources=True)
print(dataset)

NEO4J_URI = os.environ['NEO4J_URI']
NEO4J_USERNAME = os.environ['NEO4J_USERNAME']
NEO4J_PASSWORD = os.environ['NEO4J_PASSWORD']
config.DATABASE_URL = NEO4J_URI.replace('bolt://', f'bolt://{NEO4J_USERNAME}:{NEO4J_PASSWORD}@')
print(config.DATABASE_URL)

ingest_rag_dataset_to_neo4j(dataset)
```

# Visualize

Open http://<ip>:7474/browser/

Then in the right bar select "neo4j" in "use database".

Then issue:

```
MATCH (n)
WHERE n:NeoQuestion OR n:NeoDocument
OPTIONAL MATCH (n)-[r:REFERENCES]-(target)
RETURN n, r, target
```