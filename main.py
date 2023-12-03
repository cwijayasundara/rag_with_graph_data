from dotenv import load_dotenv

load_dotenv()

from langchain.document_loaders import PyPDFLoader

file_path = "docs/2023q2-alphabet-earnings-release.pdf"
book_loader = PyPDFLoader(file_path=file_path)
book_data = book_loader.load_and_split()

from langchain.indexes import GraphIndexCreator
from langchain.llms import OpenAI

llm = OpenAI(temperature=0)
index_creator = GraphIndexCreator(llm=llm)
graph = index_creator.from_text(book_data[10].page_content)

print(graph.get_triples())

from IPython.display import SVG
graph.draw_graphviz(path="book.svg")
SVG('book.svg')

import networkx as nx
from langchain.graphs.networkx_graph import NetworkxEntityGraph

graphs = [
    index_creator.from_text(doc.page_content)
    for doc in book_data
]

graph_nx = graphs[0]._graph
for g in graphs[1:]:
    graph_nx = nx.compose(graph_nx, g._graph)

graph = NetworkxEntityGraph(graph_nx)

graph.draw_graphviz(path="graph.pdf", prog='fdp')

from langchain.chains import GraphQAChain
from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(temperature=0)

chain = GraphQAChain.from_llm(
    llm=llm,
    graph=graph,
    verbose=True
)

question = """
Whats the Foreign currency exchange gain (loss) for Google?
"""

chain.run(question)

from langchain.graphs import Neo4jGraph

url = "bolt://localhost:7687"
username = "neo4j"
password = ""

graph_db = Neo4jGraph(
    url=url,
    username=username,
    password=password
)

graph_db.add_graph_documents(book_data)
graph_db.refresh_schema()

from langchain.chains import GraphCypherQAChain
from langchain.chat_models import ChatOpenAI

chain = GraphCypherQAChain.from_llm(
    cypher_llm=ChatOpenAI(temperature=0, model_name="gpt-4"),
    qa_llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo"),
    graph=graph_db,
    verbose=True,
)

question = """
Whats the Foreign currency exchange gain (loss) for Google?
"""

response = chain.run(question)
print(response)

