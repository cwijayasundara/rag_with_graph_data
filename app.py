from dotenv import load_dotenv

load_dotenv()

from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.document_loaders import PyPDFLoader
from langchain.indexes import GraphIndexCreator
from langchain.chat_models import ChatOpenAI
import networkx as nx
from langchain.graphs.networkx_graph import NetworkxEntityGraph
from langchain.chains import GraphQAChain

file_path = "docs/2023q2-alphabet-earnings-release.pdf"
sec_loader = PyPDFLoader(file_path=file_path)
sec_data = sec_loader.load_and_split()

llm = ChatOpenAI(
    model_name='gpt-3.5-turbo',
    streaming=False,
    callbacks=[StreamingStdOutCallbackHandler()],
    temperature=0)

index_creator = GraphIndexCreator(llm=llm)

graphs = [
    index_creator.from_text(doc.page_content)
    for doc in sec_data
]

graph_nx = graphs[0]._graph
for g in graphs[1:]:
    graph_nx = nx.compose(graph_nx, g._graph)

graph = NetworkxEntityGraph(graph_nx)

# graph.draw_graphviz(path="sec.pdf", prog='fdp')

chain = GraphQAChain.from_llm(
    llm=llm,
    graph=graph,
    verbose=True
)

question = """
What the revenues of 2023?
"""

response = chain.run(question)
print(response)

# from langchain_experimental.graph_transformers.diffbot import (
#     DiffbotGraphTransformer
# )
#
# diffbot_nlp = DiffbotGraphTransformer(
#     diffbot_api_key=diffbot_api_key
# )
#
# sec_graph = diffbot_nlp.convert_to_graph_documents(
#     sec_data
# )

from langchain.graphs import Neo4jGraph

url = "bolt://localhost:7687"
username = "neo4j"
password = ""

graph_db = Neo4jGraph(
    url=url,
    username=username,
    password=password
)

graph_db.add_graph_documents(sec_data)
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
What the revenues of 2023?
"""

response = chain.run(question)
print(response)
