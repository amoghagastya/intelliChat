from ctypes import alignment
import streamlit as st
import numpy as np
import cohere
import pandas as pd
from dotenv import load_dotenv 
import os
import pinecone
import streamlit.components.v1 as components

load_dotenv()

PINECONE = os.environ.get("PINECONE_KEY")
COHERE = os.environ.get("COHERE_KEY")

co = cohere.Client(COHERE)

st.set_page_config(
    page_title="Intent Discovery", page_icon="ℹ", layout = "wide")

st.sidebar.header("Information Retrieval")
st.markdown("# Information Retrieval - Semantic Search")

st.write("Using Existing Knowledge Bases, we can integrate Semantic Search with Conversational Agents with ease.")
st.write("Note: the agent below is not trained on ANY intents or training phrases and only uses the default fallback intent.")
components.html(
    """
    <div align=center>
    <script src="https://www.gstatic.com/dialogflow-console/fast/messenger/bootstrap.js?v=1"></script>
<df-messenger
  intent="WELCOME"
  chat-title="Cohere-embed"
  agent-id="fce6c5b1-a51a-46d3-9221-d1375561eaf3"
  language-code="en"
></df-messenger>
</div>
    """,height = 500, width = 650, scrolling=True)

# pinecone.init(PINECONE, environment='us-west1-gcp')

# index_name = 'cohere-pinecone-test'

# # if the index does not exist, we create it
# if index_name not in pinecone.list_indexes():
#     pinecone.create_index(
#         index_name,
#         dimension=4096,
#         metric='cosine'
#     )

# # connect to index
# index = pinecone.Index(index_name)

# batch_size = 10

# # st.write(df)
# embeds = co.embed(
#             texts=df['title'].tolist(),
#             model='large',
#             truncate='LEFT'
#         ).embeddings

# shape = np.array(embeds).shape
       
# ids = [ids for ids in df['id'].tolist()]

# meta = [{
#     'title': row['title'],
#     'url' : row['URL']
# } for i, row in df.iterrows()]

# # create list of (id, vector, metadata) tuples to be upserted
# to_upsert = list(zip(ids, embeds, meta))

# index.upsert(to_upsert)
# # let's view the index statistics

# # delete_response = index.delete(deleteAll = True)
# # index.describe_index_stats()
# print(index.describe_index_stats())

df = pd.read_csv('./RCkb - Sheet1.csv', sep=',')
df = df.astype({'id':'string'})

st.write("Our Vector Search Knowledge Database looks something like this -")
st.dataframe(df)
