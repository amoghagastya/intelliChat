import streamlit as st
import time
import numpy as np
import cohere
import pandas as pd
import umap
# from babyplots import Babyplot
import altair as alt
from dotenv import load_dotenv 
import os
load_dotenv()

COHERE = os.environ.get("COHERE_KEY")

co = cohere.Client(COHERE)

st.set_page_config(
    page_title="Intent Discovery", page_icon="🔎", layout = "wide")

st.sidebar.header("Intent Discovery")
st.markdown("# Intent Discovery")

st.write("Leverage the data you already have, and uncover possible intents that can be created.")

# Get text embeddings
def get_embeddings(text,model='large'):
  output = co.embed(
                model=model,
                texts=[text])
  return output.embeddings[0]

def plot(df,embeds):
    # Plot
    reducer = umap.UMAP(n_neighbors=5) 
    umap_embeds = reducer.fit_transform(embeds)
    df['x'] = umap_embeds[:,0]
    df['y'] = umap_embeds[:,1]
    chart = alt.Chart(df).mark_circle(size=60).encode(
        x=#'x',
        alt.X('x',
            scale=alt.Scale(zero=False),
            axis=alt.Axis(labels=False, ticks=False, domain=False)
        ),
        y=
        alt.Y('y',
            scale=alt.Scale(zero=False),
            axis=alt.Axis(labels=False, ticks=False, domain=False)
        ),
        tooltip=['queries']
    ).configure(background="#FDF7F0"
    ).properties(
        width=700,
        height=400,
        title='Intent Clusters'
    )

    chart.interactive()
    return st.altair_chart(chart, use_container_width=True)

def generate_intents(text):
    response = co.generate(
    model='xlarge',
    prompt='Given the following utterances, predict the name of the intent, I still have not received my new card, I ordered over a week ago, I ordered a card but it has not arrived. Help please!\nName -> card_arrival\n--\nI would like to reactivate my card, Where do I link the new card?, I have received my card, can you help me put it in the app?, How do I link a card that I already have?\nName -> card_linking\n--\nWhy is the exchange rate wrong when I purchase something abroad?, I bought an item and noticed the exchange rate was not correct, I bought something overseas and the wrong exchange rate is on my statement.\nName -> card_payment_wrong_exchange_rate\n--\nI need information about an extra €1 fee in my statement, Why is there an extra 1 pound charge on my card?, Why are there so many fees on my statement?, What would be the reason there\'s an extra fee on my statement?\nName -> extra_charge_on_statement\n--\nWhen are you open?,When do you close?,What are the hours?,Are you open on weekends?\nAre you available on holidays?\nName -> opening_hours\n--\n' + text + '\nName -> ',
    max_tokens=20,
    temperature=0.8,
    k=0,
    p=1,
    frequency_penalty=0,
    presence_penalty=0,
    stop_sequences=["--"],
    return_likelihoods='NONE')
    print('Prediction: {}'.format(response.generations[0].text))
    return response.generations[0].text



col1, col2 = st.columns([1,1.7], gap="medium")

with col1:
    st.markdown('\n\n\n')
    query = st.text_area("Enter your unstructured data on each line below ⤵️", """What are the hours?
Are you open on weekends?
Are you available on holidays?
How much is a burger?
What's the price of a meal?
How much for a few burgers?
Do you have a vegan option?
Do you have vegetarian?
Do you serve non-meat alternatives?
My statement has a dollar I have been charged showing up on it.
Why is there a fee for an extra pound in my statement?
I'm not okay with this fee on my statement.
Do you support all fiat currencies?
Does your system support multiple currency.
What currencies can I hold money in?
The card I have doesn't work.
My card doesn't accept any transaction at all. What's wrong??
How can i check if my card is working?
Nothing goes through on my card.
will i be able to open an account for my daughter
How old do I need to be to open an account
What age can sign up for services?""", height=500)

    submit = st.button('Submit')
    if 'num' not in st.session_state:
        st.session_state['num'] =  0
    if submit:
        with st.spinner('Wait for it...'):
                # embeds = 
                queries = query.split('\n')
                df = pd.DataFrame(queries)
                df.columns = ['queries']
                # df['query_embeds'] = df['queries'].apply(get_embeddings)
                embeds = co.embed(
                    texts=df['queries'].tolist(),
                    model='large',
                    truncate='LEFT'
                ).embeddings
                with col2:
                    plot(df,embeds)
                
utts = st.text_input("Paste the clusters here to Generate Intent Name")
ints = st.button('Generate')
if ints:
    text = ','.join(utts.split(' '))
    intent_name = generate_intents(text)
    st.write("Intent Name : ", intent_name)


