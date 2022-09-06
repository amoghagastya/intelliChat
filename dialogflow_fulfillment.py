from flask import Flask, request
import cohere
import os
import pinecone

app = Flask(__name__)

PINECONE = os.environ.get("PINECONE_KEY")
COHERE = os.environ.get("COHERE_KEY")

co = cohere.Client(COHERE)

# connect to index
# index = pinecone.Index(index_name)


@app.route('/')  # this is the home page route
def hello_world(
):  # this is the home page function that generates the page code
    return "Hello world!"


@app.route('/webhook', methods=['POST'])
def webhook():
    req = request.get_json(silent=True, force=True)
    fulfillmentText = ''
    query_result = req.get('queryResult')
    if query_result.get('action') == 'input.unknown':
        index_name = 'cohere-pinecone-test'
        index = pinecone.Index(index_name)
        query = query_result.get('queryText')
        pinecone.init(PINECONE, environment='us-west1-gcp')
        index_name = 'cohere-pinecone-test'
        xq = co.embed(texts=[query], model='large', truncate='LEFT').embeddings


# query, returning the top 10 most similar results
    res = index.query(xq, top_k=3, include_metadata=True)
    # print('res is', res.matches)
    for i, r in enumerate(res.matches):
        title = r.get('metadata').get('title')
        url = r.get('metadata').get('url')
        # print(title, url)
        if i == 0:
            r1 = title + "\n" + url
        if i == 1:
            r2 = title + "\n" + url
        if i == 2:
            r3 = title + "\n" + url

    return {
        "fulfillmentText":
        "Check out these articles I found - \n\n" + r1 + "\n\n" + r2 + "\n\n" + r3 + '\n',
        "source":
        "webhookdata"
    }

if __name__ == '__main__':
    app.run(host='0.0.0.0',
            port=8080)  # This line is required to run Flask on repl.it
