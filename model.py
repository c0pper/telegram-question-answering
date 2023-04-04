# !pip install -qU pinecone-client sentence-transformers torch
import torch
from sentence_transformers import SentenceTransformer
import pinecone
from transformers import BartTokenizer, BartForConditionalGeneration
from pprint import pprint
import os

RETRIEVER_MODEL_NAME = "efederici/mmarco-sentence-BERTino"
TOKENIZER_MODEL_NAME = "efederici/bart-lfqa-it"
GENERATOR_MODEL_NAME = "efederici/bart-lfqa-it"
pinecone_index_name = "abstractive-question-answering-telegram-chat"

device = "cuda" if torch.cuda.is_available() else "cpu"

retriever = SentenceTransformer(
    RETRIEVER_MODEL_NAME,
    device=device
)

pinecone.init(
    api_key=os.environ.get("pinecone_key"),
    environment="us-east4-gcp"
)

index = pinecone.Index(pinecone_index_name)

tokenizer = BartTokenizer.from_pretrained(TOKENIZER_MODEL_NAME)
generator = BartForConditionalGeneration.from_pretrained(GENERATOR_MODEL_NAME)


def query_pinecone(query, top_k):
    xquery = retriever.encode([query]).tolist()
    xcontext = index.query(xquery, top_k=top_k, include_metadata=True)
    return xcontext


def format_query(query, context):
    context = [f"<p> {m['metadata']['text']}" for m in context]
    context = " ".join(context)
    query = f"Q: {query}\n\nC: {context}"
    return query


def generate_answer(query, gen_minlength=10, gen_maxlength=40):
    inputs = tokenizer([query], max_length=1024, return_tensors="pt")
    ids = generator.generate(inputs["input_ids"], num_beams=2, min_length=gen_minlength, max_length=gen_maxlength)
    answer = tokenizer.batch_decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0].replace("<p>",
                                                                                                                 "").replace(
        "\n", " ")
    return answer


def ask(query, top_k=5, gen_minlength=10, gen_maxlength=40):
    context = query_pinecone(query, top_k=top_k)
    query = format_query(query, context["matches"])
    answer = generate_answer(query, gen_minlength, gen_maxlength)

    print(f'{answer}\n\n\nContext:\n\n')
    context_list = []
    for doc in context["matches"]:
        print(doc["metadata"]["date"], doc["metadata"]["text"], end="\n---\n")
        context_list.append(f'{doc["metadata"]["date"]}: {doc["metadata"]["text"]}\n\n')
    context_str = "".join(context_list)

    return f'{answer}\n\nContext:\n{context_str}'



if __name__ == '__main__':
    ask("Chi è giuppa?", top_k=5, gen_minlength=5, gen_maxlength=50)
