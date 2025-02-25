import torch
from transformers import (
    pipeline,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFacePipeline


def initialize_model(model_name=None, model_path=None):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    if model_name:        
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=bnb_config, device_map="auto", cache_dir=model_path
        )
    elif model_path:
        print(model_name,model_path)
        
        tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=model_path)

        model = AutoModelForCausalLM.from_pretrained(
            model_path, quantization_config=bnb_config, device_map="auto"
        )
    return tokenizer, model


def create_vector_store(train_data, embedding_model_name):
    embedding = HuggingFaceEmbeddings(model_name=embedding_model_name)
    documents = [
        f"Q: {q}\\nA: {a}" for q, a in zip(train_data["question"], train_data["answer"])
    ]
    return FAISS.from_texts(documents, embedding)


def create_qa_chain(vector_store, model, tokenizer, prompt_template):
    retriever = vector_store.as_retriever(search_model="cosine", search_kwargs={"k": 5})
    text_gen_pipeline = pipeline(
        model=model,
        tokenizer=tokenizer,
        task="text-generation",
        do_sample=True,
        temperature=0.1,
        return_full_text=False,
        max_new_tokens=64,
    )
    llm = HuggingFacePipeline(pipeline=text_gen_pipeline)
    prompt = PromptTemplate(
        input_variables=["context", "question"], template=prompt_template
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )
