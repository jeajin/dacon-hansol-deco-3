import os
import yaml
import argparse
import pandas as pd
from tqdm import tqdm
from datasets import Dataset
from transformers import pipeline
from sentence_transformers import SentenceTransformer

from utils.data_utils import load_data, preprocess_data, create_qa_data
from utils.model_utils import initialize_model, create_vector_store, create_qa_chain

import torch._dynamo
torch._dynamo.config.suppress_errors = True
from torch.utils.data import DataLoader

class QADataSet(Dataset):
    def __init__(self, test_data):
        self.test_data = test_data
    
    def __len__(self):
        return len(self.test_data)
    
    def __getitem__(self, idx):
        # 단순히 질문을 가져오는 것만 수행합니다
        question = self.test_data[idx].itertuples(index=False).question
        return {"query": question}


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg-path", type=str, default=None)
    return parser.parse_args()


def load_config(cfg_path):
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def main():
    """
    변경사항 1. 변수명 수정
        - train -> train_df
        - test -> test_df
        - combined_train_data -> train_data
        - combined_test_data -> test_data

    변경사항 2. Inference 기본 코드 수정
        - 일정 idx마다 출력되는 것 대신 tqdm 사용
        - test_data를 iterrows()로 순회하는 대신 itertuples()로 순회
        - for문을 사용하여 test_data를 순회하며 결과를 저장하는 대신 list comprehension 사용
    """
    # load config
    cfg = load_config(arg_parser().cfg_path)
    train_data_path = cfg["paths"]["train_data"]
    test_data_path = cfg["paths"]["test_data"]
    submission_path = cfg["paths"]["submission"]
    output_path = cfg["paths"]["output"]
    model_name = cfg["model"]["model_name"]
    model_path = cfg["model"]["model_path"]
    embedding_model_name = cfg["model"]["embedding_model"]
    prompt_template = cfg["prompt_template"]
    batch_size = cfg["settings"]["batch_size"]

    # load data
    train_df, test_df = load_data(train_data_path, test_data_path)
    train_df = preprocess_data(train_df)
    test_df = preprocess_data(test_df)
    train_data = create_qa_data(train_df, is_train=True)
    test_data = create_qa_data(test_df, is_train=False)

    # Import model
    tokenizer, model = initialize_model(model_name, model_path)

    # Create vector store
    vector_store = create_vector_store(train_data, embedding_model_name)

    # Generate RAG chain
    qa_chain = create_qa_chain(vector_store, model, tokenizer, prompt_template, batch_size)

    # Batch processing
    test_dataset = Dataset.from_pandas(test_data)
    # DataLoader 생성
    # qa_dataset = QADataSet(test_dataset)
    # qa_dataloader = DataLoader(qa_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Inference
    print("테스트 실행 시작... 총 테스트 샘플 수:", len(test_data))
    test_results = [
        qa_chain.invoke(row.question)["result"]
        for row in tqdm(
            test_data.itertuples(index=False), total=len(test_data), desc="Processing"
        )
    ]

   
    # test_results = []
    # 이제 DataLoader를 사용해 배치 단위로 데이터를 처리합니다
    # for batch in tqdm(qa_dataloader, desc="Processing"):
    #     print()
    #     questions = batch
    #     batch_results = [res["result"] for res in qa_chain.batch(questions)]
    #     test_results += batch_results

    # for i in tqdm(range(0, len(test_data), batch_size), desc="Processing"):
    #     batch = test_data.iloc[i : i + batch_size]
    #     questions = [{"query": row.question} for row in batch.itertuples(index=False)]
    #     #print(questions)
    #     batch_results = [res["result"] for res in qa_chain.batch(questions)]
    #     test_results+=batch_results



    #origin
    # test_results = [
    #     qa_chain.invoke(row.question)["result"] for row in tqdm(test_data.itertuples(index=False), total=len(test_data), desc="Processing")
    # ]



    # Submission
    embedding = SentenceTransformer(embedding_model_name)
    pred_embeddings = embedding.encode(test_results)

    submission = pd.read_csv(submission_path, encoding="utf-8-sig")
    submission.iloc[:, 1] = test_results
    submission.iloc[:, 2:] = pred_embeddings
    submission.to_csv(output_path, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
