from llama_index import SimpleDirectoryReader, GPTListIndex, readers, GPTVectorStoreIndex, LLMPredictor, PromptHelper, ServiceContext
from langchain import OpenAI
import sys
import os
import jsonpickle

def construct_index(directory_path):
    # set maximum input size
    max_input_size = 4096
    # set number of output tokens
    num_outputs = 2000
    # set maximum chunk overlap
    max_chunk_overlap = 20
    # set chunk size limit
    chunk_size_limit = 600 

    # define prompt helper
    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)

    # define LLM
    llm_predictor = LLMPredictor(llm=OpenAI(temperature=0.5, model_name="text-davinci-003", max_tokens=num_outputs))
 
    documents = SimpleDirectoryReader(directory_path).load_data()
    
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)
    index = GPTVectorStoreIndex.from_documents(documents, service_context=service_context)

    with open('index.json', 'w') as f:
        f.write(jsonpickle.encode(index))

    return index

def ask_ai():
    with open('index.json', 'r') as f:
        index = jsonpickle.decode(f.read())

    while True: 
        query = input("What do you want to ask? ")
        response = index.query(query, k=1)
        if response:
            print(f"Response: {response[0].text}")
        else:
            print("Sorry, I don't have an answer to that question.")
        print(f"Response: {response}")

if __name__ == "__main__":

    os.environ["OPENAI_API_KEY"] = input("Paste your OpenAI key here and hit enter:")
    construct_index("context_data/data")
    ask_ai()

