from llama_index import SimpleDirectoryReader, GPTListIndex, readers, GPTSimpleVectorIndex, LLMPredictor, PromptHelper, ServiceContext
from langchain.chat_models import ChatOpenAI
from langchain import OpenAI, ConversationChain
from langchain.memory import ConversationBufferMemory
import uuid
import sys
import os

def construct_index(directory_path):
    # set maximum input size
    max_input_size = 1024
    # set number of output tokens
    num_outputs = 1000
    # set maximum chunk overlap
    max_chunk_overlap = 10
    # set chunk size limit
    chunk_size_limit = 512 

    # define prompt helper
    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)

    # define LLM
    llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0.7, model_name="text-davinci-003", max_tokens=num_outputs))
 
    documents = SimpleDirectoryReader(directory_path).load_data()
    
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)

    index = GPTSimpleVectorIndex.from_documents(documents, service_context=service_context)

    index.save_to_disk('index.json')

    return index

def ask_ai():
    index = GPTSimpleVectorIndex.load_from_disk('index.json')
    llm = OpenAI(temperature=0.7, model_name="text-davinci-003")
    memory = ConversationBufferMemory(return_messages=True)
    conversation_id = str(uuid.uuid4())
    conversation = ConversationChain(llm=llm, memory=memory)

    while True:
        query = input("Ask anything about 1 Finance : ")
        if not query:
            print("Please enter something to get the response")
        else:
            memory.chat_memory.add_user_message(query)
            query_config = {
                'history': [memory.chat_memory]
            }
            response = index.query(query).response
            memory.chat_memory.add_ai_message(response)
        print(f"Response: {response}")
        print(memory.chat_memory)
        print("\n")

if __name__ == "__main__":

    os.environ["OPENAI_API_KEY"] = input("OpenAI key here and hit enter : ")
    if not os.path.exists('index.json'):
        construct_index("context_data/data")
    ask_ai()

