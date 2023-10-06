from src.data_handler import *
from src.model_handler import *
from src.utils import *
data_path = './data/'
data_handler = DataHandler(data_path)
data_handler.load_data()
data_handler.chunk_data()
db = data_handler.create_vector_db()

pipe = Pipeline(vector_db=db)

query = "Em que ano foi fundada a Terra dos Robôs?"
llm_response = pipe.search(query)
process_llm_response(llm_response)