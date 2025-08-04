# Balamurugan-1511-AI_AGENT_sample
This project demonstrates how to use the ```google/flan-t5-xl``` model from HuggingFace with the LangChain framework to perform text generation and create a question-answering chatbot.

##  Features
Utilizes the powerful FLAN-T5 XL model

Integrated with LangChain using HuggingFacePipeline

Accepts user input prompts and returns intelligent responses

Supports customizable prompt templates

##  Installation
Make sure you have Python 3.8+ and then install the required dependencies:

```Python
pip install langchain langchain-core transformers accelerate
```
## Usage
```Python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_community.llms import HuggingFacePipeline

```

# Load model and tokenizer
```Python
model_id = "google/flan-t5-xl"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id, device_map="auto")


pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
llm = HuggingFacePipeline(pipeline=pipe)

from langchain_core.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_messages([
("system", "You are a helpful assistant."),
("human", "{input}")
])
chain = prompt | llm
response = chain.invoke({"input": "Who is the founder of Google?"})
print("Answer:", response)

```

<img width="1121" height="469" alt="image" src="https://github.com/user-attachments/assets/6a026507-f8ff-4c95-b9e6-59a335cb102e" />

##  Output
<img width="893" height="295" alt="image" src="https://github.com/user-attachments/assets/1260c040-2e55-4f90-a245-c1285a2ad315" />


## 📚 Dependencies
Transformers

LangChain

Accelerate

## 🔮 Future Ideas
Extend with memory and multi-turn conversations

Add Streamlit or Gradio UI

Deploy as an API or chatbot










Ask ChatGPT
