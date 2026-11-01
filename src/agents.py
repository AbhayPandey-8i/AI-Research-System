from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tools import web_search, scrape_url
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
load_dotenv()


#llm model setup
llm = ChatGroq(model="openai/gpt-oss-120b",temperature=0,  api_key=os.getenv("GROK_API_KEY"))


writer_llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=os.getenv("GROK_API_KEY"),
    max_tokens=3000
)

critic_llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=os.getenv("GROK_API_KEY"),
    max_tokens=800
)

#1st agent
def build_search_agent():
    return create_agent(
        model=llm,
        tools=[web_search]

    )


#2nd agent
def build_reader_agent():
    return create_agent(
        model=llm,
        tools=[scrape_url]
    )

#writer chain (we will use runnable here)
writer_prompt = ChatPromptTemplate.from_messages([
    ("system", "you are an expert research writer. write clear, structured and insightful reports."),
    ("human", """write a detailed research report on the topic below.
    
    Topic: {topic}
    
    Research Gathered:
    {research}
    
    Structure the report as:
    - Introduction
    - Key findings (minimum 3 well explained points)
    - Conclusion
    - Source (list all URLs found in the research)
    
    Be detailed, factual and professional.""")
])

writer_chain = writer_prompt | writer_llm | StrOutputParser()

#critic chain (gives feedback based on system and suggestions to improve)
critic_prompt = ChatPromptTemplate.from_messages([
    ("system", "you are a sharp and constructive research critic. Be honest and specific."),
    ("human", """Review the research report  below and evaluate it strictly.
    
    Report:
    {report}
    
    Respond in exact format:
    
    Score: X/10
    
    Strengths:
    - ...
    - ...
    
    Areas to Improve:
    - ...
    - ...
    
    One line verdict
    ...""")
])


critic_chain = critic_prompt | critic_llm | StrOutputParser()
