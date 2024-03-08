from prompts import REACT
from langchain import PromptTemplate
from langchain.agents import create_react_agent
from langchain.agents import AgentExecutor
import re
from langchain_google_vertexai import VertexAI
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from typing import Optional, Type
import pandas as pd
from run_search import get_retriever, get_semantic_chain
import vertexai

PROJECT_ID = "lloyds-genai24lon-2701"
LOCATION = "us-central1"


class MyToolInput(BaseModel):
    inputs: str = Field(description="some inputs")


class DocumentLookup(BaseTool):
    name = "document_lookup"
    description = "this tool will fetch the document for a provided valid VRN identifier, you must only provide the VRN as input. the output will be JSON if a document was found"
    args_schema: Type[BaseModel] = MyToolInput

    def _run(
        self, vrn: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        data = pd.read_csv("fos_50_summary_records.csv")
        doc = data[data["id"] == vrn]
        print(doc)
        if len(doc) > 0:
            date = doc["date"].tolist()[0]
            summary = doc["text_summary"].tolist()[0]
            url = f"https://www.financial-ombudsman.org.uk/decision/{vrn}.pdf"
            company = doc["company"].tolist()[0]
            decision = doc["decision"].tolist()[0]
            area = doc["area"].tolist()[0]
            context = f"""
            Below is the date and summary of the document with the VRN {vrn} 

            Date: {date}\n

            Company: {company}

            Summary: {summary}

            URL: {url}

            Area: {area}

            Decision: {decision}
            """
            return context
        
        return "No documents were found."

    async def _arun(
        self, tool_input: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("my_tool does not support async")


class SemanticSearch(BaseTool):
    name = "semantic_search"
    description = "this tool will run semantic search over a vector database containing documents and retrieve the top 5 most relevant documents"
    args_schema: Type[BaseModel] = MyToolInput
    retriever = get_retriever()

    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        # for the query return the most relevant documents
        chain = get_semantic_chain(self.retriever)
        q_dict = {'query': query}
        answer = chain(q_dict)
        return answer

    async def _arun(
        self, tool_input: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("my_tool does not support async")


def create_agent():
    vertexai.init(project=PROJECT_ID, location=LOCATION)
    llm = VertexAI(model_name="gemini-pro")
    tools = [DocumentLookup(), SemanticSearch()]
    prompt_str = f"""
    You are an AI chatbot which can help with FOS (Financial Ombudsman Service)

    In all of your responses, you should include the key information (e.g. amounts paid) in a concise manner.
    {REACT}
    """
    agent_prompt = PromptTemplate(
        template=prompt_str,
        input_variables=re.findall("{(.*?)}", prompt_str)
    )
    agent = create_react_agent(tools=tools, llm=llm, prompt=agent_prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, return_intermediate_steps=True)
    print(agent_executor)
    return agent_executor
