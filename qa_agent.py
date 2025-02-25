from langchain.schema import AgentAction, AgentFinish
from langchain.prompts import StringPromptTemplate
from langchain.agents import AgentOutputParser
import re
from langchain.chains import RetrievalQA
from langchain.agents import Tool
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from vector_store import create_chroma_vector_store, create_pinecone_vector_store, load_and_process_pdf
import config

# Initialize LLM
llm = ChatOpenAI(model_name="gpt-4o", temperature=0.2, openai_api_key=config.OPENAI_API_KEY)

# Define custom prompt template
class CustomPromptTemplate(StringPromptTemplate):
    template: str
    tools: list

    def format(self, **kwargs) -> str:
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        kwargs["agent_scratchpad"] = thoughts
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        return self.template.format(**kwargs)


# Define custom output parser
class CustomOutputParser(AgentOutputParser):
    def parse(self, llm_output: str):
        """Parses LLM output into AgentAction or AgentFinish"""
        if "Final Answer:" in llm_output:
            return AgentFinish(
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )

        match = re.match(r"Action: (.*?)[\n]*Action Input:[\s]*(.*)", llm_output, re.DOTALL)
        if not match:
            raise ValueError(
                f"Could not parse LLM output: `{llm_output}`"
            )

        action = match.group(1).strip()
        action_input = match.group(2)
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)


# Create retrieval tool for the agent
def create_tools(vector_store):
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})  # Fetch more relevant chunks
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

    tools = [
        Tool(
            name="Syllabus Question Answering Tool",
            func=qa_chain.run,
            description="Useful for answering questions about the syllabus."
        ),
    ]
    return tools


# Create Zero-Shot ReAct Agent
def create_zero_shot_agent(tools):
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        output_parser=CustomOutputParser(),  
        handle_parsing_errors=True,  
    )
    return agent
