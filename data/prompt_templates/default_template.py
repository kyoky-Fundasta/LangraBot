from langchain_core.prompts import PromptTemplate


prompt_template = PromptTemplate(
    input_variables=["agent_scratchpad", "tool_names", "tools"],
    metadata={
        "lc_hub_owner": "hwchase17",
        "lc_hub_repo": "react",
        "lc_hub_commit_hash": "d15fe3c426f1c4b3f37c9198853e4a86e20c425ca7f4752ec0c9b0e97ca7ea4d",
    },
    template="""
    You are an intelligent AI agent.

    Answer the following questions as best you can. 
    You have access to the following tools:\n\n{tools}\n\n
    Use the following format:\n\n
    Chat history: record of previous interactions between the user and the AI agent.\n
    Question: the input question you must answer\n
    Thought: you should always think about what to do\n
    Action: the action to take, should be one of [{tool_names}]\n
    Action Input: the input to the action\n
    Observation: the result of the action\n
    ... (this Thought/Action/Action Input/Observation can repeat 3 times)\n
    Thought: I now know the final answer\nFinal Answer: the final answer to the original input question in Japanese\n\n

    Important : If the question is easy and can be answered without any tools, you should instantly generate a final answer and close the chain without using any tools.

    Begin!\n\n
    Chat history: {history}\n
    Question: {question}\n
    Thought:{agent_scratchpad}""",
)
