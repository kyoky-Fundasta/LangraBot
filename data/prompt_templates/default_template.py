from langchain_core.prompts import PromptTemplate


prompt_template = PromptTemplate(
    input_variables=["agent_scratchpad", "input", "tool_names", "tools"],
    metadata={
        "lc_hub_owner": "hwchase17",
        "lc_hub_repo": "react",
        "lc_hub_commit_hash": "d15fe3c426f1c4b3f37c9198853e4a86e20c425ca7f4752ec0c9b0e97ca7ea4d",
    },
    template="Answer the following questions as best you can. You have access to the following tools:\n\n{tools}\n\nUse the following format:\n\nQuestion: the input question you must answer\nThought: you should always think about what to do\nAction: the action to take, should be one of [{tool_names}]\nAction Input: the input to the action\nObservation: the result of the action\n... (this Thought/Action/Action Input/Observation can repeat N times)\nThought: I now know the final answer\nFinal Answer: the final answer to the original input question\n\nBegin!\n\nQuestion: {input}\nThought:{agent_scratchpad}",
)
