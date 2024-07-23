import os
from data.const import GraphState, env_tavily
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_community.tools.tavily_search import TavilySearchResults
from module.tools.utils import format_searched_docs

os.environ["TAVILY_API_KEY"] = env_tavily


# Web search API
def search_on_web(state: GraphState) -> GraphState:
    # Tavily web search
    search = TavilySearchAPIWrapper()
    search_tool = TavilySearchResults(max_results=4, api_wrapper=search)
    search_result = search_tool.invoke({"query": user_question})

    # # Test data for saving tavily search api
    # search_result = tavily_result1

    # print("##Tavily:", search_result)
    # Reshape the search_result
    search_result = format_searched_docs(search_result)

    # Preserve it in the state.
    return search_result
    return GraphState(
        question=state["question"],
        context=state["context"],
        web=search_result,
        chat_history=state["chat_history"],
        answer=state["answer"],
        relevance=state["relevance"],
    )
