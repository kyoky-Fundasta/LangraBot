import os
from data.const import GraphState, env_tavily
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_community.tools.tavily_search import TavilySearchResults
from module.tools.utils import format_searched_docs
from langchain.tools.base import BaseTool
from data.test_data.tavily_search_sample import tavily_result2

os.environ["TAVILY_API_KEY"] = env_tavily


class web_search(BaseTool):
    name: str = "web_search"
    description: str = (
        """ウェブの情報を検索してユーザーの質問に答えることが出来ます。
        google_searchで関連情報が見つからなかった場合には、このツールでオンライン情報を検索してみてください。
        以下の時にはこのツールを使ってください。
        １．Fundasta_policyツールを使ったがユーザーの質問に答えることが出来なかった。
        ２．ユーザーがリアルタイム情報について質問をした。
        ３．ユーザーがLLMがまだ学習をしていない最近の情報について質問をした。
        ４．他のツールでユーザーの質問に答えることが出来なかった時、最後にこのツールを使ってみてください
        """
    )

    def _run(self, input_str: str):

        search = TavilySearchAPIWrapper()
        search_tool = TavilySearchResults(max_results=6, api_wrapper=search)
        search_result = search_tool.invoke({"query": input_str})

        search_result = format_searched_docs(search_result, input_str)
        # print("\n\nClass :", type(search_result), search_result)
        return "\n\n" + search_result + "\n\n"

        ## Return a dummy search result. To save api calls.
        # return tavily_result2

    def _arun(self, input_str: str):
        raise NotImplementedError("Async method not implemented")


# # Web search API
# def search_on_web(state: GraphState) -> GraphState:
#     # Tavily web search
#     search = TavilySearchAPIWrapper()
#     search_tool = TavilySearchResults(max_results=6, api_wrapper=search)
#     search_result = search_tool.invoke({"query": state["question"]})

#     # # Test data for saving tavily search api fee
#     # search_result = tavily_result1

#     # print("##Tavily:", search_result)
#     # Reshape the search_result
#     search_result = format_searched_docs(search_result)
#     state["web"]=search_result
#     # Preserve it in the state.
#     return state
#     # return tavily_result2


if __name__ == "__main__":
    # Example GraphState
    state = GraphState(
        question="FundastAの資本金はいくらですか",
        answer="",
        context="",
        chat_history=[],
        web="",
        relevance=0.0,
    )

    # Using the function
    # new_state = search_on_web(state)
    # print("Function :", new_state)

    # Using the class
    tool = web_search()
    result = tool._run(state["question"])
    # print("\n\nClass :", type(result), result)
