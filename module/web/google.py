import requests
from data.const import env_google, CSE_ID
from module.tools.utils import format_searched_google
from langchain.tools.base import BaseTool

google_api_key = env_google
google_cx = CSE_ID


class google_search(BaseTool):
    name: str = "google_search"
    description: str = (
        """Googleでユーザーの質問を検索することができます。オンラインで情報を検索する時に使います。
        web_searchで関連情報が見つからなかった場合には、このツールでオンライン情報を検索してみてください。
        以下の時にはこのツールを使ってください。
        １．Fundasta_policyツールを使ったがユーザーの質問に答えることが出来なかった。
        ２．ユーザーがリアルタイム情報について質問をした。
        ３．ユーザーがLLMがまだ学習をしていない最近の情報について質問をした。
        ４．他のツールでユーザーの質問に答えることが出来なかった時、最後にこのツールを使ってみてください
        """
    )

    def _run(self, input_str: str):
        search_results = []
        search_url = f"https://www.googleapis.com/customsearch/v1?q={input_str}&key={google_api_key}&cx={google_cx}"
        response = requests.get(search_url)
        results = response.json()
        search_results = format_searched_google(
            [results["items"][i] for i in range(6)], input_str
        )
        # print(search_results)
        return "\n\n" + search_results + "\n\n"

    def _arun(self, input_str: str):
        raise NotImplementedError("Async method not implemented")


if __name__ == "__main__":
    google_search("FundastA")
