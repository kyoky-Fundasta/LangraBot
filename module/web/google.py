import requests
from data.const import env_google, CSE_ID
from module.tools.utils import format_searched_google

google_api_key = env_google
google_cx = CSE_ID


def google_search(query):
    search_results = []
    search_url = f"https://www.googleapis.com/customsearch/v1?q={query}&key={google_api_key}&cx={google_cx}"
    response = requests.get(search_url)
    results = response.json()
    search_results = format_searched_google(
        [results["items"][i] for i in range(10)], query
    )
    print(search_results)


if __name__ == "__main__":
    google_search("FundastA")
