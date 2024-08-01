fundasta_str = """
以下は'FundastA_Policy'ツールから抜粋された就業規則の内容です。
この資料の内容からユーザーの質問に関連する情報があるか確認してください。
・ユーザーの質問：{question}

・就業規則の内容：
本規則は、株式会社FundastA（以下、会社という）の従業員の労働条件、服務規律その他の就業に関す
る事項を定める。この規則は、株式会社FundastA(以下「当社」とする)の全ての従業員に適⽤します。ただし、期間を定め
て雇⽤される契約従業員・パートタイマー・アルバイト、定年後に期間を定めて雇⽤される嘱託従業員
の就業に関する必要な事項については、個別に結ぶ雇⽤契約書を適⽤して、本規則は適⽤しない。

"""

web_str = """
以下は'web_search'ツールを利用して、オンラインでユーザーの質問を検索した結果です。
この検索結果からユーザーの質問に関連する情報があるか確認してください。
・ユーザーの質問：{question}

・検索結果：

"""


def format_docs(docs):
    return "\n".join(
        [
            f"<document><content>{fundasta_str}\n{doc.page_content}</content><source>FundastA 就業規則 PDFファイル</source><page>{int(doc.metadata['page-number'])+1}</page></document>"
            for doc in docs
        ]
    )


def format_searched_docs(docs):
    return "\n".join(
        [
            f"<document><content>{web_str}\n{doc['content']}</content><source>{doc['url']}</source></document>"
            for doc in docs
        ]
    )
