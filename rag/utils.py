def format_docs(docs):
    return "\n".join(
        [
            f"<document><content>{doc.page_content}</content><source>FundastA 就業規則 PDFファイル</source><page>{int(doc.metadata['page-number'])+1}</page></document>"
            for doc in docs
        ]
    )


def format_searched_docs(docs):
    return "\n".join(
        [
            f"<document><content>{doc['content']}</content><source>{doc['url']}</source></document>"
            for doc in docs
        ]
    )
