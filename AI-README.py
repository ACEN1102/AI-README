import json
import re
import time

import chromadb
import markdown
import requests
from github import Github
from github.GithubException import RateLimitExceededException

MAX_RETRIES = 10
RETRY_DELAY = 5

# Chroma HTTP 客户端
client = chromadb.HttpClient(host='localhost', port=9690)

# 服务地址
embedding_service_url = ""
rerank_service_url = ""
language_model_url = ""

# GitHub PAT
github_pat = Github("")


# GitHub API 搜索指定的公开代码仓库
def search_github_repositories(query):
    try:
        repositories = github_pat.search_repositories(query=query)
        return repositories
    except RateLimitExceededException as e:
        print(f"Rate limit exceeded. Please wait a while and try again later. Exception: {e}")
        return None
    except Exception as e:
        print(f"An error occurred during GitHub repository search: {e}")
        return None


# 获取仓库主页的介绍文本（README.md内容）
def get_readme_content(repo):
    try:
        # 获取README文件信息
        readme = repo.get_readme()
        # 构造 README.md 的下载链接
        name = repo.full_name
        html_url = readme.html_url
        prefix = f"https://github.com/{name}/blob/"  # 前缀
        suffix = html_url.replace(prefix, "")  # 后缀
        download_url = f"https://raw.githubusercontent.com/{name}/{suffix}"
        print('README文档地址:' + download_url)
        # 使用 requests 获取 README.md 内容
        response = requests.get(download_url)
        response.raise_for_status()  # 如果请求不成功，则抛出异常
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"Failed to fetch README: {e}")
        return ""
    except Exception as e:
        print(f"An error occurred while processing repository {repo.full_name}: {e}")
        return ""


# 获取文本的嵌入向量
def get_sentence_embedding(sentence):
    payload = json.dumps({
        "inputs": [
            sentence,
        ]
    })
    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json'
    }
    try:
        response = requests.post(embedding_service_url, headers=headers, data=payload, timeout=10)
        response.raise_for_status()
        response_data = response.json()
        if response_data:
            embedding_vector = response_data[0]
            return embedding_vector
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
    except requests.exceptions.ConnectionError as conn_err:
        print(f"Connection error occurred: {conn_err}")
    except requests.exceptions.Timeout as timeout_err:
        print(f"Timeout error occurred: {timeout_err}")
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


# 使用 rerank 服务进行重新排序
def rerank_query(query, entities):
    original_payload = {
        "pairs": [query, entities]
    }
    payload = json.dumps(original_payload)
    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json'
    }
    try:
        print("正在发送rerank服务请求...")
        response = requests.post(rerank_service_url, headers=headers, data=payload)
        print(f"Response status code: {response.status_code}")
        response.raise_for_status()
        rerank_result = response.json().get('data')  # 假设这里得到的是一个最接近的文本段的标识符或索引
        return rerank_result  # 返回最接近的文本段标识符或索引

    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP错误发生: {http_err}")
    except requests.exceptions.ConnectionError as conn_err:
        print(f"连接错误发生: {conn_err}")
    except requests.exceptions.Timeout as timeout_err:
        print(f"请求超时发生: {timeout_err}")
    except Exception as e:
        print(f"发生错误: {e}")
    return None


# 使用自然语言模型进行提问
def ask_question(payload, headers):
    try:
        attempt = 1
        while attempt <= MAX_RETRIES:
            try:
                print(f"正在向自然语言模型提问... (尝试次数: {attempt}/{MAX_RETRIES})")
                response = requests.post(language_model_url, headers=headers, data=payload)
                if response.status_code == 200:
                    return response.json()
                else:
                    print(f"Language model service request failed with status code {response.status_code}")
            except requests.exceptions.HTTPError as http_err:
                print(f"HTTP错误发生: {http_err}")
            except requests.exceptions.ConnectionError as conn_err:
                print(f"连接错误发生: {conn_err}")
            except requests.exceptions.Timeout as timeout_err:
                print(f"请求超时发生: {timeout_err}")
            except Exception as e:
                print(f"发生错误: {e}")

            if attempt < MAX_RETRIES:
                print(f"等待 {RETRY_DELAY} 秒后重试...")
                time.sleep(RETRY_DELAY)
            attempt += 1

        print(f"达到最大尝试次数 ({MAX_RETRIES})，无法完成请求.")
        return None

    except Exception as e:
        print(f"Error asking question to language model: {e}")
        return None


# 处理 README 内容并进行嵌入向量的处理和上传
def process_readme_content(repo, paragraphs):
    try:
        # 创建或获取用于嵌入的集合
        collection_name = f"{repo.full_name.replace('/', '_')}_embeddings"
        collection = client.get_or_create_collection(name=collection_name)

        # 处理每个段落的嵌入向量
        embeddings = []
        for i, paragraph in enumerate(paragraphs):
            embedding = get_sentence_embedding(paragraph)
            if embedding:
                embeddings.append(embedding)
                # 将嵌入向量存入集合
                document_id = f"{i}"
                collection.add(embeddings=[embedding], ids=[document_id], documents=[paragraph])

        return embeddings, collection

    except Exception as e:
        print(f"处理 {repo.full_name} 的README内容时出错：{e}")
        return None


def split_into_paragraphs(readme_content):
    sections = re.split(r'\n{2,}|\n(?=[A-Za-z0-9\s]+:)', readme_content)

    # 去除空格和空白行
    sections = [section.strip() for section in sections if section.strip()]

    return sections


def main():
    try:
        query = input("请输入你的问题: ").strip()
        search_query = input("请输入要搜索的仓库: ").strip()
        # query = "请介绍一下CHATGLM3"
        # search_query = "CHATGLM3"
        print("正在搜索符合条件的仓库...")
        repositories = search_github_repositories(search_query)
        if repositories:
            for repo in repositories[:1]:
                print(f"正在获取仓库 {repo.full_name} 的README文件...")
                readme_content = re.sub('<[^>]+>', '', markdown.markdown(get_readme_content(repo)))
                paragraphs = split_into_paragraphs(readme_content)
                if paragraphs:
                    print(f"正在调用embeddings服务处理仓库 {repo.full_name} 的README内容...")
                    embeddings, collection = process_readme_content(repo, paragraphs)
                    if embeddings:
                        print(f"找到 {len(embeddings)} 条嵌入向量.")
                        # entities = [json.dumps({'entity': emb}) for emb in embeddings]
                        # print("正在使用rerank服务计算余弦距离")
                        # rerank_result = rerank_query(query, entities)
                        # min_index = rerank_result.index(min(rerank_result))
                        if query is not None:
                            query_vectors = get_sentence_embedding(query)
                            # 在Chroma数据库中查询最接近文本段的信息
                            result = collection.query(
                                query_embeddings=query_vectors,
                                n_results=5,
                                include=["documents", "distances"]
                            )
                            first_document = result['documents'][0][0]
                            second_document = result['documents'][0][1]
                            third_document = result['documents'][0][2]
                            closest_text_segment = (first_document + second_document + third_document).strip()
                            if closest_text_segment:
                                # 构建向自然语言模型提问的payload
                                payload = json.dumps({
                                    "model": "千问Qwen1.5-14B-Chat",
                                    "messages": [
                                        {
                                            "role": "system",
                                            "content": "使用中文，你可以参考以下文本来回答用户的提问:" + closest_text_segment
                                        },
                                        {
                                            "role": "user",
                                            "content": query  # 用户的查询作为用户消息传递
                                        }
                                    ]
                                })
                                headers = {
                                    'Content-Type': 'application/json'
                                }
                                print("正在向自然语言模型提问...")
                                answer = ask_question(payload, headers)
                                if answer:
                                    print("答案如下:")
                                    print(answer['choices'][0]['message']['content'])
                                else:
                                    print("无法获取答案.")
                            else:
                                print("未找到最接近的文本段.")
                        else:
                            print("未能找到最接近的文本段标识符.")
                    else:
                        print(f"未找到仓库 {repo.full_name} 的嵌入向量.")
                else:
                    print(f"无法获取仓库 {repo.full_name} 的README内容.")
        else:
            print("仓库搜索失败，请检查你的GitHub令牌或稍后重试.")

    except Exception as e:
        print(f"发生错误: {e}")


if __name__ == "__main__":
    main()
