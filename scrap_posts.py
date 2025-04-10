import json
import logging
import os
import re
import sys
from time import sleep

import requests
from lxml.html import fromstring
from tenacity import retry, stop_after_attempt, wait_chain, wait_fixed, RetryError

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/111.0.0.0 Safari/537.36"
)


def extract_with_regex(name, pattern, text, group_idx=1, err_msg=""):
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(group_idx)
    raise ValueError(f"{name} not found in the text. {err_msg}")


def load_state(file):
    if os.path.exists(file):
        with open(file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"threads": []}


def save_state(file, state):
    with open(file, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=4, ensure_ascii=False)


def extract_text_from_html(html_content):
    return " ".join([node.strip() for node in html_content.xpath('./text()') if node.strip()])


def replace_img_with_text(tree, shortname, replacement_text):
    # Para cada imagem com data-shortname igual ao desejado
    for img in tree.xpath(f"//img[@data-shortname='{shortname}']"):
        parent = img.getparent()
        if parent is None:
            continue
        # Obtém o índice do elemento imagem no pai
        idx = parent.index(img)
        # Insere o texto no lugar da imagem
        # Se for o primeiro nó, adiciona ao .text do pai,
        # caso contrário, adiciona ao .tail do nó anterior
        if idx == 0:
            if parent.text:
                parent.text += replacement_text
            else:
                parent.text = replacement_text
        else:
            previous = parent[idx - 1]
            if previous.tail:
                previous.tail += replacement_text
            else:
                previous.tail = replacement_text
        # Remove o elemento imagem
        parent.remove(img)


class ForumScraper:
    def __init__(self, base_url: str, cookie: str, state_file: str, flood_interval: float = 3):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})
        self.session.cookies.update({"xf_user": cookie})
        self.payload = {"_xfToken": self._get_authorization()}
        self.state_file = state_file
        self.flood_interval = flood_interval

        self.state = load_state(state_file)

    def _get_authorization(self):
        res = self.session.get(self.base_url)
        res.raise_for_status()
        html = fromstring(res.content)
        token = html.xpath('.//input[@name="_xfToken"]/@value')
        if not token:
            raise RuntimeError("Not logged in!")
        return token[0]

    def search(self, user_id):
        threads = self.state["threads"]

        page_url = self.state.get("last_page_url", self.search_url(user_id))
        sleep(self.flood_interval)

        if not page_url:
            return []

        while True:
            print("page_url", page_url)
            thread_list, page_next = self._extract_posts(page_url)
            if not thread_list:
                break

            threads.extend(thread_list)

            if not page_next:
                page_url = self._get_older_page_url(page_url)
                if not page_url:
                    break
                continue

            page_url = self.base_url + page_next[0]

            self.state["threads"] = threads
            self.state["last_page_url"] = page_url
            save_state(self.state_file, self.state)

            sleep(self.flood_interval)

        return threads

    def search_url(self, user_id):
        search_url = f"{self.base_url}/search/member?user_id={user_id}"
        response = self.session.get(search_url, allow_redirects=False)
        response.raise_for_status()
        return response.headers.get("Location")

    def _extract_posts(self, page_url):
        response = self.session.get(page_url)
        response.raise_for_status()
        sleep(self.flood_interval)

        html = fromstring(response.content.decode(response.apparent_encoding, errors='ignore'))

        page_next = html.xpath('//a[@class="pageNav-jump pageNav-jump--next"]/@href')

        threads = []

        extract_results = html.find_class("block-body")

        if not extract_results:
            return threads, page_next

        for li in extract_results[0]:
            if li.xpath("./div/div/div[2]/ul/li[2]")[0].text_content() == "Thread":
                continue

            title_elem = li.xpath("div/div/h3/a")[0]
            title = title_elem.text_content().replace("\u00A0", " ")
            post_id = title_elem.attrib["href"].split("/post-")[1]
            quote, post = self.get_post_content(post_id)

            print({"title": title, "post_id": post_id, "quote": quote, "post": post})

            threads.append({"title": title, "post_id": post_id, "quote": quote, "post": post})

        return threads, page_next

    def _get_older_page_url(self, page_url):
        response = self.session.get(page_url)
        response.raise_for_status()
        html = fromstring(response.content)

        older_elem = html.find_class("block-footer-controls")
        if older_elem:
            return self.base_url + older_elem[0].xpath(".//a/@href")[0]
        return None

    def get_post_content(self, post_id: str) -> [str, str]:
        url = f"{self.base_url}/posts/{post_id}/show?_xfToken={self.payload['_xfToken']}&_xfResponseType=json"
        res = self.session.get(url)
        res.raise_for_status()
        tree = fromstring(res.json()["html"]["content"])

        blockquote = tree.xpath('//div[@class="bbWrapper"]/blockquote/div[2]/div[1]')
        quote = extract_text_from_html(blockquote[0]) if blockquote else ""

        texts = ""
        bb_wrapper = tree.xpath('//div[@class="bbWrapper"]')
        if bb_wrapper:
            replace_img_with_text(bb_wrapper[0], "negresco", "preto")
            texts = extract_text_from_html(bb_wrapper[0])

        return quote, texts


# Configure logging
logging.basicConfig(
    filename="log.txt",
    level=logging.ERROR,
    format="%(asctime)s - Attempt %(retry_number)d - Error: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)


def log_attempt(retry_state):
    """Logs each failed attempt with the error message."""
    logging.error(
        retry_state.outcome.exception(),
        # stack_info=True,
        extra={"retry_number": retry_state.attempt_number}
    )


def execute(base_url: str, cookie: str, state_file: str, user_id: str):
    """Executes the given function with retry handling."""

    @retry(
        stop=stop_after_attempt(5),  # Maximum of 5 attempts
        wait=wait_chain(
            *[wait_fixed(5) for _ in range(2)],  # 2 attempts with 5s delay
            *[wait_fixed(30) for _ in range(2)],  # 2 attempts with 30s delay
            wait_fixed(3600)  # Final attempt after 1 hour
        ),
        after=log_attempt  # Log each failed attempt
    )
    def wrapped_func():
        scraper = ForumScraper(base_url, cookie, state_file, flood_interval=0.5)
        scraper.search(user_id)
        print("Done!")

    try:
        wrapped_func()
    except RetryError as err:
        print(f"All attempts failed. Exiting the program. Error: {err}")
        sys.exit(1)  # Exit the program with an error


if __name__ == "__main__":
    BASE_URL = "https://forum.onipotentes.club"
    COOKIE = "meu_xf_user_cookie"
    STATE_FILE = "fiat_mobi_posts.json"
    USER_ID = "349"
    execute(BASE_URL, COOKIE, STATE_FILE, USER_ID)
