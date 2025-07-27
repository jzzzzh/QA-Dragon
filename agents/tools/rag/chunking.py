import re
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter


class TextSplitter():
    DEFAULT_SEPARATORS = ["\n", "<br>", '. ']
    DEFAULT_ATTRIBUTE_SEPARATORS = ["\n", "<br>"]
    DEFAULT_INVALID_TITLES = [
        "notes",
        "references",
        "see also",
        "external links",
        "footnotes",
        "further reading",
        "topics",
    ]
    DEFAULT_INVALID_ATTRIBUTES = ["website", "logo", "image"]
    DEFAULT_INVALID_SYMBOLS = ["{{", "}}", "\[\[", "\]\]", "<", ">"]


    def __init__(self, chunk_size: int = 100, overlap_size: int = 30) -> None:
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size

        self.text_splitter = RecursiveCharacterTextSplitter(
            separators=self.DEFAULT_SEPARATORS,
            keep_separator="end",
            chunk_size=self.chunk_size,
            chunk_overlap=self.overlap_size,
            length_function=self._word_count,
        )
    
    @staticmethod
    def _word_count(text: str) -> int:
        return len(text.split())

    @staticmethod
    def _split_article(article: str) -> List[dict]:
        # Split the article by section titles (== or ===)
        pattern = re.compile(r"==+(.*?)==+")

        last_pos = 0
        last_title = ""
        sections = []
        for match in pattern.finditer(article):
            start, end = match.span()

            if last_pos < start:
                content = article[last_pos:start].strip()
                sections.append({"title": last_title, "content": content})

            last_pos = end
            last_title = match.group(1).strip()

        # Add the last section
        if last_pos < len(article):
            content = article[last_pos:].strip()
            sections.append({"title": last_title, "content": content})

        return sections
    
    def _split_content(self, content: str) -> List[str]:
        # replace consecutive periods with a single period
        content = re.sub(r'\.{2,}', '.', content)
        splited_contents = self.text_splitter.split_text(content)
        splited_contents = [
            content.strip() for content in splited_contents if content.strip()
        ]
        return splited_contents
    
    def article_chunking(self, article: str) -> List[dict]:
        """
        Splits an article into chunks based on section titles and content length.
        Args:
            article (str): The article to be chunked.
            chunk_size (int): The maximum word count for each chunk.
        Returns:
            list: A list of dictionaries, each containing a title and content.
        """
        sections = self._split_article(article)
        
        chunks = []
        for section in sections:
            title = section["title"]
            content = section["content"]

            # Skip sections with invalid titles
            if title.lower() in self.DEFAULT_INVALID_TITLES:
                continue

            if content == "":
                continue
            
            splitted_content = self._split_content(content)
            chunks.extend([{"title": title, "content": content} for content in splitted_content])
        return chunks


    def attribute_chunking(self, attribute: dict) -> List[dict]:
        chunks = []
        for key, content in attribute.items():
            if key.lower() in self.DEFAULT_INVALID_ATTRIBUTES:
                continue
                
            if content is None or content == [] or content == {}:
                continue

            content = str(content).strip()
            content = re.sub(r"|".join(self.DEFAULT_INVALID_SYMBOLS), "", content)
            for line in re.split(r"|".join(self.DEFAULT_ATTRIBUTE_SEPARATORS), content):
                if not line.strip():
                    continue
                
                # Limit the number of words in each chunk
                words = line.split()
                if len(words) > self.chunk_size:
                    line = " ".join(words[:self.chunk_size])

                chunks.append({"title": key, "content": line})
        return chunks


if __name__ == "__main__":

    article = open("logs/article1.txt", "r").read()

    text_splitter = TextSplitter(chunk_size=100, overlap_size=30)
    chunks = text_splitter.article_chunking(article)
    for chunk in chunks:
        print(f"Title: {chunk['title']}")
        print(f"Content: {chunk['content']}")
        print('-' * 40)
