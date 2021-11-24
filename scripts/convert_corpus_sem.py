import json

import typer
from pathlib import Path
import random
import spacy
from spacy.tokens import Span, DocBin, Doc
from spacy.vocab import Vocab
from wasabi import Printer

msg = Printer()




def main(corpus_loc: Path, train_file: Path, dev_file: Path, test_file: Path):
    vocab = Vocab()

    nlp = spacy.load("en_core_web_lg", disable=['tok2vec', 'ner'])

    docs = {"train": [], "dev": [], "test": [], "all": []}

    conversion_dict = {"SS": "Se", "Sy": "NA", "Se": "Se", "NA": "NA"}

    categories = set([v for v in conversion_dict.values()])

    with corpus_loc.open("r", encoding="utf8") as corpusfile:
        for line in corpusfile:
            _id, text, trigger, effect = line.strip().split("#")
            doc = nlp(text)
            doc.cats = {category: 0 for category in categories}
            doc.cats[conversion_dict[effect]] = 1
            docs["all"].append(doc)

    split_1 = int(0.7 * len(docs["all"]))
    split_2 = int(0.9 * len(docs["all"]))
    random.shuffle(docs["all"])
    docs["train"] = docs["all"][:split_1]
    docs["test"] = docs["all"][split_1:split_2]
    docs["dev"] = docs["all"][split_2:]

    docbin_all(docs, train_file, test_file, dev_file)


def docbin_all(docs, train_file, test_file, dev_file):

    docbin = DocBin(docs=docs["train"], store_user_data=True)
    docbin.to_disk(train_file)
    msg.info(
        f"{len(docs['train'])} training sentences "
    )

    docbin = DocBin(docs=docs["dev"], store_user_data=True)
    docbin.to_disk(dev_file)
    msg.info(
        f"{len(docs['dev'])} dev sentences"
    )

    docbin = DocBin(docs=docs["test"], store_user_data=True)
    docbin.to_disk(test_file)
    msg.info(
        f"{len(docs['test'])} test sentences"
    )


if __name__ == "__main__":
    typer.run(main)
