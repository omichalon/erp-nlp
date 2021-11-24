import json

import typer
from pathlib import Path
import random
import spacy
from spacy.tokens import Span, DocBin, Doc
from spacy.vocab import Vocab
from wasabi import Printer

msg = Printer()



def convert(input_file, output_file, nlp, conversion_dict):
  docbin = DocBin().from_disk(input_file)
  docs = [doc for doc in docbin.get_docs(nlp.vocab)]

  categories = set([v for v in conversion_dict.values()])
  for doc in docs:
      old_cats = dict(doc.cats)
      doc.cats = {category: 0 for category in categories}
      for cat in old_cats:
          doc.cats[conversion_dict[cat]] = old_cats[cat]

  docbin = DocBin(docs=docs, store_user_data=True)
  docbin.to_disk(output_file)


def main(corpus_loc: Path, train_file: Path, dev_file: Path, test_file: Path, train_sem_file: Path, dev_sem_file: Path, test_sem_file: Path, train_syn_file: Path, dev_syn_file: Path, test_syn_file: Path):
    """Creating the corpus from the Prodigy annotations."""
    vocab = Vocab()

    nlp = spacy.load("en_core_web_lg", disable=['tok2vec', 'ner'])

    docs = {"train": [], "dev": [], "test": [], "all": []}

    categories = set()
    with corpus_loc.open("r", encoding="utf8") as corpusfile:
        for line in corpusfile:
            _, _, _, effect = line.strip().split("#")
            categories.add(effect)

    with corpus_loc.open("r", encoding="utf8") as corpusfile:
        for line in corpusfile:
            _id, text, trigger, effect = line.strip().split("#")
            doc = nlp(text)
            doc.cats = {category: 0 for category in categories}
            doc.cats[effect] = 1
            docs["all"].append(doc)


    split_1 = int(0.7 * len(docs["all"]))
    split_2 = int(0.9 * len(docs["all"]))
    random.shuffle(docs["all"])
    docs["train"] = docs["all"][:split_1]
    docs["test"] = docs["all"][split_1:split_2]
    docs["dev"] = docs["all"][split_2:]

    # Normal
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

    # converting for syn
    conversion_dict = {"SS": "Sy", "Sy": "Sy", "Se": "NA", "NA": "NA"}
    convert(test_file, test_syn_file, nlp, conversion_dict)
    convert(train_file, train_syn_file, nlp, conversion_dict)
    convert(dev_file, dev_syn_file, nlp, conversion_dict)

    # converting for sem
    conversion_dict = {"SS": "Se", "Sy": "NA", "Se": "Se", "NA": "NA"}
    convert(test_file, test_sem_file, nlp, conversion_dict)
    convert(train_file, train_sem_file, nlp, conversion_dict)
    convert(dev_file, dev_sem_file, nlp, conversion_dict)



if __name__ == "__main__":
    typer.run(main)
