import sys
import spacy
from spacy.tokens import DocBin, Doc
from spacy.training import Example
from spacy.scorer import Scorer
from wasabi import Printer
import typer
from pathlib import Path

"""      - "python ./scripts/eval_ss_ost.py ./training/${vars.config_syn}/model-best ./training/${vars.config_sem}/model-best ./{vars.test_file}"
"""


msg = Printer()

def get_final_label(syn, sem):
  d = {
       ("NA", "NA"): "NA",
       ("NA", "Se"): "Se",
       ("Sy", "NA"): "Sy",
       ("Sy", "Se"): "SS"
  }
  return d[(syn, sem)]

def get_best_label(d):
  return max(d, key=d.get)



def main(model_sem: Path, model_syn: Path, test_syn_file: Path, test_sem_file: Path, output: Path):
    nlp_sem = spacy.load(model_sem)
    nlp_syn = spacy.load(model_syn)

    docbin_sem = DocBin().from_disk(test_sem_file)
    docs_sem = sorted([doc for doc in docbin_sem.get_docs(nlp_sem.vocab)], key = lambda d: d.text)
    docbin_syn = DocBin().from_disk(test_syn_file)
    docs_syn = sorted([doc for doc in docbin_syn.get_docs(nlp_syn.vocab)], key = lambda d: d.text)

    preds = []
    golds = []

    debug = open(output.with_name("debug.out"), "w")

    examples = []
    for i, (doc_syn, doc_sem) in enumerate(zip(docs_syn, docs_sem)):
      gold_label=get_final_label(get_best_label(doc_syn.cats), get_best_label(doc_sem.cats))
      golds.append(gold_label)
      pred_sem = nlp_sem(doc_sem.text)
      pred_syn = nlp_syn(doc_syn.text)
      pred_label = get_final_label(get_best_label(pred_syn.cats), get_best_label(pred_sem.cats))
      preds.append(pred_label)

      gold = Doc(pred_sem.vocab).from_bytes(pred_sem.to_bytes())
      gold.cats ={"NA": 0, "SS":0, "Se":0, "Sy":0}
      gold.cats[gold_label] = 1

      pred = Doc(pred_sem.vocab).from_bytes(pred_sem.to_bytes())
      pred.cats ={"NA": 0, "SS":0, "Se":0, "Sy":0}
      pred.cats[pred_label] = 1

      examples.append(Example(pred, gold))

      print(i, "\n syn ", doc_syn, doc_syn.cats, f"pred_syn: {pred_syn.cats}", "\n sem ", doc_sem, doc_sem.cats, f"pred_sem: {pred_sem.cats}", f"final: pred {pred.cats} VS gold {gold.cats}", file=debug)

    debug.close()
    scorer = Scorer()
    scores = scorer.score_cats(examples, "cats", labels = ["NA", "Sy", "Se", "SS"])
    import json
    with open(output, "w") as fout:
    	json.dump(scores, fout, indent=4)


if __name__ == "__main__":
    typer.run(main)
