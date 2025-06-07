import sacrebleu

# Example data
refs = ["this is a test .", "another example sentence ."]
hyps = ["this is test .", "another example ."]

# sacrebleu expects list of reference‐lists
# here we have only one reference per example
bleu = sacrebleu.corpus_bleu(hyps, [refs])

print(f"BLEU score: {bleu.score:.2f}")