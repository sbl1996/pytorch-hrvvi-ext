# from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
# from toolz import curry
# from toolz.curried import get
#
# from horch.train.metrics import Average
#
#
# @curry
# def take_until_eos(eos_index, tokens):
#     for i, token in enumerate(tokens):
#         if token == eos_index:
#             return tokens[:i]
#     return tokens
#
#
# def bleu(preds, y, eos_index):
#     preds = preds.argmax(dim=1)
#     output = lmap(take_until_eos(eos_index), preds.tolist())
#     target = lmap(take_until_eos(eos_index), y.tolist())
#     target = lmap(lambda x: [x], target)
#     score = corpus_bleu(
#         target, output, smoothing_function=SmoothingFunction().method1)
#     return score
#
#
# class Bleu(Average):
#
#     def __init__(self, eos_index):
#         self.eos_index = eos_index
#         super().__init__(output_transform=self.output_transform)
#
#     def output_transform(self, output):
#         preds, target, batch_size = get(["preds", "target", "batch_size"], output)
#         preds = preds[0]
#         target = target[0]
#         return bleu(preds, target, self.eos_index), batch_size