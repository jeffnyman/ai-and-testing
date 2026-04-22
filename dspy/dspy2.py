import sys
import dspy

# Same backend setup as script.py. Nothing changes here; the LM is
# indifferent to which predictor strategy the module uses.
lm = dspy.LM(
  "ollama_chat/jeffnyman/ts-reasoner", api_base="http://localhost:11434", api_key=""
)

dspy.configure(lm=lm)


# Same signature as script.py, unchanged. This is the point: the
# declaration of what you want is completely independent of how
# DSPy asks for it.
class QASignature(dspy.Signature):
  question: str = dspy.InputField()
  answer: str = dspy.OutputField()


class ChainOfThoughtQA(dspy.Module):
  def __init__(self):
    super().__init__()

    # The only change from script.py: Predict becomes ChainOfThought.
    # DSPy will inject a reasoning step into the compiled prompt
    # automatically and add a `reasoning` field to the Prediction
    # object. No prompt rewriting required on your part.
    self.predictor = dspy.ChainOfThought(QASignature)

  def forward(self, question: str):
    return self.predictor(question=question)


if __name__ == "__main__":
  q = (
    sys.argv[1]
    if len(sys.argv) > 1
    else "What is the answer to life, the universe and everything?"
  )
  print(q)

  qa = ChainOfThoughtQA()

  # The Prediction object now has two fields: `reasoning` and
  # `answer`. DSPy extended the signature internally to add the
  # reasoning step. You never declared it, and you didn't touch
  # the prompt.
  result = qa(question=q)
  print("=== Prediction ===")
  print(result)

  # Compare this generated prompt carefully against the one from
  # script.py. The system message now includes a `reasoning` output
  # field that appears before `answer`, and the response format
  # reflects that ordering. DSPy compiled a different prompt from
  # the same signature by changing one word.
  print("=== Generated Prompt ===")
  dspy.inspect_history(n=1)
  print("========================\n")
