import sys
import dspy

lm = dspy.LM(
  "ollama_chat/jeffnyman/ts-reasoner", api_base="http://localhost:11434", api_key=""
)

dspy.configure(lm=lm)


# --- Step 1 Signature ---
# Takes a question and produces a detailed answer. This is the same
# contract as script.py and script2.py, but now it's one stage in a
# larger pipeline rather than the whole program.
class DetailedAnswerSignature(dspy.Signature):
  question: str = dspy.InputField()
  detailed_answer: str = dspy.OutputField()


# --- Step 2 Signature ---
# Takes the detailed answer from step 1 and distills it to a single
# sentence. Notice this signature knows nothing about the original
# question; it only sees what the previous step produced. Each
# signature declares a local contract, not a global one.
class SummarySignature(dspy.Signature):
  detailed_answer: str = dspy.InputField()
  summary: str = dspy.OutputField(desc="A single concise sentence.")


# --- Pipeline Module ---
# A single dspy.Module can own multiple predictors. The forward()
# method defines how data flows between them: this is the pipeline.
# DSPy compiles each predictor against its own signature, so two
# distinct prompts are generated and sent to the LM in sequence.
class QAPipeline(dspy.Module):
  def __init__(self):
    super().__init__()

    # Step 1: expand the question into a detailed answer.
    self.answer = dspy.ChainOfThought(DetailedAnswerSignature)

    # Step 2: compress that answer into a summary.
    self.summarize = dspy.ChainOfThought(SummarySignature)

  def forward(self, question: str):
    # The output of step 1 becomes the input of step 2. Any DSPy
    # Prediction objects support attribute access, which means that
    # detailed_answer flows through as a plain string; no manual
    # parsing or string stitching required.
    expanded = self.answer(question=question)
    summarized = self.summarize(detailed_answer=expanded.detailed_answer)
    return summarized


if __name__ == "__main__":
  q = (
    sys.argv[1]
    if len(sys.argv) > 1
    else "What is the answer to life, the universe and everything?"
  )
  print(q)

  pipeline = QAPipeline()

  # The Prediction object here comes from the final step, which is
  # SummarySignature, so it carries the `reasoning` and `summary`
  # fields. The intermediate detailed_answer is consumed internally
  # by the pipeline.
  result = pipeline(question=q)
  print("=== Prediction ===")
  print(result)

  # n=2 shows both compiled prompts in sequence, one per pipeline
  # step. Comparing them reveals that DSPy generated two structurally
  # different prompts from two different signatures, all from one
  # forward() call.
  print("=== Generated Prompts ===")
  dspy.inspect_history(n=2)
  print("=========================\n")
