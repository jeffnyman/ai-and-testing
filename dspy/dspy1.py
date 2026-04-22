import sys
import dspy

# --- Layer 1: Backend ---
# dspy.LM wraps the language model. The string uses LiteLLM's
# provider/model format: "ollama_chat" is the provider, the rest
# is the model name. DSPy delegates all actual inference to this
# object.
lm = dspy.LM(
  "ollama_chat/jeffnyman/ts-reasoner", api_base="http://localhost:11434", api_key=""
)

# Registers the LM globally so every DSPy module in this process
# uses it without needing it passed around explicitly.
dspy.configure(lm=lm)


# --- Layer 2: Declaration ---
# A Signature declares the contract of a task: what goes in and
# what comes out, with types. It does not say anything about how
# to ask for it; that is DSPy's job. Think of it like a function's
# type signature in a typed language: it describes the interface,
# not the implementation.
class QASignature(dspy.Signature):
  # InputField and OutputField mark which side of the contract
  # each attribute belongs to. DSPy uses these annotations to build
  # the prompt structure automatically. Field names become section
  # headers, types become constraints, and the ordering determines
  # the prompt layout.
  question: str = dspy.InputField()
  answer: str = dspy.OutputField()


# --- Layer 3: Module ---
# dspy.Module is DSPy's equivalent of a PyTorch nn.Module, meaning a
# composable unit of computation. Simple programs have one module;
# complex pipelines chain many together, each with its own signature
# and predictor.
class SimpleQA(dspy.Module):
  def __init__(self):
    super().__init__()

    # dspy.Predict is the simplest DSPy "layer". Given a Signature,
    # it compiles a structured prompt, calls the LM, and parses the
    # response back into typed fields. The idea is that this handles
    # the prompt engineering so you don't have to write or maintain
    # prompt strings by hand.
    self.predictor = dspy.Predict(QASignature)

  # forward() is the execution entry point, called when you invoke
  # the module like a function (qa(question=...)). DSPy follows the
  # convention of PyTorch here.
  def forward(self, question: str):
    return self.predictor(question=question)


if __name__ == "__main__":
  q = (
    sys.argv[1]
    if len(sys.argv) > 1
    else "What is the answer to life, the universe and everything?"
  )
  print(q)

  # Instantiating the module compiles the predictor against the
  # signature.
  qa = SimpleQA()

  # Calling the module runs forward(), which sends the compiled
  # prompt to the LM and returns a Prediction object: which is a
  # structured result with typed fields, not a raw string that you
  # would have to parse yourself.
  result = qa(question=q)
  print("=== Prediction ===")
  print(result)

  # inspect_history shows the exact prompt DSPy generated and the
  # raw model response. This makes the "compilation" visible: you
  # can see the system message DSPy wrote, the section markers it
  # uses to parse the output, and  how your input was embedded.
  # n=1 shows the most recent call only.
  print("=== Generated Prompt ===")
  dspy.inspect_history(n=1)
  print("========================\n")
