# Minimal end-to-end workflow
INPUT prompt: Text

parts = RETRIEVE_PARTS(
  prompt,
  category="generic",
  top=20
)

selected = SELECT_PARTS(parts, top=5)
draft = COMPOSE(selected)
scores = EVALUATE(draft, evaluator="mock")

OUTPUT draft
