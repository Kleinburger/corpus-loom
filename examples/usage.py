from typing import List
from pydantic import BaseModel, Field
from corpusloom import OllamaClient

def main():
    client = OllamaClient(model="gpt-oss:20b", keep_alive="10m", default_options={"temperature": 0.1, "num_ctx": 16384})
    client.add_files(["README.md"])
    ctx = client.build_context("outline features of this library", top_k=3)
    print("Context:\n", ctx)
    res = client.generate(f"Using the context below, summarize the library's features.\n{ctx}")
    print("\n\nGenerated summary:\n", res.response_text)
    class MiniPlan(BaseModel):
        requirement_id: str
        overview: str
        cases: List[str] = Field(default_factory=list)
    obj = client.generate_json(prompt="Create a MiniPlan for REQ-123 from the context below.\n" + ctx, schema=MiniPlan)
    print("\nValidated JSON:", obj)

if __name__ == "__main__":
    main()
