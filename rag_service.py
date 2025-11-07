from fastapi import FastAPI
app = FastAPI(title="RAG Service (placeholder)")
@app.get("/up")
def up(): return {"ok": True}
