import os
from dotenv import load_dotenv
import openai

# Load Groq API key
load_dotenv()
print("Loaded API Key:", os.getenv("GROQ_API_KEY"))
openai.api_key = os.getenv("GROQ_API_KEY")
openai.api_base = "https://api.groq.com/openai/v1"


def get_prompt(user_code):
    return f"""
You are a parallelization expert. I will give you a serial Python function. Your task is to optimize it for performance by either:

1. Converting it into a parallel version using Numba (with @njit and prange), OR
2. Converting it into a fully vectorized version using NumPy — if that's possible.

📌 Follow these rules strictly:

- Only parallelize loops that are **independent** (i.e., iterations don’t depend on previous or future ones).
- Use `@njit(parallel=True)` and `prange()` for loops that are safe to parallelize.
- If the loop can be expressed as a single NumPy expression (element-wise), prefer vectorization over parallelization.
- Avoid modifying global variables or printing inside loops.
- Do not use `with prange(...)`. Use `for i in prange(...)`.
- Keep the function signature and input/output behavior the same.
- The goal is to **reduce runtime** without changing the result.

Return both:
- The optimized version (parallel or vectorized).
- A short explanation of what you did and why.

Here is the serial code:
```python
{user_code}
```"""

# Code generation function
def generate_parallel_code(user_code, model="llama3-70b-8192"):
    user_prompt = get_prompt(user_code)
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.2,
    )
    return response['choices'][0]['message']['content']
