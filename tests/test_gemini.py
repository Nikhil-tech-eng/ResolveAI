# import os, time
# from dotenv import load_dotenv
# load_dotenv()

# from google import genai

# key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
# print("Key found:", bool(key))

# client = genai.Client(api_key=key)
# t = time.perf_counter()
# try:
#     r = client.models.generate_content(model="gemini-3.8-flash", contents="say hi")
#     print("Reply:", r.text)
# except Exception as e:
#     print("ERROR:", repr(e))
# print(f"Took {time.perf_counter() - t:.1f}s")

import os, time
from dotenv import load_dotenv
load_dotenv()

from google import genai

key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
client = genai.Client(api_key=key)

models = [
    "gemini-3.8-flash",
    "gemini-flash-latest",
    "gemini-flash-lite-latest",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
]

for m in models:
    t = time.perf_counter()
    try:
        r = client.models.generate_content(model=m, contents="say hi")
        print(f"OK    {m}: {r.text.strip()[:40]!r} ({time.perf_counter() - t:.1f}s)")
    except Exception as e:
        print(f"FAIL  {m}: {str(e)[:80]} ({time.perf_counter() - t:.1f}s)")