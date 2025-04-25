# 🚗 OBD Code Decoder – RAG-based Car Diagnostic Simulation

This project simulates an in-car system that decodes OBD (On-Board Diagnostics) error codes and provides helpful messages like issue description, cause, and fix using a Retrieval-Augmented Generation (RAG) system powered by Gemini.

---

## ⚙️ How It Works

1. `.md` files (converted from manuals and code lists) are embedded into a vector DB using Chroma.
2. When a car error code like `P0522` is input, the system finds the relevant chunks.
3. The Gemini model analyzes the context and replies like a car dashboard would.

---

## 🧪 How to Run Locally

1. Clone the repo
2. Set up virtual environment
3. Install dependencies
4. Create a .env file in the root folder and add your GEMINI_API_KEY
5. Create the database: python dbcreate.py
6. Run the program: python queryhandle.py

