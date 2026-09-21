from pathlib import Path
import sys

PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]

sys.path.insert(
    0,
    str(PROJECT_ROOT)
)

from src.rag.llm_client import call_llm

OLLAMA_MODEL = (
    "llama3.2:latest"
)


def analysis_agent(
    question,
    retrieved_documents
):

    print(
        "\n[ANALYSIS AGENT]"
    )

    print(
        "Analyzing retrieved information..."
    )

    context = ""

    for i, document in enumerate(
        retrieved_documents,
        start=1
    ):

        context += f"""
SOURCE {i}

Page: {document['page']}

Source: {document['source']}

{document['text']}

--------------------------------
"""

    prompt = f"""
You are the Analysis Agent in a
document-grounded RAG decision
support system.

Analyze ONLY the retrieved
document context.

QUESTION:

{question}

RETRIEVED CONTEXT:

{context}


IMPORTANT RULES:

1. Use ONLY information present
   in the retrieved context.

2. Do NOT use outside knowledge.

3. Do NOT invent information.

4. Do NOT make assumptions.

5. Treat information according
   to the section where it appears.

6. "TECHNICAL SKILLS" means
   technical skills.

7. "AREA OF INTEREST" means
   interests, NOT technical skills.

8. "PROJECTS" can be used as
   supporting evidence for skills,
   but do not invent skills from
   project names.

9. If a technical skill is
   explicitly listed, include it
   even if it appears only once.

10. Do NOT decide that a skill is
    stronger simply because it is
    repeated in multiple sources.

11. Do NOT infer proficiency level
    unless the document explicitly
    states it.

12. Do NOT classify personality
    development or soft skills as
    technical skills.

13. Do NOT include a
    "Skills Not Mentioned" section.

14. Clearly separate facts directly
    stated in the document from
    supporting evidence.

15. If the requested information
    is not available in the context,
    clearly state that it is not
    available.


For questions about technical
skills:

- Extract the technical skills
  explicitly stated in the
  TECHNICAL SKILLS section.

- Group them according to the
  categories shown in the document.

- Use the professional summary
  and projects only as supporting
  evidence.

- Do not confuse AREA OF INTEREST
  with TECHNICAL SKILLS.

- Do not rank skills unless the
  document provides evidence for
  ranking.


Provide a concise and structured
analysis.

ANALYSIS:
"""

    return call_llm(
        prompt,
        model=OLLAMA_MODEL,
        temperature=0.0
    )