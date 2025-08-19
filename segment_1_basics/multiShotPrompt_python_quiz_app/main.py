from utils.models import chat, load_prompts
import json, re

SYSTEM_PROMPT = {
    "question-writer": (
        "You are a Python quiz assistant.\n"
        "- Generate ONE question in the style of the examples.\n"
        "- Return ONLY raw JSON (no code fences, no prose).\n"
        "- Fields: question, correct_answer, explanation, type ('mcq' or 'short'), choices (optional for mcq).\n"
        "- Use only valid ASCII identifiers and valid Python syntax.\n"
        "- Keep questions short and unambiguous.\n"
        "- while providing a code block, make sure to format properly"
    ),
    "grader": (
        "You are a Python quiz grader.\n"
        "- Compare the student's answer to the gold answer.\n"
        "- Return ONLY JSON with fields: correctness ('correct'|'incorrect'), short_feedback.\n"
        "- Be encouraging if incorrect; accept case-insensitive equivalents."
    ),
    "hinter": (
        "You are a Python tutor.\n"
        "- Provide ONE short hint (<= 20 words) that nudges, not spoils.\n"
        "- Return ONLY the hint text."
    )
}

def parse_json_block(text: str) -> dict:
    # strip code fences if present
    text = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", text.strip(), flags=re.IGNORECASE)
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1:
        return json.loads(text[start:end+1])
    return json.loads(text)

def generate_question(
    topic: str,
    difficulty: str,
    model: str = "free"
)->str:
    examples = load_prompts("prompts/question_writer.txt")
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT["question-writer"]},
        {"role": "user", "content": examples},
        {"role": "user", "content":f"""Topic: {topic}
                    Difficulty: {difficulty}
                    Generate ONE new question as JSON.

                    Constraints:
                    - If type is "mcq", include 3–5 choices and ensure 'correct_answer' is one of them.
                    - For outputs that are lists/strings/dicts, represent them exactly as Python would print.
                    - Do NOT use emojis or non-ASCII identifiers.
                    - Avoid invalid syntax like stray colons or incomplete slices."""}
    ]
    output = chat(model, messages, as_json=True)
    q = parse_json_block(output)

    # safety: ensure mcq answer is in choices
    if (q.get("type","").lower() == "mcq") and q.get("choices") and q.get("correct_answer") not in q["choices"]:
        q["choices"].append(q["correct_answer"])
    return q 

def get_hint(
    question: str,
    model: str = "free"
) -> str:
    examples = load_prompts("prompts/hinter.txt")
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT["hinter"]},
        {"role": "user", "content": examples.strip()},
        {"role": "user", "content": (
            "Now, for the following question, generate ONLY ONE new hint (do not repeat any examples above):\n"
            f"Question: {question}\n"
            "Hint:"
        )}
    ]
    return chat(model, messages).strip()

def grade_answers(
    question: str,
    gold: str,
    student: str,
    model: str = "free"
) -> dict:
    examples = load_prompts("prompts/grader.txt")
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT["grader"]},
        {"role": "user", "content": examples.strip()},
        {"role": "user", "content": (
            f"Question: {question}\n"
            f"Gold Answer: {gold}\n"
            f"Student Answer: {student}\n"
            "If the student's answer matches the gold answer (case-insensitive), reply with JSON: "
            "{'correctness': 'correct', 'short_feedback': 'Inspirational feedback'}.\n"
            "If not, reply with JSON: {'correctness': 'incorrect', 'short_feedback': 'Encouraging feedback and why it is incorrect'}."
        )}
    ]
    output = chat(model, messages)
    try:
        return json.loads(output)
    except Exception:
        return {"correctness": "unknown", "short_feedback": output.strip()[:200]}

def run_quiz():
    model = input("Choose model(free/paid):").strip().lower()
    topic = input("Enter topic (e.g. Lists, Functions): ").strip()
    difficulty = input("Difficulty (Easy/Medium/Hard): ").strip()

    print("\n === GENERATING QUESTION ===")
    q = generate_question(topic, difficulty, model)
    
    q_text = q.get("question", "No Question?")
    q_type = (q.get("type") or "short").lower()
    choices = q.get("choices") or []
    gold = (q.get("correct_answer") or "").strip()
    expl = (q.get("explanation") or "").strip()

    print("\nQ:", q_text)
    if q_type == "mcq" and choices:
        for i, c in enumerate(choices, 1):
            print(f"  {i}. {c}")

    user = input("\nYour answer (or type 'hint'): ").strip()
    if user.lower() == "hint":
        print("Hint:", get_hint(q_text, model=model))
        user = input("Your answer: ").strip()

    print("\n=== GRADING ===")
    result = grade_answers(q_text, gold, user, model=model)
    print("Result:", result.get("correctness", "unknown").upper())
    print("Feedback:", result.get("short_feedback", ""))
    print("Gold:", gold)
    print("Why:", expl)

if __name__=="__main__":
    run_quiz()