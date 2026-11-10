import json
import csv

with open("data/responses.json", "r" , encoding="utf-8-sig") as file:
    responses = json.load(file)

with open("data/questions.csv", "r", encoding="utf-8-sig") as file:
    questions = list(csv.DictReader(file))

with open("data/rules.json", "r", encoding="utf-8-sig") as file:
    rules = json.load(file)

sorted_rules = sorted(
    rules,
    key=lambda rule: rule.get("priority", 0),
    reverse=True
)

def get_response(intent):
    if intent in responses:
        return responses[intent]
    else:
        return responses["fallback"]

def find_intent(question):
    question_lower = question.lower()
    for rule in sorted_rules:

        skip = False

        for exclude in rule.get("exclude", []):
            if exclude in question_lower:
                skip = True
                break

        if skip:
            continue

        for keyword in rule["keywords"]:
            if keyword in question_lower:
                return rule["intent"]

    return "fallback"

question = input("You: ")

intent = find_intent(question)
answer = get_response(intent)

print("intent:", intent)
print("Bot:", answer)