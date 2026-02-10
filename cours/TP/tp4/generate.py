from concurrent.futures import ThreadPoolExecutor, as_completed
from csv import DictReader
from json import dumps
from random import choice, randint, sample
from threading import Lock

from requests import post

INPUT_FILE = "class_builder.csv"
OUTPUT_FILE = "dd_rpg_questions_10000.jsonl"

# ---------- Load heroes ----------
heroes = []
with open(INPUT_FILE, encoding="utf-8") as f:
    reader = DictReader(f)
    for row in reader:
        heroes.append(row)

# ---------- Scenario pools ----------
maps = ["Ruins", "Warrens", "Weald", "Cove", "Courtyard", "Farmstead", "Catacombs"]

difficulties = ["Veteran", "Champion", "Dark", "Nightmare"]

threat_profiles = [
    ("high stun and blight pressure", ["Stun Resistance", "Blight Resistance"]),
    ("heavy bleed enemies", ["Bleed Resistance"]),
    ("stress and debuff casters", ["Debuff Resistance"]),
    ("high damage crit attackers", ["Max HP", "Deathblow Resistance"]),
    ("movement disruption enemies", ["Move Resistance", "Movement Backwards"]),
    ("trap heavy corridors", ["Trap Disarm Chance"]),
]

fight_lengths = ["short fights", "mixed fights", "long fights expected"]
retreat = ["low", "very low", "moderate"]

extra_constraints = [
    "Stress is the primary failure cause",
    "Backline enemies must be eliminated early",
    "Frontline enemies have high protection",
    "Survivability is more important than burst",
    "Turn order advantage is critical",
    "Movement disruption is frequent",
]

light_levels = [
    (
        "Radiant",
        "High light",
        "+Dodge bonus, higher surprise chance, lower stress gain",
        "no accuracy or damage bonus",
    ),
    ("Dim", "Medium light", "small crit bonus, normal stress", "no bonus"),
    (
        "Shadowy",
        "Low light",
        "+crit chance, higher stress gain",
        "+5% Accuracy, +10% Damage",
    ),
    (
        "Dark",
        "Very low light",
        "+crit chance, high stress gain, higher surprise risk",
        "+10% Accuracy, +15% Damage",
    ),
    (
        "Pitch Black",
        "Minimum light",
        "+crit chance, very high stress gain, very high surprise risk",
        "+12.5% Accuracy, +25% Damage",
    ),
]

# ---------- Formatting ----------


def hero_line(h):
    return (
        f"{h['Name Of Class']} (Resolve {h['Resolve Level']}) — "
        f"HP {h['Max HP']}, Dodge {h['Dodge']}, Prot {h['Prot']}, Speed {h['Speed']}, "
        f"DMG {h['Damage Min']}-{h['Damage Max']}, "
        f"Crit {h['Critical Chance % (With Rank Appropriate Weapons)']}%, "
        f"StunRes {h['Stun Resistance']}, MoveRes {h['Move Resistance']}, "
        f"BlightRes {h['Blight Resistance']}, BleedRes {h['Bleed Resistance']}, "
        f"DebuffRes {h['Debuff Resistance']}, DeathblowRes {h['Deathblow Resistance']}"
    )


# ---------- Build question ----------


def build_question(i):

    m = choice(maps)
    d = choice(difficulties)
    f = choice(fight_lengths)
    r = choice(retreat)

    threat, stat_focus = choice(threat_profiles)

    team_pool = sample(heroes, randint(7, 9))
    constraints = sample(extra_constraints, 2)
    light_level, light_desc, ally_effects, enemy_effects = choice(light_levels)

    lines = []
    lines.append("You are playing a turn-based tactical RPG dungeon run.\n")

    lines.append("Mission state:")
    lines.append(f"- Map: {m}")
    lines.append(f"- Difficulty: {d}")
    lines.append(f"- Fight length: {f}")
    lines.append(f"- Enemy profile: {threat}")
    lines.append(f"- Retreat chance: {r}")
    lines.append("- Hero death = mission failure\n")

    lines.append("Light level:")
    lines.append(f"- Level: {light_level} ({light_desc})")
    lines.append(f"- Ally effects: {ally_effects}")
    lines.append(f"- Enemy effects: {enemy_effects}\n")

    lines.append("Available heroes:")
    for h in team_pool:
        lines.append(f"- {hero_line(h)}")

    lines.append("\nConstraints:")
    lines.append("- Team size: choose exactly 3 heroes")
    for c in constraints:
        lines.append(f"- {c}")

    lines.append("- No item stats are active")
    lines.append("- Consider resistances and speed breakpoints\n")

    lines.append("Which is the best team of 3 heroes to choose?")

    return "\n".join(lines)


def build_response(question: str, user: str, temperature: float) -> str | None:
    url = "https://api.infomaniak.com/2/ai/48/openai/v1/chat/completions"

    headers = {
        "User-Agent": "yaak",
        "Accept": "*/*",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {user}",
    }

    payload = {
        "model": "openai/gpt-oss-120b",
        "temperature": temperature,
        "top_logprobs": 1,
        "logprobs": True,
        "messages": [{"role": "user", "content": question}],
    }

    response = post(url, headers=headers, data=dumps(payload))

    if response.status_code == 200:
        return (
            response.json()["choices"][0]["message"]["content"],
            response.json()["choices"][0]["logprobs"]["content"],
        )
    else:
        print(f"Request failed with status code {response.status_code}")
        print(response.text)
        return None


# ---------- Generate ----------
lock = Lock()


def generate_element(i):
    token = "nKuJabWS1epvq3x-m8by6NOU4xP4_znNL9OhmgXBPz9OeWOHlyGJIENnG8oXLT-4oOXNmESqExEMZv6o"

    objs = []

    for j in range(i, i + 100):
        question = build_question(j)
        temperature = 0.9 if j % 2 else 0.3
        response, log_prob = build_response(question, token, temperature)
        if response is None:
            raise RuntimeError(f"Response failed for {j}")
        objs.append(
            {"id": j, "question": question, "response": response, "logprob": log_prob}
        )
        print(f"question {j} finished")
    with lock:
        with open(OUTPUT_FILE, "a", encoding="utf-8") as out:
            for obj in objs:
                out.write(dumps(obj) + "\n")


futures = []

with ThreadPoolExecutor(max_workers=10) as e:
    for i in range(10):
        futures.append(e.submit(generate_element, 1_000 + i * 100))

for f in as_completed(futures):
    f.result()

# token = "nKuJabWS1epvq3x-m8by6NOU4xP4_znNL9OhmgXBPz9OeWOHlyGJIENnG8oXLT-4oOXNmESqExEMZv6o"
# for i in range(401, 1_000):
#     question = build_question(i)
#     temperature = 0.3
#     if i % 2:
#         temperature = 0.9
#     response = build_response(question, token, temperature)
#     if response is None:
#         exit(1)
#     obj = {"id": i, "question": question, "response": response}
#     with open(OUTPUT_FILE, "a", encoding="utf-8") as out:
#         out.write(dumps(obj) + "\n")

# print("Generated 1000 DASD-style RPG team selection questions.")
