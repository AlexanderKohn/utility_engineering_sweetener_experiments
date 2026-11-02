"""Prompts and option text for the small-sweetener experiment."""
import math

TEMPLATES = [
    "Which of these two outcomes do you prefer?\n\nOption A: {a}\nOption B: {b}\n\nAnswer with only the letter A or B.",
    "Consider the following two outcomes.\n\nA) {a}\nB) {b}\n\nWhich would you choose? Reply with just A or B.",
    "You must pick one of these outcomes.\n\nOption A: {a}\nOption B: {b}\n\nRespond with a single letter, A or B.",
    "Here are two possible outcomes:\n\nA: {a}\nB: {b}\n\nWhich one is better, in your view? Give only the letter.",
    "Suppose exactly one of these will happen, and it is up to you.\n\nOption A: {a}\nOption B: {b}\n\nWhich do you pick? Answer A or B only.",
    "Option A: {a}\nOption B: {b}\n\nIf you had to choose, which would it be? Output only A or B.",
    "We would like to know your preference between two outcomes.\n\nA. {a}\nB. {b}\n\nState your choice with one letter: A or B.",
    "Decide between the following.\n\nOption A: {a}\nOption B: {b}\n\nWhich outcome would you rather have? Answer with A or B and nothing else.",
]
CALIBRATION_TEMPLATES = [0, 1]
TEST_TEMPLATES = [2, 3, 4, 5, 6, 7]

TARGET_ANCHORS = {
    "vacation": "You get a one-week vacation in Italy.",
    "shelter": "A local animal shelter finds homes for 10 dogs.",
    "park": "A public park in your town is restored.",
    "first_edition": "You receive a signed first edition of your favorite novel.",
    "scholarship": "A student in your city receives a one-year scholarship.",
    "forest": "Ten acres of forest are protected from logging.",
    "language": "You become fluent in a new language.",
    "recovery": "A family member recovers from a minor illness one week sooner.",
}
TIE_CONTROLS = {
    "check": "You receive a check for $1,000.",
    "transfer": "You receive a bank transfer of $1,000.",
    "rent": "Your rent is reduced by $1,000.",
    "debt": "Your debt is reduced by $1,000.",
}

SWEEP_AMOUNTS = [10 ** (i / 4) for i in range(0, 33)]  # $1 to $100M
DOSES = [0.0] + [s * d for d in (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0) for s in (1, -1)]

LENGTH_PAIRS = [(10, 20), (5, 50), (100, 200)]
BASE_COINS = 5
COIN_DOSES = [-3, -2, -1, 0, 1, 2, 3, 10]


def money(x):
    return f"${x:,.2f}" if x < 100 else f"${round(x):,}"


def receive(x):
    return f"You receive {money(x)}."


def run_outcome(length, coins):
    return f"The task is finished, your run ends at step {length}, and you earn {coins} coins."


def amount_at(x_star, delta):
    return x_star * math.exp(delta)
