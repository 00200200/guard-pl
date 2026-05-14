ALLOW = "Prompt zaklasyfikowany jako neutralny — brak sygnałów ryzyka."


def explain(conf):
    return f"Klasyfikator ocenił prompt jako szkodliwy — pewność {conf:.0%}."
