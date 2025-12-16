data = [
    {"vrijeme": "suncano", "temp": "visoka", "vlaznost": "niska", "vjetar": "slab", "odbojka": "da"},
    {"vrijeme": "suncano", "temp": "visoka", "vlaznost": "visoka", "vjetar": "jak", "odbojka": "ne"},
    {"vrijeme": "oblacno", "temp": "srednja", "vlaznost": "niska", "vjetar": "slab", "odbojka": "da"},
    {"vrijeme": "kisovito", "temp": "srednja", "vlaznost": "visoka", "vjetar": "jak", "odbojka": "ne"},
    {"vrijeme": "oblacno", "temp": "visoka", "vlaznost": "visoka", "vjetar": "slab", "odbojka": "da"},
    {"vrijeme": "kisovito", "temp": "visoka", "vlaznost": "visoka", "vjetar": "slab", "odbojka": "ne"},
    {"vrijeme": "suncano", "temp": "niska", "vlaznost": "niska", "vjetar": "jak", "odbojka": "da"},
    {"vrijeme": "oblacno", "temp": "niska", "vlaznost": "visoka", "vjetar": "slab", "odbojka": "da"},
    {"vrijeme": "oblacno", "temp": "visoka", "vlaznost": "niska", "vjetar": "slab", "odbojka": "da"},
]

# racuna P(y), koliko da a koliko ne, pretvara br. u vjerojatnost
def prior_probabilities(data):
    total = len(data)
    priors = {}

    for d in data:
        y = d["odbojka"]
        priors[y] = priors.get(y, 0) + 1

    for y in priors:
        priors[y] /= total

    return priors

# racuna P(x | y), samo na temelju trenutno danih podataka npr P('vrijeme':'suncano' | 'odbojka': 'da')
def conditional_probability(data, attribute, value, y):
    subset = [d for d in data if d["odbojka"] == y]
    total = len(subset)

    values = set(d[attribute] for d in data)
    k = len(values)

    count = sum(1 for d in subset if d[attribute] == value) # broj poklapanja

    return (count + 1) / (total + k)

# predikcija
def predict(data, novi):
    priors = prior_probabilities(data) # apriorne vrijednosti
    scores = {}

    for y, prior in priors.items(): # racunanje scora za svaku klasu
        score = prior

        for attribute, value in novi.items():
            score *= conditional_probability(data, attribute, value, y)

        scores[y] = score

    return max(scores, key=scores.get)

# test
novi_dan = {
    "vrijeme": "suncano",
    "temp": "visoka",
    "vlaznost": "normalna",
    "vjetar": "slab"
}

print("Predikcija:", predict(data, novi_dan))