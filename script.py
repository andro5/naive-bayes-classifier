data = [
    {'vrijeme':'suncano','temp':'visoka','vlaznost':'visoka','vjetar':'slab','y':'ne'},
    {'vrijeme':'suncano','temp':'visoka','vlaznost':'visoka','vjetar':'jak','y':'ne'},
    {'vrijeme':'oblacno','temp':'visoka','vlaznost':'visoka','vjetar':'slab','y':'da'},
    {'vrijeme':'kisno','temp':'srednja','vlaznost':'visoka','vjetar':'slab','y':'da'},
    {'vrijeme':'kisno','temp':'niska','vlaznost':'normalna','vjetar':'slab','y':'da'},
    {'vrijeme':'kisno','temp':'niska','vlaznost':'normalna','vjetar':'jak','y':'ne'},
    {'vrijeme':'oblacno','temp':'niska','vlaznost':'normalna','vjetar':'jak','y':'da'},
    {'vrijeme':'suncano','temp':'srednja','vlaznost':'visoka','vjetar':'slab','y':'ne'},
    {'vrijeme':'suncano','temp':'niska','vlaznost':'normalna','vjetar':'slab','y':'da'},
    {'vrijeme':'kisno','temp':'srednja','vlaznost':'normalna','vjetar':'slab','y':'da'},
    {'vrijeme':'suncano','temp':'srednja','vlaznost':'normalna','vjetar':'jak','y':'da'},
    {'vrijeme':'oblacno','temp':'srednja','vlaznost':'visoka','vjetar':'jak','y':'da'},
    {'vrijeme':'oblacno','temp':'visoka','vlaznost':'normalna','vjetar':'slab','y':'da'},
    {'vrijeme':'kisno','temp':'srednja','vlaznost':'visoka','vjetar':'jak','y':'ne'}
]

# funkcija koja broji koliko puta se vrijednost pojavljuje u datasetu
def count_where(dataset, atribut, vrijednost, klasa=None):
    broj = 0
    for d in dataset:
        if klasa is None:
            if d[atribut] == vrijednost:
                broj += 1
        else:
            if d[atribut] == vrijednost and d['y'] == klasa:
                broj += 1
    return broj

# treniranje modela naive bayes
def train(dataset):
    model = {}
    total = len(dataset)

    # pronadji klase
    klase = []
    for d in dataset:
        if d['y'] not in klase:
            klase.append(d['y'])

    # apriorne vjerojatnosti
    priors = {}
    for c in klase:
        broj_c = 0
        for d in dataset:
            if d['y'] == c:
                broj_c += 1
        priors[c] = broj_c / total

    # atributi
    attributes = []
    for k in dataset[0]:
        if k != 'y':
            attributes.append(k)

    # vrijednosti atributa (skup vrijednosti)
    values = {}
    for a in attributes:
        values[a] = []
        for d in dataset:
            if d[a] not in values[a]:
                values[a].append(d[a])

    # uvjetne vjerojatnosti
    cond_probs = {}
    for a in attributes:
        cond_probs[a] = {}
        for c in klase:
            cond_probs[a][c] = {}
            count_c = count_where(dataset, 'y', c)
            k = len(values[a])
            for v in values[a]:
                count_avc = count_where(dataset, a, v, c)
                cond_probs[a][c][v] = (count_avc + 1) / (count_c + k)

    model['priors'] = priors
    model['cond_probs'] = cond_probs
    model['attributes'] = attributes
    model['values'] = values
    model['classes'] = klase
    return model

model = train(data)

# predikcija bez logaritama
def predict(novi):
    best_class = None
    best_prob = -1

    for c in model['classes']:
        # pocinjemo od apriorne vjerojatnosti
        prob = model['priors'][c]

        # mnozimo uvjetne vjerojatnosti
        for a in model['attributes']:
            v = novi[a]
            if v in model['cond_probs'][a][c]:
                prob *= model['cond_probs'][a][c][v]
            else:
                prob *= 0.000001  # minimalna vrijednost

        # odaberi najvecu probabilnost
        if prob > best_prob:
            best_prob = prob
            best_class = c

    return best_class

# primjer podatka
novi_podatak = {
    'vrijeme': 'suncano',
    'temp': 'visoka',
    'vlaznost': 'normalna',
    'vjetar': 'slab'
}

print("predikcija:", predict(novi_podatak))