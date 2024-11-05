import math as mt

# elaborando os dados iniciais

n = 11
r = 2

# encontrando o espaço amostral
EA = mt.comb(n, r)
print("Espaço Amostral =", EA)


# selecionar bolas

B = mt.comb(6, 1)
P = mt.comb (5, 1)

print("Brancas =", B)
print("Pretas =", P)


# resultado

Prob = (B * P)/ (EA)
Prob = Prob *100
Prob = round(Prob, 2)


print("Probabilidade =", Prob, "%")