function newSurprise = computeSurprise(MU)

E = entropy(MU.T);
zeta = sum(MU.TCounts,3);
psi = sum(zeta(:));

temp = E.*zeta./psi;
newSurprise = sum(temp(:));

