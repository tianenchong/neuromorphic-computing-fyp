rfis = 250
s = 1.5
s_arb = 1

rf3 = rfi = 1000 * rfis  
rf3s = rf3*s

rfo2 = rf3s * s_arb
print("{:} {:} {:}".format(rfi, rfo2, rfis))