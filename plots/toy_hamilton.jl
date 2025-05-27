H = [
    1 2 100
    2 1 0
    100 0 1000
]

states, Es = spectrum(H)

display(states)
display(Es)


_, E12 = spectrum(H[1:2, 1:2])

_, E13 = spectrum(H[1:2:3, 1:2:3])

@show E12 E13