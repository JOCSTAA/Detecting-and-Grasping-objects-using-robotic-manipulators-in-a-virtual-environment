from random import *
possiblemulti = [[1.2,-0.1],[1.2, 0.1],[1.1,0],[1,-0.1],[1,0.1]]
possibleyz = [-0.1, 0, 0.1]

explored = []
zebras = 0
while zebras < 5:
    posx = (sample(possiblemulti, 1))[0]
    posz = (sample(possibleyz, 1))[0]
    pasta = 0
    for val in explored:
        if posx == val:
            pasta = 1

    if pasta==0:
        explored.append(posx)
        test = [posx[0],posx[1],posz]

        print(test)