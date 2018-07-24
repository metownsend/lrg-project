def sumNsat(Nsat, z_LRG, rmag_LRG, gmag_LRG, zmag_LRG):

    import numpy as np

    sumsat = []

# 	Sum up number of background galaxies for every LRG
    for i in range(len(Nsat)):
        sumsat.append(np.sum(Nsat[i]))
# 		
    # Divvy up by redshift slice

    Nsat1z = Nsat[np.where(z_LRG < 0.2)]
    # print(len(Nsat1z))

    sumsat1z = []
    for i in range(len(Nsat1z)):
        sumsat1z.append(np.sum(Nsat1z[i]))

    # 0.2 <= z < 0.3
    Nsat2z = Nsat[np.where((z_LRG >= 0.2) & (0.3 > z_LRG))]
    # print(len(Nsat2z))

    sumsat2z = []
    for i in range(len(Nsat2z)):
        sumsat2z.append(np.sum(Nsat2z[i]))

    # 0.3 <= z < 0.4
    Nsat3z = Nsat[np.where((z_LRG >= 0.3) & (0.4 > z_LRG))]
    # print(len(Nsat3z))

    sumsat3z = []
    for i in range(len(Nsat3z)):
        sumsat3z.append(np.sum(Nsat3z[i]))

    # 0.4 <= z < 0.5
    Nsat4z = Nsat[np.where((z_LRG >= 0.4) & (0.5 > z_LRG))]
    # print(len(Nsat4z))

    sumsat4z = []
    for i in range(len(Nsat4z)):
        sumsat4z.append(np.sum(Nsat4z[i]))

    # 0.5 <= z < 0.6
    Nsat5z = Nsat[np.where((z_LRG >= 0.5) & (0.6 > z_LRG))]
    # print(len(Nsat5z))

    sumsat5z = []
    for i in range(len(Nsat5z)):
        sumsat5z.append(np.sum(Nsat5z[i]))

    # 0.6 <= z < 0.7
    Nsat6z = Nsat[np.where((z_LRG >= 0.6) & (0.7 > z_LRG))]
    # print(len(Nsat6z))

    sumsat6z = []
    for i in range(len(Nsat6z)):
        sumsat6z.append(np.sum(Nsat6z[i]))

    # z >= 0.7
    Nsat7z = Nsat[np.where(z_LRG >= 0.7)]
    # print(len(Nsat7z))

    sumsat7z = []
    for i in range(len(Nsat7z)):
        sumsat7z.append(np.sum(Nsat7z[i]))


    # Divvy up Nsat by rmag slice

    rmag_LRG = np.array(rmag_LRG)

    # bins of ~1 mag

    # 15 <= rmag < 16
    Nsat1r = Nsat[np.where((rmag_LRG >= 15.) & (16. > rmag_LRG))]
    # print(len(Nsat1))

    sumsat1r = []
    for i in range(len(Nsat1r)):
        sumsat1r.append(np.sum(Nsat1r[i]))

    # 16 <= rmag < 17
    Nsat2r = Nsat[np.where((rmag_LRG >= 16.) & (17. > rmag_LRG))]
    # print(len(Nsat2))

    sumsat2r = []
    for i in range(len(Nsat2r)):
        sumsat2r.append(np.sum(Nsat2r[i]))

    # 17 <= rmag < 18
    Nsat3r = Nsat[np.where((rmag_LRG >= 17.) & (18. > rmag_LRG))]
    # print(len(Nsat3))

    sumsat3r = []
    for i in range(len(Nsat3r)):
        sumsat3r.append(np.sum(Nsat3r[i]))

    # 18 <= rmag < 19
    Nsat4r = Nsat[np.where((rmag_LRG >= 18.) & (19. > rmag_LRG))]
    # print(len(Nsat4))

    sumsat4r = []
    for i in range(len(Nsat4r)):
        sumsat4r.append(np.sum(Nsat4r[i]))

    # 19 <= rmag < 20
    Nsat5r = Nsat[np.where((rmag_LRG >= 19.) & (20. > rmag_LRG))]
    # print(len(Nsat5))

    sumsat5r = []
    for i in range(len(Nsat5r)):
        sumsat5r.append(np.sum(Nsat5r[i]))

    # 20 <= rmag < 21
    Nsat6r = Nsat[np.where((rmag_LRG >= 20.) & (21. > rmag_LRG))]
    # print(len(Nsat6))

    sumsat6r = []
    for i in range(len(Nsat6r)):
        sumsat6r.append(np.sum(Nsat6r[i]))

    # rmag >= 21
    Nsat7r = Nsat[np.where(rmag_LRG >= 21.)]
    # print(len(Nsat7))

    sumsat7r = []
    for i in range(len(Nsat7r)):
        sumsat7r.append(np.sum(Nsat7r[i]))


    # Divvy up Nsat by gmag slice

    gmag_LRG = np.array(gmag_LRG)

    # bins of ~1 mag

    # 16 <= gmag < 17
    Nsat1g = Nsat[np.where((gmag_LRG >= 16.) & (17. > gmag_LRG))]
    # print(len(Nsat1g))

    sumsat1g = []
    for i in range(len(Nsat1g)):
        sumsat1g.append(np.sum(Nsat1g[i]))

    # 17 <= gmag < 18
    Nsat2g = Nsat[np.where((gmag_LRG >= 17.) & (18. > gmag_LRG))]
    # print(len(Nsat2g))

    sumsat2g = []
    for i in range(len(Nsat2g)):
        sumsat2g.append(np.sum(Nsat2g[i]))

    # 18 <= gmag < 19
    Nsat3g = Nsat[np.where((gmag_LRG >= 18.) & (19. > gmag_LRG))]
    # print(len(Nsat3g))

    sumsat3g = []
    for i in range(len(Nsat3g)):
        sumsat3g.append(np.sum(Nsat3g[i]))

    # 19 <= gmag < 20
    Nsat4g = Nsat[np.where((gmag_LRG >= 19.) & (20. > gmag_LRG))]
    # print(len(Nsat4g))

    sumsat4g = []
    for i in range(len(Nsat4g)):
        sumsat4g.append(np.sum(Nsat4g[i]))

    # 20 <= gmag < 21
    Nsat5g = Nsat[np.where((gmag_LRG >= 20.) & (21. > gmag_LRG))]
    # print(len(Nsat5g))

    sumsat5g = []
    for i in range(len(Nsat5g)):
        sumsat5g.append(np.sum(Nsat5g[i]))

    # 21 <= gmag < 22
    Nsat6g = Nsat[np.where((gmag_LRG >= 21.) & (22. > gmag_LRG))]
    # print(len(Nsat6g))

    sumsat6g = []
    for i in range(len(Nsat6g)):
        sumsat6g.append(np.sum(Nsat6g[i]))

    # 22 <= gmag < 23
    Nsat7g = Nsat[np.where((gmag_LRG >= 22.) & (23. > gmag_LRG))]
    # print(len(Nsat7g))

    sumsat7g = []
    for i in range(len(Nsat7g)):
        sumsat7g.append(np.sum(Nsat7g[i]))

    # gmag >= 23
    Nsat8g = Nsat[np.where(gmag_LRG >= 23.)]
    # print(len(Nsat8))

    sumsat8g = []
    for i in range(len(Nsat8g)):
        sumsat8g.append(np.sum(Nsat8g[i]))



    # Divvy up Nsat by zmag slice

    zmag_LRG = np.array(zmag_LRG)

    # bins of ~1 mag

    # 14 <= zmag < 15
    Nsat1_zmag = Nsat[np.where((zmag_LRG >= 14.) & (15. > zmag_LRG))]
    # print(len(Nsat1g))

    sumsat1_zmag = []
    for i in range(len(Nsat1_zmag)):
        sumsat1_zmag.append(np.sum(Nsat1_zmag[i]))

    # 15 <= zmag < 16
    Nsat2_zmag = Nsat[np.where((zmag_LRG >= 15.) & (16. > zmag_LRG))]
    # print(len(Nsat2g))

    sumsat2_zmag = []
    for i in range(len(Nsat2_zmag)):
        sumsat2_zmag.append(np.sum(Nsat2_zmag[i]))

    # 16 <= zmag < 17
    Nsat3_zmag = Nsat[np.where((zmag_LRG >= 16.) & (17. > zmag_LRG))]
    # print(len(Nsat3g))

    sumsat3_zmag = []
    for i in range(len(Nsat3_zmag)):
        sumsat3_zmag.append(np.sum(Nsat3_zmag[i]))

    # 17 <= zmag < 18
    Nsat4_zmag = Nsat[np.where((zmag_LRG >= 17.) & (18. > zmag_LRG))]
    # print(len(Nsat4z))

    sumsat4_zmag = []
    for i in range(len(Nsat4_zmag)):
        sumsat4_zmag.append(np.sum(Nsat4_zmag[i]))

    # 18 <= zmag < 19
    Nsat5_zmag = Nsat[np.where((zmag_LRG >= 18.) & (19. > zmag_LRG))]
    # print(len(Nsat5g))

    sumsat5_zmag = []
    for i in range(len(Nsat5_zmag)):
        sumsat5_zmag.append(np.sum(Nsat5_zmag[i]))

    # 19 <= zmag < 20
    Nsat6_zmag = Nsat[np.where((zmag_LRG >= 19.) & (20. > zmag_LRG))]
    # print(len(Nsat6g))

    sumsat6_zmag = []
    for i in range(len(Nsat6_zmag)):
        sumsat6_zmag.append(np.sum(Nsat6_zmag[i]))

    # zmag < 20
    Nsat7_zmag = Nsat[np.where(zmag_LRG >= 20.)]
    # print(len(Nsat8))

    sumsat7_zmag = []
    for i in range(len(Nsat7_zmag)):
        sumsat7_zmag.append(np.sum(Nsat7_zmag[i]))


    return(sumsat, sumsat1z, sumsat2z, sumsat3z, sumsat4z, sumsat5z, sumsat6z, sumsat7z, sumsat1r, sumsat2r, sumsat3r, sumsat4r, sumsat5r, sumsat6r, sumsat7r, sumsat1g, sumsat2g, sumsat3g, sumsat4g, sumsat5g, sumsat6g, sumsat7g, sumsat8g, sumsat1_zmag, sumsat2_zmag, sumsat3_zmag, sumsat4_zmag, sumsat5_zmag, sumsat6_zmag, sumsat7_zmag)



def sumNN(near, z_LRG, rmag_LRG, gmag_LRG, zmag_LRG):

    import numpy as np

    near = np.array(near)

    sumnear = []

# 	Sum up number of background galaxies for every LRG
    for i in range(len(near)):
        sumnear.append(np.sum(near[i]))

    # bins of ~0.1

    # z < 0.2
    near1z = near[np.where(z_LRG < 0.2)]
    # print(len(Nsat1))

    sumnear1z = []
    for i in range(len(near1z)):
        sumnear1z.append(np.sum(near1z[i]))

    # 0.2 <= z < 0.3
    near2z = near[np.where((z_LRG >= 0.2) & (0.3 > z_LRG))]
    # print(len(Nsat2))

    sumnear2z = []
    for i in range(len(near2z)):
        sumnear2z.append(np.sum(near2z[i]))

    # 0.3 <= z < 0.4
    near3z = near[np.where((z_LRG >= 0.3) & (0.4 > z_LRG))]
    # print(len(Nsat3))

    sumnear3z = []
    for i in range(len(near3z)):
        sumnear3z.append(np.sum(near3z[i]))

    # 0.4 <= z < 0.5
    near4z = near[np.where((z_LRG >= 0.4) & (0.5 > z_LRG))]
    # print(len(Nsat4))

    sumnear4z = []
    for i in range(len(near4z)):
        sumnear4z.append(np.sum(near4z[i]))

    # 0.5 <= z < 0.6
    near5z = near[np.where((z_LRG >= 0.5) & (0.6 > z_LRG))]
    # print(len(Nsat5))

    sumnear5z = []
    for i in range(len(near5z)):
        sumnear5z.append(np.sum(near5z[i]))

    # 0.6 <= z < 0.7
    near6z = near[np.where((z_LRG >= 0.6) & (0.7 > z_LRG))]
    # print(len(Nsat6))

    sumnear6z = []
    for i in range(len(near6z)):
        sumnear6z.append(np.sum(near6z[i]))

    # z >= 0.7
    near7z = near[np.where(z_LRG >= 0.7)]
    # print(len(Nsat7))

    sumnear7z = []
    for i in range(len(near7z)):
        sumnear7z.append(np.sum(near7z[i]))


    rmag_LRG = np.array(rmag_LRG)

    # bins of ~1 mag

    # 15 <= rmag < 16
    near1r = near[np.where((rmag_LRG >= 15.) & (16. > rmag_LRG))]
    # print(len(Nsat1))

    sumnear1r = []
    for i in range(len(near1r)):
        sumnear1r.append(np.sum(near1r[i]))

    # 16 <= rmag < 17
    near2r = near[np.where((rmag_LRG >= 16.) & (17. > rmag_LRG))]
    # print(len(Nsat2))

    sumnear2r = []
    for i in range(len(near2r)):
        sumnear2r.append(np.sum(near2r[i]))

    # 17 <= rmag < 18
    near3r = near[np.where((rmag_LRG >= 17.) & (18. > rmag_LRG))]
    # print(len(Nsat3))

    sumnear3r = []
    for i in range(len(near3r)):
        sumnear3r.append(np.sum(near3r[i]))

    # 18 <= rmag < 19
    near4r = near[np.where((rmag_LRG >= 18.) & (19. > rmag_LRG))]
    # print(len(Nsat4))

    sumnear4r = []
    for i in range(len(near4r)):
        sumnear4r.append(np.sum(near4r[i]))

    # 19 <= rmag < 20
    near5r = near[np.where((rmag_LRG >= 19.) & (20. > rmag_LRG))]
    # print(len(Nsat5))

    sumnear5r = []
    for i in range(len(near5r)):
        sumnear5r.append(np.sum(near5r[i]))

    # 20 <= rmag < 21
    near6r = near[np.where((rmag_LRG >= 20.) & (21. > rmag_LRG))]
    # print(len(Nsat6))

    sumnear6r = []
    for i in range(len(near6r)):
        sumnear6r.append(np.sum(near6r[i]))

    # rmag >= 21
    near7r = near[np.where(rmag_LRG >= 21.)]
    # print(len(Nsat7))

    sumnear7r = []
    for i in range(len(near7r)):
        sumnear7r.append(np.sum(near7r[i]))


    gmag_LRG = np.array(gmag_LRG)

    # bins of ~1 mag

    # 16 <= gmag < 17
    near1g = near[np.where((gmag_LRG >= 16.) & (17. > gmag_LRG))]
    # print(len(Nsat1))

    sumnear1g = []
    for i in range(len(near1g)):
        sumnear1g.append(np.sum(near1g[i]))

    # 17 <= gmag < 18
    near2g = near[np.where((gmag_LRG >= 17.) & (18. > gmag_LRG))]
    # print(len(near2g))

    sumnear2g = []
    for i in range(len(near2g)):
        sumnear2g.append(np.sum(near2g[i]))
    # print(len(sumnear2g))

    # 18 <= gmag < 19
    near3g = near[np.where((gmag_LRG >= 18.) & (19. > gmag_LRG))]
    # print(len(Nsat3))

    sumnear3g = []
    for i in range(len(near3g)):
        sumnear3g.append(np.sum(near3g[i]))

    # 19 <= gmag < 20
    near4g = near[np.where((gmag_LRG >= 19.) & (20. > gmag_LRG))]
    # print(len(Nsat4))

    sumnear4g = []
    for i in range(len(near4g)):
        sumnear4g.append(np.sum(near4g[i]))

    # 20 <= gmag < 21
    near5g = near[np.where((gmag_LRG >= 20.) & (21. > gmag_LRG))]
    # print(len(Nsat5))

    sumnear5g = []
    for i in range(len(near5g)):
        sumnear5g.append(np.sum(near5g[i]))

    # 21 <= gmag < 22
    near6g = near[np.where((gmag_LRG >= 21.) & (22. > gmag_LRG))]
    # print(len(Nsat6))

    sumnear6g = []
    for i in range(len(near6g)):
        sumnear6g.append(np.sum(near6g[i]))

    # 22 <= gmag < 23
    near7g = near[np.where((gmag_LRG >= 22.) & (23. > gmag_LRG))]
    # print(len(Nsat7))

    sumnear7g = []
    for i in range(len(near7g)):
        sumnear7g.append(np.sum(near7g[i]))

    # gmag >= 23
    near8g = near[np.where(gmag_LRG >= 23.)]
    # print(len(Nsat8))

    sumnear8g = []
    for i in range(len(near8g)):
        sumnear8g.append(np.sum(near8g[i]))


    zmag_LRG = np.array(zmag_LRG)

    # bins of ~1 mag

    # 14 <= zmag < 15
    near1_zmag = near[np.where((zmag_LRG >= 14.) & (15. > zmag_LRG))]
    # print(len(Nsat1))

    sumnear1_zmag = []
    for i in range(len(near1_zmag)):
        sumnear1_zmag.append(np.sum(near1_zmag[i]))

    # 15 <= zmag < 16
    near2_zmag = near[np.where((zmag_LRG >= 15.) & (16. > zmag_LRG))]
    # print(len(near2g))

    sumnear2_zmag = []
    for i in range(len(near2_zmag)):
        sumnear2_zmag.append(np.sum(near2_zmag[i]))
    # print(len(sumnear2g))

    # 16 <= zmag < 17
    near3_zmag = near[np.where((zmag_LRG >= 16.) & (17. > zmag_LRG))]
    # print(len(Nsat3))

    sumnear3_zmag = []
    for i in range(len(near3_zmag)):
        sumnear3_zmag.append(np.sum(near3_zmag[i]))

    # 17 <= zmag < 18
    near4_zmag = near[np.where((zmag_LRG >= 17.) & (18. > zmag_LRG))]
    # print(len(Nsat4))

    sumnear4_zmag = []
    for i in range(len(near4_zmag)):
        sumnear4_zmag.append(np.sum(near4_zmag[i]))

    # 18 <= zmag < 19
    near5_zmag = near[np.where((zmag_LRG >= 18.) & (19. > zmag_LRG))]
    # print(len(Nsat5))

    sumnear5_zmag = []
    for i in range(len(near5_zmag)):
        sumnear5_zmag.append(np.sum(near5_zmag[i]))

    # 19 <= zmag < 20
    near6_zmag = near[np.where((zmag_LRG >= 19.) & (20. > zmag_LRG))]
    # print(len(Nsat5))

    sumnear6_zmag = []
    for i in range(len(near6_zmag)):
        sumnear6_zmag.append(np.sum(near6_zmag[i]))

    # zmag >= 20
    near7_zmag = near[np.where(zmag_LRG >= 20.)]
    # print(len(Nsat8))

    sumnear7_zmag = []
    for i in range(len(near7_zmag)):
        sumnear7_zmag.append(np.sum(near7_zmag[i]))

    return(sumnear, sumnear1z, sumnear2z, sumnear3z, sumnear4z, sumnear5z, sumnear6z, sumnear7z, sumnear1r, sumnear2r, sumnear3r, sumnear4r, sumnear5r, sumnear6r, sumnear7r, sumnear1g, sumnear2g, sumnear3g, sumnear4g, sumnear5g, sumnear6g, sumnear7g, sumnear8g, sumnear1_zmag, sumnear2_zmag, sumnear3_zmag, sumnear4_zmag, sumnear5_zmag, sumnear6_zmag, sumnear7_zmag)




# def sumNbkg(Nbkg, z_LRG, rmag_LRG, gmag_LRG, zmag_LRG):	
# 
# 	import numpy as np
# 		
# 	# Sum up number of background galaxies for every LRG
# 	for i in range(len(Nbkg)):
# 		sumbkg.append(np.sum(Nbkg[i]))
# 	
# 	sumbkg = []


